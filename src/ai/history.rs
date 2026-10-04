//! 公共历史记忆：像人类一样"记住弃牌堆里有什么"。
//!
//! 规则上弃牌堆是**完全公开**的（每张牌进弃牌堆都明置，引擎里 `known_by = 全体`），
//! 人也确实看得见——每张牌落到弃牌堆时都显示在界面上。
//! 但 `PlayerView` 只暴露 `discard_top` 与 `discard_count`，所以只读视图的 Bot
//! 实际上"忘了"整个弃牌堆（实测占全部看不见的牌 **36%**）。
//! 这会直接毁掉"我要不要宣告 Cabo"的置信度：机器人无法排除"对手手里还有一张 0"。
//!
//! 这里从**公开日志**里增量重建弃牌堆构成，并用可观测的 `discard_count` / `discard_top`
//! 做自校验与修复。可恢复的信息：
//! - 摸牌后弃置：`X 弃置了 5`（点数公开）
//! - 多人交换：`翻开槽位 1、2：5 5 —— ...`（点数公开）
//! - 从弃牌堆取牌：`... 换入弃牌堆顶的 7`（**同时暴露了原先未知的那张牌**）
//! - 单人交换换出的牌：点数不在日志里（只有一个 `None` 占位）
//!
//! 记忆按座位保存，带"牌局指纹"与自校验，串场/过期会自动重置，不会污染决策。

use std::collections::HashMap;
use std::sync::Mutex;

use crate::game::view::PlayerView;
use crate::game::RANK_COPIES;

/// 弃牌堆重建结果：`known[r]` = 弃牌堆里（不含堆顶）我已知点数的张数。
#[derive(Clone, Debug, Default)]
pub struct DiscardInfo {
    pub known: [u16; 14],
    /// 弃牌堆里我不知道点数的张数（不含堆顶）。
    pub unknown: usize,
}

struct Memory {
    round: u32,
    /// 弃牌堆（底→顶）；`None` = 我不知道点数。stack[0] 是起始弃牌。
    stack: Vec<Option<u8>>,
    /// 上一批日志窗口（最新在前），用于增量找出新条目。
    window: Vec<String>,
    /// 最近一位"刚从摸牌堆抽牌"的玩家名（用于判断多人交换的来源）。
    last_drawer: Option<String>,
}

impl Memory {
    fn reset(&mut self, round: u32, first: Option<u8>) {
        self.round = round;
        self.stack = vec![first];
        self.window.clear();
        self.last_drawer = None;
    }
}

/// 每个座位一份记忆。
#[derive(Default)]
pub struct History {
    mem: Mutex<HashMap<usize, Memory>>,
    /// (名字指纹, 人数) 用于识别换局/串场。
    fingerprint: Mutex<Option<(u64, usize)>>,
}

fn name_hash(names: &[String]) -> u64 {
    // FNV-1a，够用且无需依赖
    let mut h: u64 = 0xcbf29ce484222325;
    for n in names {
        for b in n.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    /// 观察一次视图，返回该座位视角下重建出的弃牌堆信息。
    pub fn observe(&self, view: &PlayerView) -> DiscardInfo {
        let me = match view.me {
            Some(m) => m,
            None => return DiscardInfo::default(),
        };
        let names: Vec<String> = view.all_seats.iter().map(|s| s.name.clone()).collect();
        let fp = (name_hash(&names), names.len());
        {
            let mut f = self.fingerprint.lock().unwrap();
            match *f {
                Some(prev) if prev != fp => {
                    // 换了一局（不同玩家集合）：整体清空
                    self.mem.lock().unwrap().clear();
                    *f = Some(fp);
                }
                None => *f = Some(fp),
                _ => {}
            }
        }

        let mut mems = self.mem.lock().unwrap();
        let mem = mems.entry(me).or_insert_with(|| {
            let mut m = Memory { round: 0, stack: Vec::new(), window: Vec::new(), last_drawer: None };
            m.reset(view.round_no, view.discard_top);
            m
        });

        // 新一轮 / 换局：重置
        if mem.round != view.round_no {
            mem.reset(view.round_no, view.discard_top);
        }

        // 日志窗口（最新在前）→ 找出新增条目
        let now: Vec<String> = view.log.iter().map(|l| l.text.clone()).collect();
        let old = std::mem::take(&mut mem.window);
        let fresh = new_entries(&old, &now);
        mem.window = now;
        // 按时间顺序处理新条目
        for text in fresh.iter().rev() {
            apply_line(mem, text);
        }

        // 自校验 + 修复
        repair(mem, view.discard_count, view.discard_top);

        // 统计（不含堆顶）
        let mut info = DiscardInfo::default();
        let n = mem.stack.len();
        for entry in mem.stack.iter().take(n.saturating_sub(1)) {
            match entry {
                Some(r) => {
                    if (*r as usize) < 14 {
                        info.known[*r as usize] += 1;
                    }
                }
                None => info.unknown += 1,
            }
        }
        info
    }
}

/// 找出"上一批窗口之后新增"的日志条目（返回最新在前的顺序）。
///
/// 视图里的日志是"最新在前、最多 30 条"的滑动窗口，所以新窗口的前缀是新条目、
/// 后缀与旧窗口的前缀重合。这里找出最长的重合长度。
fn new_entries(old: &[String], now: &[String]) -> Vec<String> {
    if old.is_empty() {
        // 首次观察：窗口里的历史无法安全重放（可能已被截断），只取"轮开始"那一条之后的。
        return now.to_vec();
    }
    let max_overlap = old.len().min(now.len());
    let mut overlap = 0usize;
    for m in (1..=max_overlap).rev() {
        if now[now.len() - m..] == old[..m] {
            overlap = m;
            break;
        }
    }
    if overlap == 0 {
        // 窗口完全滑走：无法增量，只能放弃这批（交给 repair 兜底）
        return Vec::new();
    }
    now[..now.len() - overlap].to_vec()
}

/// 解析一条日志，更新弃牌堆栈。

/// 取出 `marker` 之后的一串点数（空格分隔），直到遇到非点数内容。
fn parse_ranks_after(text: &str, marker: &str) -> Vec<u8> {
    let Some(pos) = text.find(marker) else { return Vec::new() };
    let tail = &text[pos + marker.len()..];
    let mut out = Vec::new();
    for tok in tail.split_whitespace() {
        match tok.parse::<u8>() {
            Ok(r) if r <= 13 => out.push(r),
            _ => break,
        }
    }
    out
}


/// 解析"亮出"后面跟着的点数。日志有**两种**格式：
///
/// 1. 弃牌堆来源：`P2 亮出槽位 0 的 10，换入弃牌堆顶的 9`
/// 2. 摸牌堆来源：`P2 把刚摸的牌与槽位 0 交换，亮出 10`
///
/// 注意点数后面常常**紧跟中文标点而没有空格**（`10，换入…`），
/// 所以不能按空格分词，要按"数字段 + 分隔符（空格/、/，）"扫。
fn parse_revealed_ranks(text: &str) -> Vec<u8> {
    // 格式 1：亮出槽位 … 的 <点数>
    let after = if let Some(pos) = text.find("亮出槽位 ") {
        let tail = &text[pos + "亮出槽位 ".len()..];
        tail.find(" 的 ").map(|d| &tail[d + " 的 ".len()..])
    } else if let Some(pos) = text.find("亮出 ") {
        // 格式 2：… 亮出 <点数>
        Some(&text[pos + "亮出 ".len()..])
    } else {
        None
    };
    let Some(after) = after else { return Vec::new() };

    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in after.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if ch == ' ' || ch == '\u{3001}' || ch == '\u{ff0c}' {
            if !cur.is_empty() {
                if let Ok(r) = cur.parse::<u8>() {
                    if r <= 13 {
                        out.push(r);
                    }
                }
                cur.clear();
            }
            if ch != ' ' {
                break;
            }
        } else {
            if !cur.is_empty() {
                if let Ok(r) = cur.parse::<u8>() {
                    if r <= 13 {
                        out.push(r);
                    }
                }
            }
            break;
        }
    }
    out
}

fn apply_line(mem: &mut Memory, text: &str) {
    // 新一轮：重置（保留轮次由 observe 处理，这里只重置弃牌堆内容）
    if let Some(pos) = text.find("轮开始：每人 4 张暗牌，弃牌堆起始为 ") {
        if text.starts_with('第') && pos < 8 {
            let tail = &text[pos + "轮开始：每人 4 张暗牌，弃牌堆起始为 ".len()..];
            let first = tail.trim().parse::<u8>().ok();
            mem.stack = vec![first];
            mem.last_drawer = None;
            return;
        }
    }
    // 抽牌（记录是谁，用于判断多人交换的来源）
    if let Some(pos) = text.find(" 从摸牌堆抽了一张牌") {
        mem.last_drawer = Some(text[..pos].to_string());
        return;
    }
    // 摸牌后弃置：点数公开
    if let Some(pos) = text.rfind("弃置了 ") {
        if let Ok(r) = text[pos + "弃置了 ".len()..].trim().parse::<u8>() {
            mem.stack.push(Some(r));
            mem.last_drawer = None;
            return;
        }
    }
    // 单人交换：从弃牌堆顶取走一张（点数公开）；换出的牌**正面朝上**进入弃牌堆，
    // 引擎日志现在会写出来（`亮出 a b`），这里据实记账。
    if let Some(pos) = text.find("换入弃牌堆顶的 ") {
        if let Ok(r) = text[pos + "换入弃牌堆顶的 ".len()..].trim().parse::<u8>() {
            pop_expect(mem, Some(r));
            let outs = parse_revealed_ranks(text);
            if outs.is_empty() {
                mem.stack.push(None);
            } else {
                for out in outs {
                    mem.stack.push(Some(out));
                }
            }
            mem.last_drawer = None;
            return;
        }
    }
    // 单人交换（摸牌堆来源）：换出的那张同样正面朝上进入弃牌堆
    if text.contains("把刚摸的牌与槽位") {
        let outs = parse_revealed_ranks(text);
        if outs.is_empty() {
            mem.stack.push(None);
        } else {
            for out in outs {
                mem.stack.push(Some(out));
            }
        }
        mem.last_drawer = None;
        return;
    }
    // 多人交换：日志里带出被翻开的点数
    if text.contains("—— 点数相同，交换成功") || text.contains("—— 存在不相同的牌") {
        let from_discard = if text.contains("从弃牌堆获得") {
            true
        } else if text.contains("从摸牌堆获得") {
            false
        } else {
            // 成功行不写来源：靠"刚才是谁摸的牌"判断
            let who = text.split(" 翻开槽位").next().unwrap_or("");
            mem.last_drawer.as_deref() != Some(who)
        };
        let ranks = parse_ranks_after_colon(text);
        if text.contains("交换成功") {
            if from_discard {
                pop_any(mem);
            }
            for r in ranks {
                mem.stack.push(Some(r));
            }
        } else {
            // 失败：牌留在手上（明置），只有"从弃牌堆获得"才会拿走堆顶
            if from_discard {
                pop_any(mem);
            }
        }
        mem.last_drawer = None;
    }
}

/// 取 `：` 与 ` ——` 之间的点数列表（多人交换行）。
fn parse_ranks_after_colon(text: &str) -> Vec<u8> {
    let Some(start) = text.find('：') else { return Vec::new() };
    let rest = &text[start + '：'.len_utf8()..];
    let end = rest.find(" ——").unwrap_or(rest.len());
    rest[..end].split_whitespace().filter_map(|t| t.parse::<u8>().ok()).collect()
}

fn pop_expect(mem: &mut Memory, expect: Option<u8>) {
    if let Some(top) = mem.stack.pop() {
        // 如果弹出的与日志不符（说明账目有漂移），把观测到的那张补回去
        if let (Some(exp), Some(got)) = (expect, top) {
            if exp != got {
                mem.stack.push(Some(exp));
            }
        }
    } else if let Some(exp) = expect {
        mem.stack.push(Some(exp));
    }
}

fn pop_any(mem: &mut Memory) {
    mem.stack.pop();
}

/// 用可观测的 `discard_count` / `discard_top` 自校验并修复账目。
fn repair(mem: &mut Memory, count: usize, top: Option<u8>) {
    if count == 0 {
        mem.stack.clear();
        return;
    }
    // 长度对齐：多了就砍掉堆顶，少了就在堆顶下方补"未知"
    while mem.stack.len() > count {
        mem.stack.pop();
    }
    while mem.stack.len() < count {
        let at = mem.stack.len().saturating_sub(1);
        mem.stack.insert(at, None);
    }
    // 堆顶以观测为准
    if let Some(t) = top {
        if let Some(last) = mem.stack.last_mut() {
            *last = Some(t);
        }
    }
}

/// 便捷：把 `DiscardInfo` 摊平成"已知在弃牌堆里的牌"数组。
pub fn as_counts(info: &DiscardInfo) -> [u16; 14] {
    let mut out = [0u16; 14];
    for r in 0..14 {
        out[r] = info.known[r].min(RANK_COPIES[r] as u16);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::sim::make_ai_session;
    use crate::game::view::project;
    use crate::game::{Phase, Settings};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// 弃牌堆重建必须"账目准确、且不凭空捏造"。
    #[test]
    fn discard_tracking_is_sound() {
        let hist = History::new();
        let registry = crate::ai::BotRegistry::with_builtins();
        let mut rng = StdRng::seed_from_u64(11);
        let mut checks = 0usize;
        for seed in [5u64, 9, 21] {
            let bots = ["challenger", "challenger", "challenger"];
            let mut s = make_ai_session(seed, Settings::default(), &bots);
            s.start_game().unwrap();
            for _ in 0..4000 {
                let pid = match &s.phase {
                    Phase::Peeking { done } => (0..3).find(|p| !done.contains_key(p)),
                    Phase::Turn { current, .. } => Some(*current),
                    _ => None,
                };
                let Some(pid) = pid else { break };
                // 只在 0 号座位行动时观察（真实用法：Bot 只在自己回合被调用）
                if pid == 0 && matches!(s.phase, Phase::Turn { .. }) {
                    let view = project(&s, Some(0), 0);
                    let info = hist.observe(&view);
                    if s.discard.len() > 1 {
                        checks += 1;
                        let below = &s.discard[..s.discard.len() - 1];
                        let claimed: usize = info.known.iter().map(|&c| c as usize).sum();
                        assert_eq!(
                            claimed + info.unknown,
                            below.len(),
                            "弃牌堆账目对不上：{claimed} 已知 + {} 未知，实际 {} 张",
                            info.unknown,
                            below.len()
                        );
                        let mut truth = [0usize; 14];
                        for &c in below {
                            truth[s.cards[c as usize].card.rank as usize] += 1;
                        }
                        for r in 0..14 {
                            assert!(
                                info.known[r] as usize <= truth[r],
                                "声称知道 {r} 有 {} 张，弃牌堆里其实只有 {} 张",
                                info.known[r],
                                truth[r]
                            );
                        }
                    }
                }
                let view = project(&s, Some(pid), 0);
                let cmd = registry.get(bots[pid]).unwrap().decide(&view, &mut rng);
                if s.apply(pid, &cmd).is_err() {
                    let _ = s.apply(pid, &crate::ai::fallback_command(&view));
                }
                if matches!(s.phase, Phase::RoundEnd) {
                    s.next_round().unwrap();
                }
            }
        }
        assert!(checks > 50, "样本太少：{checks}");
    }
}
