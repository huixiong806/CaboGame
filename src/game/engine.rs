//! 命令应用与回合推进：规则状态机的全部转移逻辑。
//!
//! 行动结构（2026-10 修订）：A 摸牌（弃置/发动能力）、B 交换（双来源、1~N 张）、
//! C 宣告 Cabo。

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rand::seq::SliceRandom;
use rand::Rng;

use super::{
    power_kind_of, Card, CardId, CardState, Command, GameError, LogKind, Pending, Phase, Pile,
    PlayerId, PowerKind, PowerUse, RANK_COPIES, Session, SlotId, INITIAL_SLOTS,
};

impl Session {
    // ---------------------------------------------------------------- 大厅/开局

    /// 座位是否都已就绪（真人或 AI），且人数在 2~4。
    pub fn seats_ready(&self) -> bool {
        (2..=4).contains(&self.players.len())
            && self.players.iter().all(|p| p.is_ai || p.token.is_some())
    }

    /// 发牌并进入"初始查看"阶段。新的一局会清零累计分。
    pub fn start_game(&mut self) -> Result<(), GameError> {
        if self.phase != Phase::Lobby {
            return Err(GameError::WrongPhase);
        }
        if !self.seats_ready() {
            return Err(GameError::Invalid("座位未就绪：需要 2~4 名玩家（真人或 AI）"));
        }
        for p in &mut self.players {
            p.total_score = 0;
            p.peeked_slots.clear();
        }
        self.round_no = 0;
        self.deal_and_begin_peek()
    }

    /// 结算后开下一轮。
    pub fn next_round(&mut self) -> Result<(), GameError> {
        if self.phase != Phase::RoundEnd {
            return Err(GameError::WrongPhase);
        }
        self.deal_and_begin_peek()
    }

    /// 游戏结束后再来一局（清零分数，回到大厅重新开局）。
    pub fn rematch(&mut self) -> Result<(), GameError> {
        if !matches!(self.phase, Phase::GameOver { .. }) {
            return Err(GameError::WrongPhase);
        }
        self.phase = Phase::Lobby;
        self.start_game()
    }

    fn deal_and_begin_peek(&mut self) -> Result<(), GameError> {
        // 组牌：0 有 2 张，1~12 各 4 张，13 有 2 张；洗混。
        let mut deck: Vec<Card> = Vec::with_capacity(52);
        for (rank, &copies) in RANK_COPIES.iter().enumerate() {
            for _ in 0..copies {
                deck.push(Card { id: deck.len() as u16, rank: rank as u8 });
            }
        }
        deck.shuffle(&mut self.rng);
        self.cards = deck
            .iter()
            .map(|&Card { id, rank }| CardState {
                card: Card { id, rank },
                revealed: false,
                known_by: BTreeSet::new(),
            })
            .collect();
        self.discard.clear();
        self.cabo_caller = None;
        self.extra_turns.clear();
        self.public_events.clear();
        self.peek_flash.clear();
        self.round_no += 1;

        // 每人 4 张暗牌（牌的 id 已随机对应点数，按序发牌即随机发牌）。
        let mut next_id = 0u16;
        for p in &mut self.players {
            p.slots = (next_id..next_id + INITIAL_SLOTS as u16).collect();
            next_id += INITIAL_SLOTS as u16;
        }
        self.deck = (next_id..self.cards.len() as u16).collect();
        // 起始弃牌：翻开一张，公共知识。
        let first = self.deck.pop().expect("52 张牌足够");
        self.cards[first as usize].revealed = true;
        self.cards[first as usize].known_by = (0..self.players.len()).collect();
        self.discard.push(first);
        self.log_public(
            LogKind::Info,
            format!("第 {} 轮开始：每人 4 张暗牌，弃牌堆起始为 {}", self.round_no, self.rank_str(first)),
        );

        self.phase = Phase::Peeking { done: BTreeMap::new() };
        self.log_public(LogKind::Peek, "请每位玩家秘密查看自己的 2 张牌");
        Ok(())
    }

    // ---------------------------------------------------------------- 命令入口

    /// 应用一名玩家的命令（由服务层调用；错误会显示给该玩家）。
    pub fn apply(&mut self, pid: PlayerId, cmd: &Command) -> Result<(), GameError> {
        // 该玩家行动时，其旧的"最近查看"闪显先清除（本次命令可能设置新闪显）。
        self.peek_flash.remove(&pid);
        self.apply_inner(pid, cmd)
    }

    fn apply_inner(&mut self, pid: PlayerId, cmd: &Command) -> Result<(), GameError> {
        match cmd {
            Command::PeekInitial { slots } => self.cmd_peek_initial(pid, *slots),
            Command::PeekToggle { slot } => self.cmd_peek_toggle(pid, *slot),
            Command::BeginDraw => self.cmd_begin_draw(pid),
            Command::DiscardDrawn { power } => self.cmd_discard_drawn(pid, *power),
            Command::DrawSwap { slots } => self.cmd_draw_swap(pid, slots.clone()),
            Command::ArmPower { kind } => self.cmd_arm_power(pid, *kind),
            Command::PowerPickOwn { slot } => self.cmd_power_pick_own(pid, *slot),
            Command::PowerPickOther { player, slot } => {
                self.cmd_power_pick_other(pid, *player, *slot)
            }
            Command::BeginSwap => self.cmd_begin_swap(pid),
            Command::SwapToggle { slot } => self.cmd_swap_toggle(pid, *slot),
            Command::SwapCommit => self.cmd_swap_commit(pid),
            Command::SwapOnce { slots } => self.cmd_swap_once(pid, slots),
            Command::CallCaboArm => self.cmd_cabo_arm(pid),
            Command::CallCabo => self.cmd_cabo(pid),
            Command::Cancel => self.cmd_cancel(pid),
        }
    }

    /// 当前应该行动的玩家。
    pub fn actor(&self) -> Option<PlayerId> {
        match &self.phase {
            Phase::Turn { current, .. } => Some(*current),
            _ => None,
        }
    }

    fn phase_turn(&self) -> Result<(PlayerId, Option<&Pending>), GameError> {
        match &self.phase {
            Phase::Turn { current, pending } => Ok((*current, pending.as_ref())),
            _ => Err(GameError::WrongPhase),
        }
    }

    fn require_turn(&self, pid: PlayerId) -> Result<(), GameError> {
        let (current, _) = self.phase_turn()?;
        if pid != current {
            return Err(GameError::NotYourTurn);
        }
        Ok(())
    }

    fn set_pending(&mut self, pending: Option<Pending>) {
        if let Phase::Turn { pending: p, .. } = &mut self.phase {
            *p = pending;
        }
    }

    /// 记录一次秘密查看：记忆模式下日志不带点数、只设置闪显；显示模式照旧写入日志。
    fn record_private_peek(&mut self, pid: PlayerId, text_with_value: String, text_plain: &str) {
        if self.hide_private_value() {
            self.peek_flash.insert(pid, text_with_value);
            self.log_to(pid, LogKind::Peek, text_plain);
        } else {
            self.log_to(pid, LogKind::Peek, text_with_value);
        }
    }

    // ---------------------------------------------------------------- 初始查看

    fn cmd_peek_initial(&mut self, pid: PlayerId, slots: [SlotId; 2]) -> Result<(), GameError> {
        if !matches!(self.phase, Phase::Peeking { .. }) {
            return Err(GameError::WrongPhase);
        }
        if let Phase::Peeking { done } = &self.phase {
            if done.contains_key(&pid) {
                return Err(GameError::Invalid("你已经查看过了"));
            }
        }
        if slots[0] == slots[1] {
            return Err(GameError::Invalid("必须选择两个不同的槽位"));
        }
        for &s in &slots {
            let cid = self.hand_card(pid, s)?;
            self.cards[cid as usize].known_by.insert(pid);
        }
        let (a, b) = (slots[0], slots[1]);
        let ra = self.rank_str(self.players[pid].slots[a as usize]);
        let rb = self.rank_str(self.players[pid].slots[b as usize]);
        self.record_private_peek(
            pid,
            format!("你查看了自己的槽位 {a} 和 {b}：{ra}、{rb}"),
            &format!("你查看了自己的槽位 {a} 和 {b}"),
        );

        let Phase::Peeking { done } = &mut self.phase else { unreachable!() };
        done.insert(pid, slots);
        if done.len() < self.players.len() {
            return Ok(());
        }
        // 全员提交：统一公开"谁看了哪两个槽位"（查看结果仍然保密），随机确定起始玩家。
        let picks: Vec<(PlayerId, [SlotId; 2])> = done.iter().map(|(&k, &v)| (k, v)).collect();
        let names: Vec<String> = self.players.iter().map(|p| p.name.clone()).collect();
        for (i, slots) in &picks {
            self.public_events.push(super::PublicEvent::InitialPeek { player: *i, slots: *slots });
            self.log_public(
                LogKind::Peek,
                format!("{} 查看了自己的槽位 {} 和 {}", names[*i], slots[0], slots[1]),
            );
        }
        let starter = self.rng.random_range(0..self.players.len());
        self.log_public(LogKind::Info, format!("随机确定 {} 先行", names[starter]));
        self.phase = Phase::Turn { current: starter, pending: None };
        Ok(())
    }

    fn cmd_peek_toggle(&mut self, pid: PlayerId, slot: SlotId) -> Result<(), GameError> {
        // 人类 UI：第一次点选进入暂存，凑满两张自动提交。
        if !matches!(self.phase, Phase::Peeking { .. }) {
            return Err(GameError::WrongPhase);
        }
        if let Phase::Peeking { done } = &self.phase {
            if done.contains_key(&pid) {
                return Err(GameError::Invalid("你已经查看过了"));
            }
        }
        self.hand_card(pid, slot)?;
        let (mut staged_len, contains) = {
            let staged = &self.players[pid].peeked_slots;
            (staged.len(), staged.contains(&slot))
        };
        if contains {
            return Ok(()); // 重复点击忽略
        }
        if staged_len >= 2 {
            return Err(GameError::Invalid("最多选择两张"));
        }
        self.players[pid].peeked_slots.push(slot);
        staged_len += 1;
        if staged_len == 2 {
            let a = self.players[pid].peeked_slots[0];
            let b = self.players[pid].peeked_slots[1];
            self.players[pid].peeked_slots.clear();
            return self.cmd_peek_initial(pid, [a, b]);
        }
        Ok(())
    }

    // ---------------------------------------------------------------- 行动 A：摸牌（弃置/发动能力）

    fn cmd_begin_draw(&mut self, pid: PlayerId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        if self.phase_turn()?.1.is_some() {
            return Err(GameError::Invalid("当前有未完成的操作"));
        }
        let card = self.deck.pop().ok_or(GameError::Invalid("摸牌堆已耗尽"))?;
        self.cards[card as usize].known_by.insert(pid);
        let name = self.players[pid].name.clone();
        self.log_public(LogKind::Info, format!("{name} 从摸牌堆抽了一张牌"));
        self.record_private_peek(
            pid,
            format!("你抽到了 {}", self.rank_str(card)),
            "你抽到了一张牌",
        );
        self.set_pending(Some(Pending::Drew { card }));
        Ok(())
    }

    fn drawn_of(&self) -> Result<CardId, GameError> {
        match self.phase_turn()?.1 {
            Some(Pending::Drew { card }) => Ok(*card),
            _ => Err(GameError::Invalid("当前没有待处理的抽牌")),
        }
    }

    fn cmd_discard_drawn(&mut self, pid: PlayerId, power: Option<PowerUse>) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let card = self.drawn_of()?;
        if let Some(pu) = power {
            self.execute_power(pid, card, pu)?;
        }
        self.discard_drawn_public(pid, card, power.is_some());
        self.set_pending(None);
        self.finish_turn();
        Ok(())
    }

    // ---------------------------------------------------------------- 行动 A 的能力（逐步操作）

    fn cmd_arm_power(&mut self, pid: PlayerId, kind: PowerKind) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let card = self.drawn_of()?;
        if power_kind_of(self.cards[card as usize].card.rank) != Some(kind) {
            return Err(GameError::Invalid("这张牌没有该能力"));
        }
        self.set_pending(Some(Pending::PowerAiming { card, kind, my_slot: None }));
        Ok(())
    }

    fn cmd_power_pick_own(&mut self, pid: PlayerId, slot: SlotId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let Some(Pending::PowerAiming { card, kind, my_slot: _ }) = self.phase_turn()?.1.cloned()
        else {
            return Err(GameError::Invalid("当前不在能力瞄准状态"));
        };
        self.hand_card(pid, slot)?;
        match kind {
            PowerKind::Peek => {
                self.set_pending(None);
                self.execute_power(pid, card, PowerUse::PeekOwn { slot })?;
                self.discard_drawn_public(pid, card, true);
                self.finish_turn();
            }
            PowerKind::Swap => {
                self.set_pending(Some(Pending::PowerAiming { card, kind, my_slot: Some(slot) }));
            }
            PowerKind::Spy => return Err(GameError::Invalid("请选择其他玩家的牌")),
        }
        Ok(())
    }

    fn cmd_power_pick_other(
        &mut self,
        pid: PlayerId,
        target: PlayerId,
        slot: SlotId,
    ) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let Some(Pending::PowerAiming { card, kind, my_slot }) = self.phase_turn()?.1.cloned()
        else {
            return Err(GameError::Invalid("当前不在能力瞄准状态"));
        };
        if target == pid || target >= self.players.len() {
            return Err(GameError::Invalid("请选择其他玩家"));
        }
        self.hand_card(target, slot)?;
        match kind {
            PowerKind::Spy => {
                self.set_pending(None);
                self.execute_power(pid, card, PowerUse::Spy { player: target, slot })?;
                self.discard_drawn_public(pid, card, true);
                self.finish_turn();
            }
            PowerKind::Swap => {
                let my = my_slot.ok_or(GameError::Invalid("请先选择自己的槽位"))?;
                self.set_pending(None);
                self.execute_power(pid, card, PowerUse::Swap { my_slot: my, player: target, slot })?;
                self.discard_drawn_public(pid, card, true);
                self.finish_turn();
            }
            PowerKind::Peek => return Err(GameError::Invalid("请选择自己的槽位")),
        }
        Ok(())
    }

    /// 行动 A 收尾：把刚抽的牌公开进弃牌堆（能力已结算）。
    fn discard_drawn_public(&mut self, pid: PlayerId, card: CardId, powered: bool) {
        self.public_events.push(super::PublicEvent::Discard {
            player: pid, rank: self.cards[card as usize].card.rank, powered,
        });
        let rank = self.rank_str(card);
        let name = self.players[pid].name.clone();
        self.cards[card as usize].revealed = true;
        self.cards[card as usize].known_by = (0..self.players.len()).collect();
        self.discard.push(card);
        if powered {
            self.log_public(LogKind::Power, format!("{name} 发动能力后弃置了 {rank}"));
        } else {
            self.log_public(LogKind::Info, format!("{name} 弃置了 {rank}"));
        }
    }

    /// 结算一次能力使用（不做阶段检查，供逐步命令共用）。
    fn execute_power(&mut self, pid: PlayerId, card: CardId, pu: PowerUse) -> Result<(), GameError> {
        let rank = self.cards[card as usize].card.rank;
        let kind = power_kind_of(rank).ok_or(GameError::Invalid("这张牌没有能力"))?;
        let name = self.players[pid].name.clone();
        match (kind, pu) {
            (PowerKind::Peek, PowerUse::PeekOwn { slot }) => {
                let cid = self.hand_card(pid, slot)?;
                self.cards[cid as usize].known_by.insert(pid);
                let r = self.rank_str(cid);
                self.log_public(LogKind::Power, format!("{name} 发动【偷看】，查看自己的槽位 {slot}"));
                self.record_private_peek(
                    pid,
                    format!("你偷看自己的槽位 {slot}：{r}"),
                    &format!("你偷看了自己的槽位 {slot}"),
                );
            }
            (PowerKind::Spy, PowerUse::Spy { player, slot }) => {
                if player == pid || player >= self.players.len() {
                    return Err(GameError::Invalid("间谍目标必须是其他玩家"));
                }
                let cid = self.hand_card(player, slot)?;
                self.cards[cid as usize].known_by.insert(pid);
                let target_name = self.players[player].name.clone();
                let r = self.rank_str(cid);
                self.log_public(
                    LogKind::Power,
                    format!("{name} 发动【间谍】，查看了 {target_name} 的槽位 {slot}"),
                );
                self.record_private_peek(
                    pid,
                    format!("{target_name} 的槽位 {slot} 是 {r}"),
                    &format!("你查看了 {target_name} 的槽位 {slot}"),
                );
            }
            (PowerKind::Swap, PowerUse::Swap { my_slot, player, slot }) => {
                if player == pid || player >= self.players.len() {
                    return Err(GameError::Invalid("交换对象必须是其他玩家"));
                }
                let a = self.hand_card(pid, my_slot)?;
                let b = self.hand_card(player, slot)?;
                // 牌互换位置；known_by 属于牌本身，随牌移动（集合不变）。
                self.players[pid].slots[my_slot as usize] = b;
                self.players[player].slots[slot as usize] = a;
                let target_name = self.players[player].name.clone();
                self.log_public(
                    LogKind::Swap,
                    format!(
                        "{name} 发动【交换】：自己的槽位 {my_slot} 与 {target_name} 的槽位 {slot} 交换（双方不看牌面）"
                    ),
                );
            }
            _ => return Err(GameError::Invalid("能力与目标不匹配")),
        }
        self.public_events.push(super::PublicEvent::Power { player: pid, power: pu });
        Ok(())
    }

    // ---------------------------------------------------------------- 行动 B：交换

    /// 校验并返回当前玩家某槽位的手牌。
    fn hand_card(&self, pid: PlayerId, slot: SlotId) -> Result<CardId, GameError> {
        self.players[pid]
            .slots
            .get(slot as usize)
            .copied()
            .ok_or(GameError::Invalid("该槽位没有牌"))
    }

    /// 行动 A（摸牌后）：把刚抽的牌与 1~N 张手牌交换。
    /// `slots` 为空 = 进入逐步选择模式；非空 = 选定这些槽位并立即结算。
    fn cmd_draw_swap(&mut self, pid: PlayerId, slots: Vec<SlotId>) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let card = self.drawn_of()?;
        if !slots.is_empty() && slots.len() > self.players[pid].slots.len() {
            return Err(GameError::Invalid("选择的槽位过多"));
        }
        self.set_pending(Some(Pending::SwapSelecting {
            card,
            source: Pile::Draw,
            selected: Vec::new(),
        }));
        if slots.is_empty() {
            return Ok(()); // 人类多张交换：进入逐步选择
        }
        for &s in &slots {
            self.cmd_swap_toggle(pid, s)?;
        }
        self.cmd_swap_commit(pid)
    }

    /// 行动 B：拿弃牌堆顶（公开信息，确认前可取消），与 1~N 张手牌交换。
    fn cmd_begin_swap(&mut self, pid: PlayerId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        if self.phase_turn()?.1.is_some() {
            return Err(GameError::Invalid("当前有未完成的操作"));
        }
        let top = self.discard_top().ok_or(GameError::Invalid("弃牌堆是空的"))?;
        // 弃牌堆顶本就公开，暂存不移动任何牌，可随时取消。
        self.set_pending(Some(Pending::SwapSelecting {
            card: top,
            source: Pile::Discard,
            selected: Vec::new(),
        }));
        Ok(())
    }

    fn cmd_swap_toggle(&mut self, pid: PlayerId, slot: SlotId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let Some(Pending::SwapSelecting { card, source, selected }) = self.phase_turn()?.1.cloned()
        else {
            return Err(GameError::Invalid("当前不在交换选择状态"));
        };
        // 可以选任何有牌的槽位（包含无人看过点数的暗牌）。
        self.hand_card(pid, slot)?;
        let mut next = selected;
        match next.iter().position(|&s| s == slot) {
            Some(pos) => {
                next.remove(pos);
            }
            None => {
                next.push(slot);
                next.sort();
            }
        }
        self.set_pending(Some(Pending::SwapSelecting { card, source, selected: next }));
        Ok(())
    }

    fn cmd_swap_commit(&mut self, pid: PlayerId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        let Some(Pending::SwapSelecting { card, source, selected }) = self.phase_turn()?.1.cloned()
        else {
            return Err(GameError::Invalid("当前不在交换选择状态"));
        };
        if selected.is_empty() {
            return Err(GameError::Invalid("至少选择一张手牌"));
        }
        // 取出获得的牌（弃牌堆来源此时才真正拿走堆顶）。
        let taken = match source {
            Pile::Draw => card,
            Pile::Discard => {
                let top = self.discard.pop().ok_or(GameError::Invalid("弃牌堆是空的"))?;
                debug_assert_eq!(top, card, "弃牌堆顶在本回合内不会变化");
                top
            }
        };
        let name = self.players[pid].name.clone();
        let slots_txt = selected.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("、");
        let ranks: Vec<u8> = selected
            .iter()
            .filter_map(|&s| self.hand_card(pid, s).ok())
            .map(|c| self.cards[c as usize].card.rank)
            .collect();

        let all_success = selected.len() == 1 || ranks.iter().all(|&r| r == ranks[0]);
        self.public_events.push(super::PublicEvent::Exchange {
            player: pid, source, slots: selected.clone(), exposed: ranks.clone(),
            incoming: if source == Pile::Discard { Some(self.cards[taken as usize].card.rank) } else { None },
            success: all_success,
        });
        // 手牌保持紧凑：换走的牌按槽位从大到小依次移除，获得的牌追加到末尾。
        if all_success {
            for &s in selected.iter().rev() {
                let old = self.players[pid].slots.remove(s as usize);
                self.cards[old as usize].revealed = true;
                self.cards[old as usize].known_by = (0..self.players.len()).collect();
                self.discard.push(old);
            }
            // 获得的牌暗置入手：K 不变（弃牌堆来源人人皆知，摸牌堆来源仅自己）。
            self.cards[taken as usize].revealed = false;
            self.players[pid].slots.push(taken);
            let ranks_txt = ranks.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" ");
            if selected.len() == 1 {
                match source {
                    Pile::Discard => {
                        let tr = self.rank_str(taken);
                        let out_txt =
                            ranks.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" ");
                        // 换出的牌是正面朝上进入弃牌堆的（引擎里已标 revealed/known_by=全体），
                        // 属于公开信息 —— 必须写进日志，否则只有"当下看一眼堆顶"才能知道，
                        // 而 AI 只在自己决策点拿视图，会被后续动作压掉。
                        self.log_public(
                            LogKind::Swap,
                            format!("{name} 亮出槽位 {slots_txt} 的 {out_txt}，换入弃牌堆顶的 {tr}"),
                        );
                    }
                    Pile::Draw => {
                        let out_txt =
                            ranks.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" ");
                        self.log_public(
                            LogKind::Swap,
                            format!("{name} 把刚摸的牌与槽位 {slots_txt} 交换，亮出 {out_txt}"),
                        );
                    }
                }
            } else {
                self.log_public(
                    LogKind::Swap,
                    format!(
                        "{name} 翻开槽位 {slots_txt}：{ranks_txt} —— 点数相同，交换成功！这些牌弃置，获得的牌加入手牌末尾"
                    ),
                );
            }
        } else {
            // 失败：选中的牌全部明置并留在原位；获得的牌暗置追加到手牌末尾。
            let ranks_txt = ranks.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" ");
            for &s in &selected {
                let cid = self.players[pid].slots[s as usize];
                self.cards[cid as usize].revealed = true;
                self.cards[cid as usize].known_by = (0..self.players.len()).collect();
            }
            // 新牌暗置入手：来自弃牌堆则 K=全体不变，来自摸牌堆则 K=自己不变。
            self.cards[taken as usize].revealed = false;
            self.players[pid].slots.push(taken);
            let source_txt = match source {
                Pile::Discard => "弃牌堆".to_string(),
                Pile::Draw => "摸牌堆".to_string(),
            };
            self.log_public(
                LogKind::Swap,
                format!(
                    "{name} 翻开槽位 {slots_txt}：{ranks_txt} —— 存在不相同的牌，交换失败！这些牌保持明置，从{source_txt}获得的牌暗置加入手牌末尾"
                ),
            );
        }
        self.set_pending(None);
        self.finish_turn();
        Ok(())
    }

    fn cmd_swap_once(&mut self, pid: PlayerId, slots: &[SlotId]) -> Result<(), GameError> {
        if slots.is_empty() {
            return Err(GameError::Invalid("至少选择一张手牌"));
        }
        self.cmd_begin_swap(pid)?;
        for &s in slots {
            self.cmd_swap_toggle(pid, s)?;
        }
        self.cmd_swap_commit(pid)
    }

    // ---------------------------------------------------------------- 行动 C：宣告 Cabo

    fn cmd_cabo_arm(&mut self, pid: PlayerId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        if self.phase_turn()?.1.is_some() {
            return Err(GameError::Invalid("当前有未完成的操作"));
        }
        if self.cabo_caller.is_some() {
            return Err(GameError::Invalid("本轮已有人宣告 Cabo"));
        }
        self.set_pending(Some(Pending::ConfirmCabo));
        Ok(())
    }

    fn cmd_cabo(&mut self, pid: PlayerId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        match self.phase_turn()?.1 {
            None | Some(Pending::ConfirmCabo) => {}
            Some(_) => return Err(GameError::Invalid("当前有未完成的操作")),
        }
        if self.cabo_caller.is_some() {
            return Err(GameError::Invalid("本轮已有人宣告 Cabo"));
        }
        self.cabo_caller = Some(pid);
        self.public_events.push(super::PublicEvent::Cabo { player: pid });
        // 从宣告者的下一位开始，其余每人再加时一回合（加时中不能再宣告）。
        let n = self.players.len();
        let mut queue = VecDeque::new();
        for step in 1..n {
            queue.push_back((pid + step) % n);
        }
        self.extra_turns = queue;
        let name = self.players[pid].name.clone();
        self.log_public(LogKind::Cabo, format!("📣 {name} 宣告 Cabo！其余玩家各再行动一次"));
        self.set_pending(None);
        self.finish_turn();
        Ok(())
    }

    // ---------------------------------------------------------------- 取消暂存

    fn cmd_cancel(&mut self, pid: PlayerId) -> Result<(), GameError> {
        self.require_turn(pid)?;
        match self.phase_turn()?.1 {
            // 交换-弃牌堆来源未获得秘密信息，可取消；摸牌堆来源已秘密查看，必须完成交换。
            Some(Pending::SwapSelecting { source: Pile::Discard, .. })
            | Some(Pending::PowerAiming { .. })
            | Some(Pending::ConfirmCabo) => {
                self.set_pending(None);
                Ok(())
            }
            Some(Pending::SwapSelecting { source: Pile::Draw, .. }) => {
                Err(GameError::Invalid("已从摸牌堆获得的牌必须完成交换"))
            }
            Some(Pending::Drew { .. }) => Err(GameError::Invalid("已抽的牌必须弃置或交换")),
            None => Err(GameError::Invalid("没有可取消的操作")),
        }
    }

    // ---------------------------------------------------------------- 回合推进 / 结算

    /// 当前行动完成后的推进：先看摸牌堆是否耗尽，再看 Cabo 加时队列。
    fn finish_turn(&mut self) {
        if self.deck.is_empty() {
            self.log_public(LogKind::Info, "摸牌堆已耗尽，本轮立即结束");
            self.end_round();
            return;
        }
        let names: Vec<String> = self.players.iter().map(|p| p.name.clone()).collect();
        if let Some(caller) = self.cabo_caller {
            match self.extra_turns.pop_front() {
                Some(next) => {
                    self.log_public(LogKind::Cabo, format!("加时回合：轮到 {}", names[next]));
                    self.phase = Phase::Turn { current: next, pending: None };
                }
                None => {
                    self.log_public(
                        LogKind::Cabo,
                        format!("加时回合结束，亮牌结算（{} 宣告的 Cabo）", names[caller]),
                    );
                    self.end_round();
                }
            }
        } else {
            let current = match &self.phase {
                Phase::Turn { current, .. } => *current,
                _ => 0,
            };
            let next = (current + 1) % self.players.len();
            self.phase = Phase::Turn { current: next, pending: None };
        }
    }

    /// 亮牌、计分、累计，并判断游戏是否结束。
    fn end_round(&mut self) {
        let all: BTreeSet<PlayerId> = (0..self.players.len()).collect();
        // 翻开所有剩余手牌。
        for p in &mut self.players {
            for cid in p.slots.iter() {
                self.cards[*cid as usize].revealed = true;
                self.cards[*cid as usize].known_by = all.clone();
            }
        }
        // 计算各家点数总和与本轮得分。
        let sums: Vec<u32> = self
            .players
            .iter()
            .map(|p| p.slots.iter().map(|c| self.cards[*c as usize].card.rank as u32).sum())
            .collect();
        let caller = self.cabo_caller;
        let penalty = self.settings.cabo_penalty;
        let mut scores: Vec<u32> = sums.clone();
        if let Some(c) = caller {
            let others_min = sums
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != c)
                .map(|(_, &v)| v)
                .min()
                .unwrap_or(u32::MAX);
            scores[c] = if sums[c] < others_min { 0 } else { sums[c] + penalty };
        }
        for (i, p) in self.players.iter_mut().enumerate() {
            p.round_score = Some(scores[i]);
            p.total_score += scores[i];
        }
        // 亮牌明细日志。
        for i in 0..self.players.len() {
            let detail = self.players[i]
                .slots
                .iter()
                .map(|c| self.cards[*c as usize].card.rank.to_string())
                .collect::<Vec<_>>()
                .join(" + ");
            let count = self.players[i].slot_count();
            let extra = if Some(i) == caller {
                if scores[i] == 0 {
                    "，Cabo 成功 +0".to_string()
                } else {
                    format!("，Cabo 未严格最低：+{penalty} 惩罚")
                }
            } else {
                String::new()
            };
            self.log_public(
                LogKind::Score,
                format!(
                    "{}：{detail} = {}（{count} 张牌），本轮 +{}{}",
                    self.players[i].name, sums[i], scores[i], extra
                ),
            );
        }
        // 判断游戏结束。
        let target = self.settings.target_score;
        let max_total = self.players.iter().map(|p| p.total_score).max().unwrap_or(0);
        if max_total >= target {
            let min = self.players.iter().map(|p| p.total_score).min().unwrap_or(0);
            let winners: Vec<PlayerId> = (0..self.players.len())
                .filter(|&i| self.players[i].total_score == min)
                .collect();
            let wn: Vec<&str> = winners.iter().map(|&i| self.players[i].name.as_str()).collect();
            self.log_public(LogKind::Score, format!("🏆 游戏结束！最低分获胜：{}", wn.join("、")));
            self.phase = Phase::GameOver { winners };
        } else {
            self.phase = Phase::RoundEnd;
            self.log_public(LogKind::Info, "本轮结束，等待房主开始下一轮");
        }
        self.peek_flash.clear();
    }
}
