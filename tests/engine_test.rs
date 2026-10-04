//! 规则引擎集成测试：直接操作公共 API 验证规则要点（2026-10 修订版规则，
//! 摸牌后先看再决策、统一交换、紧凑手牌无空位）。

use cabo::ai::BotRegistry;
use cabo::game::sim::{make_ai_session, run_game};
use cabo::game::view::project;
use cabo::game::{
    Command, GameError, Pending, Phase, PlayerState, PowerUse, Settings, Session, RANK_COPIES,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn human_session(n: usize) -> Session {
    human_session_with(n, Settings::default())
}

fn human_session_with(n: usize, settings: Settings) -> Session {
    let players = (0..n)
        .map(|i| PlayerState {
            name: format!("P{}", i + 1),
            is_ai: false,
            bot_id: String::new(),
            token: Some(format!("tok{i}")),
            slots: Vec::new(),
            peeked_slots: Vec::new(),
            total_score: 0,
            round_score: None,
            score_reset_used: false,
            score_reset_this_round: false,
        })
        .collect();
    Session::new_lobby(42, settings, players)
}

/// 所有人完成初始查看（槽位 0、1），进入回合阶段。
fn peek_all(s: &mut Session) {
    for pid in 0..s.players.len() {
        s.apply(pid, &Command::PeekInitial { slots: [0, 1] }).unwrap();
    }
    assert!(matches!(s.phase, Phase::Turn { .. }));
}

/// 把摸牌堆顶换成指定点数的牌。
fn put_on_deck_top(s: &mut Session, rank: u8) -> u16 {
    let id = s
        .deck
        .iter()
        .copied()
        .find(|&id| s.cards[id as usize].card.rank == rank)
        .unwrap_or_else(|| panic!("牌池里找不到点数 {rank}"));
    s.deck.retain(|&x| x != id);
    s.deck.push(id);
    id
}

/// 把某座位的 4 张牌强制设为指定点数。优先从摸牌堆/弃牌堆取牌，
/// 不够时从其他（非 `avoid`）玩家手牌"借"（把被顶替的牌还给来源槽位）。
fn force_hand(s: &mut Session, pid: usize, ranks: &[u8; 4], avoid: &[usize]) {
    for (slot, &r) in ranks.iter().enumerate() {
        let cur = s.players[pid].slots[slot];
        if s.cards[cur as usize].card.rank == r {
            continue;
        }
        let mut found: Option<(u16, Option<(usize, usize)>)> = None;
        if let Some(i) = s.deck.iter().position(|&id| s.cards[id as usize].card.rank == r) {
            let id = s.deck.remove(i);
            found = Some((id, None));
        }
        if found.is_none() {
            if let Some(i) = s.discard.iter().position(|&id| s.cards[id as usize].card.rank == r) {
                let id = s.discard.remove(i);
                found = Some((id, None));
            }
        }
        if found.is_none() {
            'outer: for q in 0..s.players.len() {
                if q == pid || avoid.contains(&q) {
                    continue;
                }
                for j in 0..s.players[q].slots.len() {
                    let id = s.players[q].slots[j];
                    if s.cards[id as usize].card.rank == r {
                        found = Some((id, Some((q, j))));
                        break 'outer;
                    }
                }
            }
        }
        let (id, origin) = found.unwrap_or_else(|| panic!("找不到点数 {r}"));
        let old = s.players[pid].slots[slot];
        s.players[pid].slots[slot] = id;
        match origin {
            None => s.deck.push(old),
            Some((q, j)) => s.players[q].slots[j] = old,
        }
    }
}

/// 空过一个加时回合：摸一张牌直接弃置（手牌不变）。
fn idle_via_draw_discard(s: &mut Session, pid: usize) {
    s.apply(pid, &Command::BeginDraw).unwrap();
    s.apply(pid, &Command::DiscardDrawn { power: None }).unwrap();
}

fn settle_via_cabo(s: &mut Session, caller: usize) {
    s.phase = Phase::Turn {
        current: caller,
        pending: None,
    };
    s.apply(caller, &Command::CallCabo).unwrap();
    while let Phase::Turn { current, .. } = s.phase {
        idle_via_draw_discard(s, current);
    }
}

#[test]
fn high_pairs_overrides_cabo_then_reset_runs_before_game_over() {
    let mut s = human_session(3);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[12, 13, 12, 13], &[]);
    force_hand(&mut s, 1, &[0, 0, 1, 1], &[0]);
    force_hand(&mut s, 2, &[2, 2, 3, 3], &[0, 1]);
    for (p, total) in s.players.iter_mut().zip([70, 50, 49]) {
        p.total_score = total;
    }
    let before = project(&s, Some(0), 0);
    assert!(
        !before.all_seats[0].is_high_pairs,
        "未结算时不公开高牌双对组合信息"
    );
    settle_via_cabo(&mut s, 1); // 普通规则下宣告者严格最低，高牌双对仍覆盖其 0 分。
    assert_eq!(
        s.players.iter().map(|p| p.round_score).collect::<Vec<_>>(),
        [Some(0), Some(50), Some(50)]
    );
    assert_eq!(
        s.players.iter().map(|p| p.total_score).collect::<Vec<_>>(),
        [70, 50, 99]
    );
    assert!(matches!(s.phase, Phase::RoundEnd));
    let v = project(&s, Some(0), 0);
    assert!(v.all_seats[0].is_high_pairs);
    assert!(v.all_seats[1].score_reset_this_round);
    assert!(v.all_seats[1].score_reset_used);
    assert!(s.log.iter().any(|e| e.text.contains("高牌双对")));
    assert!(s.log.iter().any(|e| e.text.contains("100 → 重置为 50")));
    s.next_round().unwrap();
    assert!(s.players[1].score_reset_used, "下一轮不能恢复资格");
    assert!(s
        .players
        .iter()
        .all(|p| p.round_score.is_none() && !p.score_reset_this_round));
    assert!(!project(&s, Some(0), 0).all_seats[0].is_high_pairs);
}

#[test]
fn high_pairs_applies_on_deck_exhaustion_and_extra_card_invalidates_it() {
    for expanded in [false, true] {
        let mut s = human_session(2);
        s.start_game().unwrap();
        peek_all(&mut s);
        force_hand(&mut s, 0, &[13, 12, 13, 12], &[]);
        force_hand(&mut s, 1, &[1, 1, 1, 1], &[0]);
        if expanded {
            let extra = s.deck.pop().unwrap();
            s.players[0].slots.push(extra);
        }
        let expected_sum: u32 = s.players[0]
            .slots
            .iter()
            .map(|&id| s.cards[id as usize].card.rank as u32)
            .sum();
        while !s.deck.is_empty() {
            let actor = s.actor().unwrap();
            idle_via_draw_discard(&mut s, actor);
        }
        assert!(matches!(s.phase, Phase::RoundEnd));
        assert_eq!(s.cabo_caller, None);
        assert_eq!(
            s.players[0].round_score,
            Some(if expanded { expected_sum } else { 0 })
        );
        assert_eq!(
            s.players[1].round_score,
            Some(if expanded { 4 } else { 50 })
        );
    }
}

#[test]
fn failed_cabo_can_reset_and_win_the_match() {
    for already_used in [false, true] {
        let mut s = human_session(3);
        s.start_game().unwrap();
        peek_all(&mut s);
        force_hand(&mut s, 0, &[5, 5, 0, 0], &[]);
        force_hand(&mut s, 1, &[2, 2, 3, 4], &[0]);
        force_hand(&mut s, 2, &[1, 1, 1, 1], &[0, 1]);
        for (p, total) in s.players.iter_mut().zip([80, 90, 95]) {
            p.total_score = total;
        }
        s.players[0].score_reset_used = already_used;
        settle_via_cabo(&mut s, 0);
        assert_eq!(s.players[0].round_score, Some(20));
        assert_eq!(
            s.players.iter().map(|p| p.total_score).collect::<Vec<_>>(),
            [if already_used { 100 } else { 50 }, 101, 99]
        );
        assert_eq!(
            s.phase,
            Phase::GameOver {
                winners: vec![if already_used { 2 } else { 0 }]
            }
        );
        assert_eq!(s.players[0].score_reset_this_round, !already_used);
        s.rematch().unwrap();
        assert!(s
            .players
            .iter()
            .all(|p| p.total_score == 0 && !p.score_reset_used && !p.score_reset_this_round));
    }
}

#[test]
fn two_player_successful_cabo_cannot_manufacture_reset_penalty() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[5, 5, 0, 0], &[]);
    force_hand(&mut s, 1, &[2, 2, 3, 4], &[0]);
    s.players[0].total_score = 80;
    s.players[1].total_score = 90;
    settle_via_cabo(&mut s, 0);
    assert_eq!(s.players[0].round_score, Some(0));
    assert_eq!(s.players[0].total_score, 80);
    assert!(!s.players[0].score_reset_used);
    assert_eq!(s.phase, Phase::GameOver { winners: vec![0] });
}

#[test]
fn reset_boundary_precedes_configured_end_threshold() {
    for (target, total, used, expected, end) in [
        (100, 90, false, 50, false),
        (100, 91, false, 101, true),
        (100, 90, true, 100, true),
        (60, 90, false, 50, false),
        (50, 90, false, 50, true),
    ] {
        let mut s = human_session_with(
            2,
            Settings {
                target_score: target,
                ..Settings::default()
            },
        );
        s.start_game().unwrap();
        peek_all(&mut s);
        force_hand(&mut s, 0, &[0, 0, 1, 1], &[]);
        force_hand(&mut s, 1, &[2, 2, 3, 3], &[0]);
        s.players[1].total_score = total;
        s.players[1].score_reset_used = used;
        settle_via_cabo(&mut s, 0);
        assert_eq!(s.players[1].total_score, expected);
        assert_eq!(s.players[1].score_reset_this_round, !used && total == 90);
        assert_eq!(matches!(s.phase, Phase::GameOver { .. }), end);
    }
}

#[test]
fn deck_composition() {
    let mut s = human_session(3);
    s.start_game().unwrap();
    assert_eq!(s.cards.len(), 52);
    let mut counts = [0usize; 14];
    for c in &s.cards {
        counts[c.card.rank as usize] += 1;
    }
    assert_eq!(counts, RANK_COPIES);
    assert_eq!(s.deck.len(), 52 - 3 * 4 - 1);
    assert_eq!(s.discard.len(), 1);
    assert!(s.cards[s.discard[0] as usize].revealed);
}

#[test]
fn lobby_requires_ready_seats() {
    let mut s = human_session(1);
    assert!(matches!(s.start_game(), Err(GameError::Invalid(_))));
    s.players.push(PlayerState {
        name: "AI".into(),
        is_ai: true,
        bot_id: "challenger".into(),
        token: None,
        slots: Vec::new(),
        peeked_slots: Vec::new(),
        total_score: 0,
        round_score: None,
        score_reset_used: false,
        score_reset_this_round: false,
    });
    assert!(s.start_game().is_ok());
}

#[test]
fn peeking_is_private_then_public() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    let c00 = s.players[0].slots[0];
    s.apply(0, &Command::PeekInitial { slots: [0, 1] }).unwrap();
    assert!(s.cards[c00 as usize].known_by.contains(&0));
    assert!(!s.cards[c00 as usize].known_by.contains(&1));
    assert!(matches!(s.phase, Phase::Peeking { .. }));
    assert!(s.apply(0, &Command::PeekInitial { slots: [0, 1] }).is_err());
    s.apply(1, &Command::PeekInitial { slots: [2, 3] }).unwrap();
    assert!(matches!(s.phase, Phase::Turn { .. }));
    let c12 = s.players[1].slots[2];
    assert!(s.cards[c12 as usize].known_by.contains(&1));
    assert!(!s.cards[c12 as usize].known_by.contains(&0));
}

#[test]
fn memory_mode_hides_log_values_and_sets_flash() {
    let mut s =
        human_session_with(2, Settings { memory_mode: true, ..Settings::default() });
    s.start_game().unwrap();
    let r0 = s.cards[s.players[0].slots[0] as usize].card.rank;
    let r1 = s.cards[s.players[0].slots[1] as usize].card.rank;
    s.apply(0, &Command::PeekInitial { slots: [0, 1] }).unwrap();
    // 日志不含点数。
    let own_log: Vec<&str> = s
        .log
        .iter()
        .filter(|e| e.audience == cabo::game::Audience::Player(0))
        .map(|e| e.text.as_str())
        .collect();
    assert!(
        own_log.iter().all(|t| !t.contains(&r0.to_string()) && !t.contains(&r1.to_string())),
        "记忆模式下日志不应包含点数: {own_log:?}"
    );
    // 闪显包含点数，仅对该玩家可见。
    let ftext = s.peek_flash.get(&0).unwrap();
    assert!(ftext.contains(&r0.to_string()) && ftext.contains(&r1.to_string()));
    // 视图：私有知识在记忆模式下不显示（knower_count=1 < n=2），但 value 仍可读（Bot 用）。
    let view = project(&s, Some(0), 0);
    let sv = &view.me_seat.as_ref().unwrap().slots[0];
    assert!(sv.i_know && !sv.shown && sv.value == Some(r0));
    // P1 看不到 P0 的知识。
    let view1 = project(&s, Some(1), 0);
    assert!(!view1.me_seat.as_ref().unwrap().slots[0].i_know);
    // 闪显按玩家独立存储：P1 的查看不会覆盖 P0 的闪显。
    assert!(s.peek_flash.contains_key(&0));
    // 该玩家下一次成功行动后自己的闪显清除。
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::BeginSwap).unwrap();
    assert!(!s.peek_flash.contains_key(&0));
}

#[test]
fn swap_discard_single_card() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    s.phase = Phase::Turn { current: 0, pending: None };
    let top = s.discard_top().unwrap();
    let before = s.players[0].slots.clone();
    let old0 = s.players[0].slots[0];
    s.apply(0, &Command::BeginSwap).unwrap();
    // 弃牌堆来源可取消。
    s.apply(0, &Command::Cancel).unwrap();
    s.apply(0, &Command::BeginSwap).unwrap();
    s.apply(0, &Command::SwapToggle { slot: 0 }).unwrap();
    s.apply(0, &Command::SwapCommit).unwrap();
    // 单张原位替换，其余手牌不动。
    let hand = &s.players[0].slots;
    assert_eq!(hand.len(), 4);
    assert_eq!(hand, &vec![top, before[1], before[2], before[3]]);
    assert!(!hand.contains(&old0), "原牌已换出");
    assert!(!s.cards[top as usize].revealed, "换入的牌面朝下");
    assert_eq!(s.cards[top as usize].known_by.len(), 2, "但来自弃牌堆，点数公开");
    assert_eq!(*s.discard.last().unwrap(), old0);
    assert!(s.cards[old0 as usize].revealed);
    assert!(matches!(s.phase, Phase::Turn { current: 1, pending: None }));
}

#[test]
fn draw_then_swap_single_is_secret() {
    // 新流程：先摸牌看到，再决定与手牌交换。
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::BeginDraw).unwrap();
    let card = match &s.phase {
        Phase::Turn { pending: Some(Pending::Drew { card }), .. } => *card,
        other => panic!("unexpected {other:?}"),
    };
    assert!(s.cards[card as usize].known_by.contains(&0));
    assert_eq!(s.cards[card as usize].known_by.len(), 1);
    // 摸牌后不可取消（必须弃置或交换）。
    assert!(s.apply(0, &Command::Cancel).is_err());
    let old1 = s.players[0].slots[1];
    let before = s.players[0].slots.clone();
    // 与槽位 1 的手牌交换（单张，必成功）。
    s.apply(0, &Command::DrawSwap { slots: vec![1] }).unwrap();
    let hand = &s.players[0].slots;
    assert_eq!(hand.len(), 4);
    assert_eq!(hand, &vec![before[0], card, before[2], before[3]]);
    assert!(!hand.contains(&old1));
    assert!(!s.cards[card as usize].revealed);
    assert_eq!(s.cards[card as usize].known_by, [0usize].into_iter().collect(), "来自摸牌堆仅自己知道");
    assert_eq!(*s.discard.last().unwrap(), old1);
    assert!(matches!(s.phase, Phase::Turn { current: 1, pending: None }));
}

#[test]
fn draw_then_swap_multi_success() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[4, 4, 9, 9], &[]);
    let before = s.players[0].slots.clone();
    s.phase = Phase::Turn { current: 0, pending: None };
    // 摸一张（摸牌堆顶强制为 6），然后与两张 4 交换 → 成功，手牌 4 → 3。
    let drawn = put_on_deck_top(&mut s, 6);
    s.apply(0, &Command::BeginDraw).unwrap();
    s.apply(0, &Command::DrawSwap { slots: vec![0, 1] }).unwrap();
    let hand = &s.players[0].slots;
    assert_eq!(hand.len(), 3, "两张换一张");
    assert_eq!(hand, &vec![drawn, before[2], before[3]], "新牌保留首个被选位置，压缩多出的空位");
    // 换出的两张 4 在弃牌堆顶且公开。
    assert!(s.cards[*s.discard.last().unwrap() as usize].revealed);
    assert_eq!(s.cards[*s.discard.last().unwrap() as usize].card.rank, 4);
    // 换入的牌仅自己知道。
    assert_eq!(s.cards[drawn as usize].known_by, [0usize].into_iter().collect());
    assert!(matches!(s.phase, Phase::Turn { current: 1, .. }));
}

#[test]
fn draw_then_swap_multi_failure_expands_hand() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[5, 6, 2, 2], &[]);
    s.phase = Phase::Turn { current: 0, pending: None };
    // 摸一张（摸牌堆顶强制为另一张 6），与槽位 0(5)、1(6) 交换 → 失败。
    let secret = put_on_deck_top(&mut s, 6);
    s.apply(0, &Command::BeginDraw).unwrap();
    s.apply(0, &Command::DrawSwap { slots: vec![0, 1] }).unwrap();
    let hand = &s.players[0].slots;
    assert_eq!(hand.len(), 5, "失败：新牌追加入手");
    // 选中的牌明置并留在手牌区。
    let five = hand[0];
    let six = hand[1];
    assert!(s.cards[five as usize].revealed);
    assert_eq!(s.cards[five as usize].known_by.len(), 2);
    assert!(s.cards[six as usize].revealed);
    // 新牌来自摸牌堆：暗置且仅自己知道，追加到末尾。
    assert_eq!(hand[4], secret);
    assert!(!s.cards[secret as usize].revealed);
    assert_eq!(s.cards[secret as usize].known_by, [0usize].into_iter().collect());
    assert!(matches!(s.phase, Phase::Turn { current: 1, pending: None }));
}

#[test]
fn non_adjacent_group_keeps_first_selected_position() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[2, 4, 9, 4], &[]);
    let before = s.players[0].slots.clone();
    let drawn = put_on_deck_top(&mut s, 6);
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::BeginDraw).unwrap();
    s.apply(0, &Command::DrawSwap { slots: vec![3, 1] }).unwrap();
    assert_eq!(s.players[0].slots, vec![before[0], drawn, before[2]]);
}

#[test]
fn visibility_distinguishes_private_shared_and_public_knowledge() {
    let mut s = human_session(3);
    s.start_game().unwrap();
    peek_all(&mut s);
    let cards = s.players[0].slots.clone();
    s.cards[cards[1] as usize].known_by = [0, 1].into_iter().collect();
    s.cards[cards[2] as usize].known_by = [0, 1, 2].into_iter().collect();
    for memory_mode in [false, true] {
        s.settings.memory_mode = memory_mode;
        let own = project(&s, Some(0), 0).me_seat.unwrap().slots;
        assert_eq!(own[0].visibility_label(), "仅你已知");
        assert_eq!(own[1].visibility_label(), "部分人已知");
        assert_eq!(own[2].visibility_label(), "公开");
        assert!(!own[2].revealed, "来自弃牌堆的暗置牌也可以全员已知");
        assert_eq!(own[0].shown, !memory_mode);
        assert!(own[2].shown);
        let other = project(&s, Some(2), 0).all_seats[0].slots.clone();
        assert_eq!(other[0].value, None);
        assert_eq!(other[1].value, None);
        assert!(other[2].shown);
        let spect = project(&s, None, 0).all_seats[0].slots.clone();
        assert_eq!(spect[0].value, None);
        assert_eq!(spect[1].value, None);
        assert!(spect[2].shown);
    }
}

#[test]
fn swap_failure_from_discard_is_public() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[5, 6, 2, 2], &[]);
    s.phase = Phase::Turn { current: 0, pending: None };
    // 从弃牌堆换入，选择两张不同的牌 → 失败；新牌来自弃牌堆，K=全体。
    s.apply(0, &Command::BeginSwap).unwrap();
    s.apply(0, &Command::SwapToggle { slot: 0 }).unwrap();
    s.apply(0, &Command::SwapToggle { slot: 1 }).unwrap();
    s.apply(0, &Command::SwapCommit).unwrap();
    assert_eq!(s.players[0].slots.len(), 5);
    let new_card = s.players[0].slots[4];
    assert!(!s.cards[new_card as usize].revealed, "新牌暗置");
    assert_eq!(s.cards[new_card as usize].known_by.len(), 2, "但来自弃牌堆，K=全体");
    let five = s.players[0].slots[0];
    assert!(s.cards[five as usize].revealed, "选中的牌明置");
    assert!(matches!(s.phase, Phase::Turn { current: 1, pending: None }));
}

#[test]
fn draw_action_discard_only_with_power() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::BeginDraw).unwrap();
    let drawn = match &s.phase {
        Phase::Turn { pending: Some(Pending::Drew { card }), .. } => *card,
        other => panic!("unexpected {other:?}"),
    };
    assert!(s.cards[drawn as usize].known_by.contains(&0));
    assert_eq!(s.cards[drawn as usize].known_by.len(), 1);
    s.apply(0, &Command::DiscardDrawn { power: None }).unwrap();
    assert_eq!(s.discard.last(), Some(&drawn));
    assert!(s.cards[drawn as usize].revealed);

    // 7 偷看能力。
    let seven = put_on_deck_top(&mut s, 7);
    s.phase = Phase::Turn { current: 1, pending: None };
    s.apply(1, &Command::BeginDraw).unwrap();
    s.apply(1, &Command::DiscardDrawn { power: Some(PowerUse::PeekOwn { slot: 0 }) }).unwrap();
    let peeked = s.players[1].slots[0];
    assert!(s.cards[peeked as usize].known_by.contains(&1));
    assert!(!s.cards[peeked as usize].known_by.contains(&0));
    assert_eq!(s.discard.last(), Some(&seven));
}

#[test]
fn spy_and_swap_powers() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    let nine = put_on_deck_top(&mut s, 9);
    s.phase = Phase::Turn { current: 1, pending: None };
    s.apply(1, &Command::BeginDraw).unwrap();
    s.apply(1, &Command::DiscardDrawn { power: Some(PowerUse::Spy { player: 0, slot: 3 }) }).unwrap();
    let seen = s.players[0].slots[3];
    assert!(s.cards[seen as usize].known_by.contains(&1));
    assert_eq!(s.discard.last(), Some(&nine));

    let jack = put_on_deck_top(&mut s, 11);
    s.phase = Phase::Turn { current: 0, pending: None };
    let p0s1 = s.players[0].slots[1];
    let p1s2 = s.players[1].slots[2];
    s.apply(0, &Command::BeginDraw).unwrap();
    s.apply(
        0,
        &Command::DiscardDrawn { power: Some(PowerUse::Swap { my_slot: 1, player: 1, slot: 2 }) },
    )
    .unwrap();
    assert_eq!(s.players[0].slots[1], p1s2);
    assert_eq!(s.players[1].slots[2], p0s1);
    assert_eq!(s.discard.last(), Some(&jack));
}

#[test]
fn extra_turns_then_no_more_cabo() {
    let mut s = human_session(3);
    s.start_game().unwrap();
    peek_all(&mut s);
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::CallCabo).unwrap();
    assert!(matches!(s.phase, Phase::Turn { current: 1, .. }));
    // 加时中不能再宣告 Cabo。
    assert!(s.apply(1, &Command::CallCabo).is_err());
    // 未轮到的玩家不能行动。
    assert!(s.apply(2, &Command::BeginDraw).is_err());
    idle_via_draw_discard(&mut s, 1);
    assert!(matches!(s.phase, Phase::Turn { current: 2, .. }));
    idle_via_draw_discard(&mut s, 2);
    assert!(matches!(s.phase, Phase::RoundEnd));
}

#[test]
fn cabo_caller_strictly_lowest_scores_zero() {
    let mut s = human_session(3);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[0, 0, 1, 1], &[]); // 2 分
    force_hand(&mut s, 1, &[10, 10, 5, 5], &[0]); // 30 分
    force_hand(&mut s, 2, &[13, 13, 7, 7], &[0, 1]); // 40 分
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::CallCabo).unwrap();
    for pid in [1usize, 2] {
        idle_via_draw_discard(&mut s, pid);
    }
    assert!(matches!(s.phase, Phase::RoundEnd));
    assert_eq!(s.players[0].round_score, Some(0), "严格最低得 0 分");
    assert_eq!(s.players[1].round_score, Some(30));
    assert_eq!(s.players[2].round_score, Some(40));
    assert_eq!(s.players[0].total_score, 0);
}

#[test]
fn cabo_tie_gets_penalty() {
    let mut s = human_session(3);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[2, 2, 3, 3], &[]); // 10 分
    force_hand(&mut s, 1, &[2, 3, 2, 3], &[0]); // 10 分（并列，非严格最低）
    force_hand(&mut s, 2, &[13, 13, 7, 7], &[0, 1]); // 40 分
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::CallCabo).unwrap();
    for pid in [1usize, 2] {
        idle_via_draw_discard(&mut s, pid);
    }
    assert_eq!(s.players[0].round_score, Some(10 + 10), "并列最低要吃惩罚");
    assert_eq!(s.players[1].round_score, Some(10));
    assert_eq!(s.players[2].round_score, Some(40));
}

#[test]
fn game_over_at_threshold_and_rematch() {
    let mut s = Session::new_lobby(
        7,
        Settings { cabo_penalty: 10, target_score: 20, memory_mode: false },
        (0..2)
            .map(|i| PlayerState {
                name: format!("P{i}"),
                is_ai: false,
                bot_id: String::new(),
                token: Some(format!("t{i}")),
                slots: Vec::new(),
                peeked_slots: Vec::new(),
                total_score: 0,
                round_score: None,
                score_reset_used: false,
                score_reset_this_round: false,
            })
            .collect(),
    );
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[0, 0, 1, 1], &[]); // 2 分
    force_hand(&mut s, 1, &[13, 13, 11, 12], &[0]); // 普通高分 49 ≥ 20；避开高牌双对组合。
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::CallCabo).unwrap();
    idle_via_draw_discard(&mut s, 1);
    match &s.phase {
        Phase::GameOver { winners } => assert_eq!(winners, &vec![0usize]),
        other => panic!("expected game over, got {other:?}"),
    }
    s.rematch().unwrap();
    assert!(matches!(s.phase, Phase::Peeking { .. }));
    assert!(s.players.iter().all(|p| p.total_score == 0));
}

#[test]
fn deck_exhaustion_ends_round_immediately() {
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    s.deck.truncate(1);
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::BeginDraw).unwrap();
    s.apply(0, &Command::DiscardDrawn { power: None }).unwrap();
    assert!(matches!(s.phase, Phase::RoundEnd));
}

#[test]
fn expanded_hand_is_scored() {
    // 交换失败追加的牌也计入总分。
    let mut s = human_session(2);
    s.start_game().unwrap();
    peek_all(&mut s);
    force_hand(&mut s, 0, &[5, 6, 2, 2], &[]);
    s.phase = Phase::Turn { current: 0, pending: None };
    s.apply(0, &Command::BeginSwap).unwrap();
    s.apply(0, &Command::SwapToggle { slot: 0 }).unwrap();
    s.apply(0, &Command::SwapToggle { slot: 1 }).unwrap();
    s.apply(0, &Command::SwapCommit).unwrap();
    assert_eq!(s.players[0].slot_count(), 5);
    let sum0: u32 =
        s.players[0].slots.iter().map(|c| s.cards[*c as usize].card.rank as u32).sum();
    assert!(sum0 >= 5 + 6, "追加的牌计入总分");
}

#[test]
fn bot_games_run_to_completion() {
    let registry = BotRegistry::with_builtins();
    for seed in 1..=8u64 {
        let mut s =
            make_ai_session(seed, Settings::default(), &["challenger", "challenger", "challenger", "challenger"]);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(7919));
        let outcome = run_game(&mut s, &registry, &mut rng, 50_000)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert_eq!(outcome.totals.len(), 4);
        assert!(outcome.rounds >= 1);
        assert!(outcome.totals.iter().all(|&t| t > 0), "总分应非零（有人达到阈值才结束）");
    }
}

#[test]
fn two_player_bot_games_run_to_completion() {
    let registry = BotRegistry::with_builtins();
    for seed in 11..=14u64 {
        let mut s = make_ai_session(seed, Settings::default(), &["challenger", "challenger"]);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(104729));
        run_game(&mut s, &registry, &mut rng, 50_000).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
}
