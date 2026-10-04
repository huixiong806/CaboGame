//! Shared settlement rules for the real engine and decision simulators.

pub const RESET_AT: u32 = 100;
pub const RESET_TO: u32 = 50;
pub const HIGH_PAIRS_OTHER_SCORE: u32 = 50;

/// The whole hand must be exactly two 12s and two 13s, in any order.
pub fn is_high_pairs(hand: &[u8]) -> bool {
    hand.len() == 4
        && hand.iter().filter(|&&r| r == 12).count() == 2
        && hand.iter().filter(|&&r| r == 13).count() == 2
}

#[derive(Debug)]
pub struct Settlement {
    pub sums: Vec<u32>,
    pub scores: Vec<u32>,
    pub totals: Vec<u32>,
    pub reset_used: Vec<bool>,
    pub reset_triggered: Vec<bool>,
    pub high_pairs: Option<usize>,
}

/// Resolve special/ordinary round scores, then accumulate and apply each player's
/// once-per-match exact-100 reset. The caller checks the end threshold afterwards.
pub fn settle(
    hands: &[Vec<u8>],
    totals: &[u32],
    reset_used: &[bool],
    caller: Option<usize>,
    penalty: u32,
) -> Settlement {
    assert_eq!(hands.len(), totals.len());
    assert_eq!(hands.len(), reset_used.len());
    let sums: Vec<u32> = hands
        .iter()
        .map(|h| h.iter().map(|&r| r as u32).sum())
        .collect();
    // The deck contains only two 13s, so at most one player can have this hand.
    let high_pairs = hands.iter().position(|h| is_high_pairs(h));
    let mut scores = sums.clone();
    if let Some(k) = high_pairs {
        scores.fill(HIGH_PAIRS_OTHER_SCORE);
        scores[k] = 0;
    } else if let Some(c) = caller {
        let other_min = sums
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != c)
            .map(|(_, &s)| s)
            .min()
            .unwrap_or(u32::MAX);
        scores[c] = if sums[c] < other_min {
            0
        } else {
            sums[c] + penalty
        };
    }
    let mut result = Settlement {
        sums,
        scores,
        totals: totals.to_vec(),
        reset_used: reset_used.to_vec(),
        reset_triggered: vec![false; hands.len()],
        high_pairs,
    };
    for p in 0..hands.len() {
        result.totals[p] += result.scores[p];
        if result.totals[p] == RESET_AT && !result.reset_used[p] {
            result.totals[p] = RESET_TO;
            result.reset_used[p] = true;
            result.reset_triggered[p] = true;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_pairs_requires_exact_multiset_and_size() {
        for hand in [
            [12, 12, 13, 13],
            [12, 13, 12, 13],
            [12, 13, 13, 12],
            [13, 12, 12, 13],
            [13, 12, 13, 12],
            [13, 13, 12, 12],
        ] {
            assert!(is_high_pairs(&hand));
        }
        for hand in [
            vec![12, 12, 13, 13, 0],
            vec![12, 13, 13],
            vec![12, 12, 12, 13],
            vec![13, 13, 12, 11, 1],
            vec![],
        ] {
            assert!(!is_high_pairs(&hand));
        }
    }

    #[test]
    fn exact_reset_is_per_player_once_and_applies_after_all_score_sources() {
        let hands = vec![vec![12, 13, 13, 12], vec![0, 0], vec![1], vec![2]];
        let result = settle(
            &hands,
            &[80, 50, 50, 51],
            &[false, false, true, false],
            Some(1),
            17,
        );
        assert_eq!(result.high_pairs, Some(0));
        assert_eq!(result.scores, [0, 50, 50, 50]);
        assert_eq!(result.totals, [80, 50, 100, 101]);
        assert_eq!(result.reset_triggered, [false, true, false, false]);
        assert_eq!(result.reset_used, [false, true, true, false]);

        // Multiple players may each exercise their own entitlement in the same round.
        let result = settle(&[vec![10], vec![20]], &[90, 80], &[false, false], None, 10);
        assert_eq!(result.scores, [10, 20]);
        assert_eq!(result.totals, [50, 50]);
        assert_eq!(result.reset_triggered, [true, true]);
    }
}
