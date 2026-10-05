//! Whole-match winner probabilities learned from real-engine boundary states.
//! Features are only public totals, reset eligibility, player count and scoring settings.
use std::{path::Path, sync::Arc};

#[derive(Debug)]
pub(super) struct MatchValue {
    weights: Vec<f32>,
}

impl MatchValue {
    pub fn load(path: &Path) -> Result<Arc<Self>, String> {
        Self::decode(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
            .map(Arc::new)
    }
    pub(super) fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 8 + 3425 * 4 || &bytes[..8] != b"CABOMV01" {
            return Err("invalid match-value model format/size".into());
        }
        let weights: Vec<_> = bytes[8..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if weights.iter().any(|x| !x.is_finite() || x.abs() > 100.) {
            return Err("invalid match-value model weights".into());
        }
        Ok(Self { weights })
    }
    pub fn predict(
        &self,
        totals: &[u32],
        used: &[bool],
        target: u32,
        penalty: u32,
    ) -> Option<Vec<f64>> {
        // Training distribution is the default rules; other settings retain frozen utility.
        if target != 100
            || penalty != 10
            || !(2..=4).contains(&totals.len())
            || totals.iter().any(|&s| s >= target)
            || used.len() != totals.len()
        {
            return None;
        }
        let n = totals.len();
        let min = *totals.iter().min().unwrap() as f32;
        let max = *totals.iter().max().unwrap() as f32;
        let mut h2 = vec![[0.; 32]; n];
        for p in 0..n {
            let x = [
                totals[p] as f32 / 100.,
                used[p] as u8 as f32,
                target as f32 / 100.,
                penalty as f32 / 10.,
                n as f32 / 4.,
                (totals[p] as f32 - min) / 100.,
                (target as f32 - max) / 100.,
            ];
            let mut h1 = [0.; 32];
            self.layer(&x, &mut h1, 0, true);
            self.layer(&h1, &mut h2[p], 256, true);
        }
        let mut pooled = [0.; 32];
        for h in &h2 {
            for j in 0..32 {
                pooled[j] += h[j] / n as f32;
            }
        }
        let temp = (8. * ((target as f32 - max) / 8.).sqrt()).max(6.);
        let mut logits = Vec::with_capacity(n);
        for p in 0..n {
            let mut cat = [0.; 64];
            cat[..32].copy_from_slice(&h2[p]);
            cat[32..].copy_from_slice(&pooled);
            let mut h3 = [0.; 32];
            self.layer(&cat, &mut h3, 1312, true);
            let mut out = [0.];
            self.layer(&h3, &mut out, 3392, false);
            logits.push((out[0] - (totals[p] as f32 - min) / temp) as f64);
        }
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = logits.iter().map(|l| (l - top).exp()).collect();
        let sum: f64 = weights.iter().sum();
        Some(weights.iter().map(|w| w / sum).collect())
    }
    fn layer(&self, x: &[f32], out: &mut [f32], offset: usize, relu: bool) {
        let w = &self.weights[offset..];
        for o in 0..out.len() {
            let mut v = w[out.len() * x.len() + o];
            for i in 0..x.len() {
                v += w[o * x.len() + i] * x[i];
            }
            out[o] = if relu { v.max(0.) } else { v };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_corrupt_model() {
        assert!(MatchValue::decode(b"CABOMV01").is_err());
        let mut bytes = b"CABOMV01".to_vec();
        bytes.resize(8 + 3425 * 4, 0);
        bytes[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(MatchValue::decode(&bytes).is_err());
    }
    #[test]
    #[ignore = "requires locally trained model; compares independent Python/Rust implementations"]
    fn local_model_inference_parity_and_seat_symmetry() {
        let path =
            std::env::var("CABO_MATCH_VALUE_TEST").unwrap_or("data/match_value/model.bin".into());
        let model = MatchValue::load(Path::new(&path)).unwrap();
        let cases =
            std::fs::read_to_string(Path::new(&path).with_extension("predictions.tsv")).unwrap();
        for line in cases.lines() {
            let x: Vec<f64> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let n = x[0] as usize;
            let totals: Vec<_> = x[3..3 + n].iter().map(|&v| v.round() as u32).collect();
            let used: Vec<_> = x[7..7 + n].iter().map(|&v| v != 0.).collect();
            let pred = model
                .predict(&totals, &used, x[1].round() as u32, x[2].round() as u32)
                .unwrap();
            for p in 0..n {
                assert!((pred[p] - x[11 + p]).abs() < 2e-6, "{pred:?} {x:?}");
            }
            let reverse_totals: Vec<_> = totals.iter().rev().copied().collect();
            let reverse_used: Vec<_> = used.iter().rev().copied().collect();
            let reverse = model
                .predict(&reverse_totals, &reverse_used, 100, 10)
                .unwrap();
            for p in 0..n {
                assert!((pred[p] - reverse[n - 1 - p]).abs() < 2e-6);
            }
            assert!((pred.iter().sum::<f64>() - 1.).abs() < 1e-9);
        }
        assert!(model.predict(&[0, 0], &[false, false], 80, 10).is_none());
    }
}
