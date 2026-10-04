//! 网络前向性能基准：决定"策略网络能不能进 PIMC rollout"。
//!
//! `cargo run --release --example nn_bench`

use cabo::ai::nn::Mlp;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn bench(sizes: &[usize], reps: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(7);
    let net = Mlp::new(sizes, &mut rng);
    let x: Vec<f32> = (0..sizes[0]).map(|i| (i as f32 * 0.01).sin()).collect();
    // 预热
    for _ in 0..1000 {
        std::hint::black_box(net.predict(&x));
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(net.predict(&x));
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let macs: usize = sizes.windows(2).map(|w| w[0] * w[1]).sum();
    println!(
        "{:>28}  参数 {:>8}  MAC {:>8}  {:.2} µs/次  ({:.2} GFLOP/s)",
        format!("{sizes:?}"),
        net.params(),
        macs,
        dt * 1e6,
        macs as f64 * 2.0 / dt / 1e9
    );
    dt
}

fn main() {
    println!("=== 单次前向耗时（决定 rollout 里能不能用）===");
    // 启发式策略在 rollout 里约 2~5 µs/次，这里给出可比的口径
    bench(&[198, 64, 64, 1], 200_000);
    bench(&[198, 96, 96, 1], 200_000);
    bench(&[198, 128, 128, 1], 100_000);
    bench(&[198, 256, 256, 256, 1], 50_000);
    println!();
    println!("参考：手写启发式策略 ≈ 2~5 µs/次；一次 rollout 约 30 步 →");
    let dt64 = bench(&[198, 64, 64, 1], 200_000);
    let dt256 = bench(&[198, 256, 256, 256, 1], 50_000);
    println!(
        "  64 宽网络：每步 {:.1}µs，一次 30 步 rollout 额外 {:.2}ms",
        dt64 * 1e6,
        dt64 * 30.0 * 1e3
    );
    println!(
        "  256 宽网络：每步 {:.1}µs，一次 30 步 rollout 额外 {:.2}ms",
        dt256 * 1e6,
        dt256 * 30.0 * 1e3
    );
}
