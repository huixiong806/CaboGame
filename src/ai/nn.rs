//! 极简神经网络：全连接 + ReLU + sigmoid，手写前向/反向 + Adam。
//!
//! 为什么手写：本项目是"单二进制、零运行时依赖"的部署形态，
//! 引入 libtorch / candle 会破坏这个前提（也需要下载模型运行时）。
//! 网络很小（输入 ~50 维，两层 64 隐层，约 8k 参数），手写完全够用，
//! 训练几百局自我对弈的数据只需几秒。

/// 一层全连接：`y = act(W x + b)`。
#[derive(Clone, Debug)]
pub struct Layer {
    pub w: Vec<f32>, // [out][in]
    pub b: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, rng: &mut dyn rand::RngCore, scale: f32) -> Layer {
        use rand::Rng;
        let mut w = vec![0f32; n_in * n_out];
        for v in w.iter_mut() {
            *v = (rng.random::<f32>() * 2.0 - 1.0) * scale;
        }
        Layer { w, b: vec![0f32; n_out], n_in, n_out }
    }

    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        for o in 0..self.n_out {
            let row = &self.w[o * self.n_in..(o + 1) * self.n_in];
            let mut acc = self.b[o];
            for i in 0..self.n_in {
                acc += row[i] * x[i];
            }
            out[o] = acc;
        }
    }
}

/// 激活函数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Relu,
    Sigmoid,
    Tanh,
    /// 线性（输出 logit，供"残差价值"之类的外部组合使用）。
    Linear,
}

impl Act {
    fn f(self, x: f32) -> f32 {
        match self {
            Act::Relu => x.max(0.0),
            Act::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Act::Tanh => x.tanh(),
            Act::Linear => x,
        }
    }
    fn df(self, y: f32) -> f32 {
        match self {
            Act::Relu => {
                if y > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Act::Sigmoid => y * (1.0 - y),
            Act::Tanh => 1.0 - y * y,
            Act::Linear => 1.0,
        }
    }
}

/// 多层感知机（最后一层输出标量 logit）。
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Layer>,
    pub acts: Vec<Act>,
}

/// 一次前向的中间缓存（训练时用）。
struct Cache {
    xs: Vec<Vec<f32>>, // 每层输入
    ys: Vec<Vec<f32>>, // 每层激活后输出
}

impl Mlp {
    /// `sizes = [n_in, h1, h2, ..., 1]`。
    pub fn new(sizes: &[usize], rng: &mut dyn rand::RngCore) -> Mlp {
        let mut layers = Vec::new();
        let mut acts = Vec::new();
        for i in 0..sizes.len() - 1 {
            let scale = (6.0 / (sizes[i] + sizes[i + 1]) as f32).sqrt();
            layers.push(Layer::new(sizes[i], sizes[i + 1], rng, scale));
            acts.push(if i + 2 == sizes.len() { Act::Sigmoid } else { Act::Relu });
        }
        Mlp { layers, acts }
    }

    /// 前向：返回输出层的激活值（sigmoid → 概率）。`cache` 为 None 时不记录中间量。
    pub fn forward(&self, x: &[f32], mut cache: Option<&mut Cache>) -> Vec<f32> {
        let mut cur = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.as_deref_mut() {
                c.xs.push(cur.clone());
            }
            let mut out = vec![0f32; layer.n_out];
            layer.forward(&cur, &mut out);
            let act = self.acts[i];
            for v in out.iter_mut() {
                *v = act.f(*v);
            }
            if let Some(c) = cache.as_deref_mut() {
                c.ys.push(out.clone());
            }
            cur = out;
        }
        cur
    }

    pub fn predict(&self, x: &[f32]) -> f32 {
        self.forward(x, None)[0]
    }

    /// 反向传播：给定一条样本 (x, target) 与 BCE 损失，把梯度累加到 `grads`。
    pub fn backward(
        &self,
        x: &[f32],
        target: f32,
        grads: &mut Mlp,
        l2: f32,
    ) -> f32 {
        let out = self.forward(x, None)[0];
        let eps = 1e-6f32;
        let p = out.clamp(eps, 1.0 - eps);
        let loss = -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());
        self.backward_delta(x, p - target, grads, l2);
        loss
    }

    /// 反向传播（外部给定输出层梯度）：用于"残差价值"这种
    /// `p = sigmoid(解析基线 + 网络输出)` 的组合模型。
    pub fn backward_delta(&self, x: &[f32], delta_out: f32, grads: &mut Mlp, l2: f32) {
        let mut cache = Cache { xs: Vec::new(), ys: Vec::new() };
        self.forward(x, Some(&mut cache));
        let mut delta = vec![delta_out];

        for li in (0..self.layers.len()).rev() {
            let layer = &self.layers[li];
            let g = &mut grads.layers[li];
            let xin = &cache.xs[li];
            // 当前层的输入梯度（传递给上一层）
            let mut dx = vec![0f32; layer.n_in];
            for o in 0..layer.n_out {
                let d = delta[o];
                if d == 0.0 {
                    continue;
                }
                let row = o * layer.n_in;
                for i in 0..layer.n_in {
                    dx[i] += layer.w[row + i] * d;
                    g.w[row + i] += d * xin[i];
                }
                g.b[o] += d;
            }
            // L2 正则
            if l2 > 0.0 {
                for (i, w) in g.w.iter_mut().enumerate() {
                    *w += l2 * layer.w[i];
                }
            }
            if li > 0 {
                // 过上一层的激活导数
                let prev_y = &cache.ys[li - 1];
                let mut next = vec![0f32; layer.n_in];
                for i in 0..layer.n_in {
                    next[i] = dx[i] * self.acts[li - 1].df(prev_y[i]);
                }
                delta = next;
            }
        }
    }

    pub fn params(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    /// 导出为扁平参数向量（用于保存/加载）。
    pub fn export(&self) -> Vec<f32> {
        let mut out = Vec::new();
        out.push(self.layers.len() as f32);
        for l in &self.layers {
            out.push(l.n_in as f32);
            out.push(l.n_out as f32);
            out.extend_from_slice(&l.w);
            out.extend_from_slice(&l.b);
        }
        out
    }

    pub fn import(data: &[f32]) -> Option<Mlp> {
        let mut it = data.iter().copied();
        let n = it.next()? as usize;
        let mut layers = Vec::new();
        let mut acts = Vec::new();
        for i in 0..n {
            let n_in = it.next()? as usize;
            let n_out = it.next()? as usize;
            let w: Vec<f32> = it.by_ref().take(n_in * n_out).collect();
            let b: Vec<f32> = it.by_ref().take(n_out).collect();
            if w.len() != n_in * n_out || b.len() != n_out {
                return None;
            }
            layers.push(Layer { w, b, n_in, n_out });
            acts.push(if i + 1 == n { Act::Sigmoid } else { Act::Relu });
        }
        Some(Mlp { layers, acts })
    }
}

/// Adam 优化器状态。
pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    t: u32,
}

impl Adam {
    pub fn new(n: usize, lr: f32) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], lr, b1: 0.9, b2: 0.999, eps: 1e-8, t: 0 }
    }

    /// 用累积梯度更新参数（并把梯度清零）。
    pub fn step(&mut self, model: &mut Mlp, grads: &mut Mlp, batch: f32) {
        self.t += 1;
        let bc1 = 1.0 - self.b1.powi(self.t as i32);
        let bc2 = 1.0 - self.b2.powi(self.t as i32);
        let mut idx = 0usize;
        for li in 0..model.layers.len() {
            let n_in = model.layers[li].n_in;
            let n_out = model.layers[li].n_out;
            for k in 0..(n_in * n_out + n_out) {
                let is_bias = k >= n_in * n_out;
                let (g, p) = if is_bias {
                    let o = k - n_in * n_out;
                    (&mut grads.layers[li].b[o], &mut model.layers[li].b[o])
                } else {
                    (&mut grads.layers[li].w[k], &mut model.layers[li].w[k])
                };
                let gval = *g / batch;
                *g = 0.0;
                self.m[idx] = self.b1 * self.m[idx] + (1.0 - self.b1) * gval;
                self.v[idx] = self.b2 * self.v[idx] + (1.0 - self.b2) * gval * gval;
                let mh = self.m[idx] / bc1;
                let vh = self.v[idx] / bc2;
                *p -= self.lr * mh / (vh.sqrt() + self.eps);
                idx += 1;
            }
        }
    }
}

impl Mlp {
    pub fn zero_grad(&self) -> Mlp {
        Mlp {
            layers: self
                .layers
                .iter()
                .map(|l| Layer { w: vec![0.0; l.w.len()], b: vec![0.0; l.b.len()], n_in: l.n_in, n_out: l.n_out })
                .collect(),
            acts: self.acts.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// 梯度检查：数值梯度 ≈ 反向传播梯度。
    #[test]
    fn gradient_is_correct() {
        let mut rng = StdRng::seed_from_u64(7);
        let net = Mlp::new(&[5, 6, 4, 1], &mut rng);
        let x = [0.3f32, -0.7, 0.2, 0.9, -0.1];
        let target = 0.75f32;
        let mut grads = net.zero_grad();
        net.backward(&x, target, &mut grads, 0.0);
        // 对前 3 个权重做数值梯度
        for k in [0usize, 5, 17, 40] {
            let mut plus = net.clone();
            let mut minus = net.clone();
            let eps = 1e-3f32;
            // 找到 k 对应的 (li, idx)
            let mut idx = 0usize;
            let mut found = None;
            for (li, l) in net.layers.iter().enumerate() {
                if k < idx + l.w.len() {
                    found = Some((li, k - idx));
                    break;
                }
                idx += l.w.len();
            }
            let Some((li, wi)) = found else { continue };
            plus.layers[li].w[wi] += eps;
            minus.layers[li].w[wi] -= eps;
            let lp = {
                let p = plus.predict(&x).clamp(1e-6, 1.0 - 1e-6);
                -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
            };
            let lm = {
                let p = minus.predict(&x).clamp(1e-6, 1.0 - 1e-6);
                -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
            };
            let numeric = (lp - lm) / (2.0 * eps);
            let analytic = grads.layers[li].w[wi];
            assert!(
                (numeric - analytic).abs() < 1e-2 * (1.0 + numeric.abs()),
                "梯度不符：数值 {numeric} vs 反向 {analytic}"
            );
        }
    }

    /// 小规模拟合：学一个 XOR 型的非线性关系。
    #[test]
    fn trains_xor() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut net = Mlp::new(&[2, 8, 8, 1], &mut rng);
        let data = [
            ([0f32, 0f32], 0f32),
            ([0f32, 1f32], 1f32),
            ([1f32, 0f32], 1f32),
            ([1f32, 1f32], 0f32),
        ];
        let mut opt = Adam::new(net.params(), 0.05);
        for _ in 0..3000 {
            let mut grads = net.zero_grad();
            for (x, y) in &data {
                net.backward(x, *y, &mut grads, 0.0);
            }
            opt.step(&mut net, &mut grads, data.len() as f32);
        }
        for (x, y) in &data {
            let p = net.predict(x);
            assert!((p - y).abs() < 0.15, "XOR 没学会：{x:?} → {p}（期望 {y}）");
        }
    }

    /// 导出/导入必须无损。
    #[test]
    fn export_import_roundtrip() {
        let mut rng = StdRng::seed_from_u64(11);
        let net = Mlp::new(&[4, 5, 3, 1], &mut rng);
        let data = net.export();
        let back = Mlp::import(&data).unwrap();
        let x = [0.1f32, 0.5, -0.3, 0.8];
        assert!((net.predict(&x) - back.predict(&x)).abs() < 1e-6);
    }
}
