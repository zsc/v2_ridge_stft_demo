# 压缩 STFT 版一阶段参数化

这份说明对应 `vq06-stft` 路径，目标定义对齐 [VQ06_STFT_PAIR_DATASET_20260401.md](/Users/zhousc6/Downloads/wav2vq0206/VQ06_STFT_PAIR_DATASET_20260401.md) 里第 `3.2` 节：

- `n_fft = 400`
- `hop = 160`
- `window = hann`
- `S = |STFT(x)|^2`
- 去掉最后一帧
- `log_spec = log10(clamp(S, min=1e-10))`
- `log_spec = max(log_spec, max(log_spec) - 8.0)`
- `compressed_stft = (log_spec + 4.0) / 4.0`

## 这次算法怎么改

和 mel 版相比，有两件事最重要：

1. 频率轴不再经过 `mel` 非线性拉伸  
   所以 ridge 模板不需要“先回线性频率，再投影回 mel”，可以直接在 STFT bin 轴上做固定宽度加粗。

2. 存储域变成了压缩后的 log-power  
   这个域里会出现负值，而且加法不再严格对应物理能量叠加。

因此这里采用的是一个更稳的两段式做法：

## 推荐做法

### A. 在压缩域上做 support 选择

先把压缩 STFT 做一个全局 floor 平移：

`shifted = compressed_stft - min(compressed_stft)`

这样 `shifted >= 0`，可以继续沿用当前一阶段的：

- 峰值检测
- `top-k` 稀疏 support
- 时间方向 TV 正则

也就是：

`compressed_stft -> shifted -> peak_mask -> ridge_optimize -> center_idx`

### B. 在 STFT power 域上做幅度拟合

然后把压缩 STFT 反解回“已经过 floor clamp 的 power”：

`power_clamped = 10 ** (4 * compressed_stft - 4)`

接着直接在 STFT bin 轴上构造模板：

`R_power[:, t] = sum_i A[i, t] * B[:, C[i, t]]`

其中：

- `C[k, T]` 是离散中心索引
- `A[k, T]` 是非负强度
- `B[F, F]` 是 STFT bin 轴上的固定宽度模板库

这里的 `B` 直接在 STFT bin 轴上高斯加粗即可，因为不再有 mel 轴扭曲。

## 为什么推荐这样做

如果直接在压缩域里做线性叠加，虽然也能用，但它只是一种近似，因为：

- `log` 压缩破坏了线性可加性
- 一个 ridge 单独渲染后的压缩值，和它在混合谱里的压缩值不是同一个量

而上面这个“两段式”保留了更合理的物理语义：

- support 在压缩域里选，数值稳定
- 强度在 power 域里拟合，分解可加
- 最后如果要回到训练目标域，再用**原样本的 reference max** 压回压缩 STFT

## 关键差异

mel 版模板：

- 中心在 mel bin
- 实际宽度先定义在线性频率，再映射回 mel
- 高频在 mel 轴上会更“挤”

STFT 版模板：

- 中心直接在 STFT bin
- 模板宽度是全频段统一的 `sigma_bins`
- 谐波间距在频率轴上更规整，后续如果你要做 harmonic comb / f0 约束，会比 mel 版自然

## 代码位置

- 核心实现：`stage1_parametric/compressed_stft_stage1.py`
- 验证脚本：`stage1_parametric/verify_compressed_stft_stage1.py`

核心函数包括：

- `whisper_power_spectrogram(...)`
- `compress_whisper_power(...)`
- `decompress_whisper_compressed_stft(...)`
- `shift_compressed_stft_for_support(...)`
- `build_thickened_stft_basis(...)`
- `fit_topk_thickened_stft_bands(...)`
- `render_stft_power_from_slots_torch(...)`

## 建议你在另一个可微项目里怎么接

如果你已经直接拿到了 dequantize 后的 `vq06-stft` 浮点张量，推荐接法是：

1. 输入 `compressed_stft`
2. 用 `shift_compressed_stft_for_support(...)` 或你自己的网络，得到固定 `center_idx`
3. 把 `compressed_stft` 反成 `power_clamped`
4. 在 power 域里渲染 ridge 参数
5. 如果训练目标仍是 `vq06-stft`，最后再用 `compress_whisper_power_torch(...)` 压回去

这样既对齐数据定义，也保留了可微训练路径。
