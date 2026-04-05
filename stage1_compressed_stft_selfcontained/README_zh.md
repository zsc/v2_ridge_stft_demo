# 压缩 STFT 一阶段 Self-contained 封装

这个目录把“压缩 Whisper-STFT 上的一阶段 ridge 参数化”收成了一套可以单独拎走的实现。它既是一个工程封装，也是当前这条算法路线的完整说明文档。

如果只看一句话，这套方法做的是：

```text
压缩 STFT
  -> 在压缩域中选出稀疏 ridge 支持
  -> 在 power 域中拟合 ridge 强度
  -> 再压回训练时使用的 compressed STFT
```

这样做的目的，是在尽量保持原始 `vq06-stft` 数据定义的前提下，把谱图里的主导窄带结构提取成一组更紧凑、更稳定、也更容易迁移的参数。

## 1. 这份封装解决的是什么问题

上游数据里的 `vq06-stft` 不是普通线性 STFT，也不是 mel 频谱，而是 Whisper 风格 `power spectrogram` 经过 log 压缩后的结果。精确定义对齐 [VQ06_STFT_PAIR_DATASET_20260401.md](/Users/zhousc6/Downloads/wav2vq0206/VQ06_STFT_PAIR_DATASET_20260401.md) 第 `3.2` 节：

```text
n_fft = 400
hop = 160
window = hann
S = |STFT(x)|^2
drop last frame
log_spec = log10(clamp(S, min=1e-10))
log_spec = max(log_spec, max(log_spec) - 8.0)
compressed = (log_spec + 4.0) / 4.0
```

因此这里的观测谱有几个很关键的性质：

- 它是 `power`，不是 `magnitude`
- 它经过了 `log10` 压缩和动态范围截断
- 它保留了完整 STFT bin 轴，不经过 mel 投影
- 它的数值域更适合做检测和稀疏选择，但不适合直接拿来做“物理意义上的线性叠加”

一阶段方法的目标，就是把这样的 `compressed STFT` 分解成：

- 一组离散的 ridge 中心位置
- 一组非负连续强度
- 一个可以渲染回近似谱图的模板化表示

这比直接保存整张二维谱图更紧凑，也更适合作为后续建模或可微渲染的中间表示。

## 2. 为什么是“两段式”，而不是直接在压缩域拟合

这套实现最核心的设计，不是某个具体优化器，而是“在哪个域做哪件事”：

1. 在压缩域做 support 选择
2. 在 power 域做幅度拟合
3. 最后再回到压缩域和训练目标对齐

原因很简单：

- 压缩域动态范围更稳定，做峰值检测、稀疏选择、时间平滑更容易
- 但压缩域不是线性的，两个 ridge 的压缩值不能简单相加来表示总能量
- power 域保留了可加性，更适合做模板叠加和 NNLS 强度拟合

也就是说，这里故意把“结构选择”和“能量拟合”拆开了：

- 结构问题在压缩域解决
- 幅度问题在功率域解决

这是整套方法能稳定工作的关键。

## 3. 整体算法流程

设：

- `F = 201` 为 STFT bin 数
- `T` 为时间帧数
- `k` 为每帧最多保留的 ridge 槽位数
- `X_cmp in R^{F x T}` 为输入的压缩 STFT

完整流程可以写成：

```text
audio
  -> power = whisper_power_spectrogram(audio)
  -> compressed = compress_whisper_power(power)
  -> shifted = compressed - min(compressed)
  -> peak_mask(shifted)
  -> ridge_optimize(shifted, peak_mask)
  -> center_idx

compressed
  -> power_clamped = decompress_whisper_compressed_stft(compressed)

(center_idx, power_clamped)
  -> basis
  -> NNLS strength fitting
  -> ridge_power
  -> ridge_compressed
  -> Griffin-Lim audio
```

其中：

- `ridge_support` 是一个稀疏的非负 ridge 图，不只是布尔 mask
- `center_idx` 是从 `ridge_support > 0` 提取出来的离散中心索引矩阵
- `slot_strength` 是对这些中心做的非负连续强度拟合
- `ridge_power` 是模板渲染后的功率谱
- `ridge_compressed` 是为了回到训练域而重新压缩得到的谱

## 4. 数学背景与每一步在做什么

### 4.1 从音频得到 Whisper 风格压缩 STFT

核心函数：

- `whisper_power_spectrogram(...)`
- `compress_whisper_power(...)`
- `decompress_whisper_compressed_stft(...)`

先把音频重采样到 `16k`，再计算：

```text
P = |STFT(x)|^2
```

实现上使用：

- `n_fft = 400`
- `hop_length = 160`
- `win_length = 400`
- `window = hann`
- `center = True`

之后做和 Whisper 路径一致的压缩：

```text
L = log10(clamp(P, 1e-10))
L_floor = max(L_max - 8.0, L)
X_cmp = (L_floor + 4.0) / 4.0
```

这里的 `L_max` 是整张谱图的全局最大值。要注意，`decompress_whisper_compressed_stft(...)` 反出来的并不是“原始 power”，而是：

```text
P_clamped = 10 ** (4 * X_cmp - 4)
```

也就是已经过 floor 截断之后的 `power`。这一点很重要，因为后一阶段拟合的目标本来就应该和训练域保持一致，而不是去追一个已经被上游压缩过程抹掉的动态范围。

### 4.2 为什么先做一个全局平移

核心函数：

- `shift_compressed_stft_for_support(...)`

压缩谱可能有负值，而当前 ridge 优化器是建立在非负输入上的，因此先做：

```text
M = X_cmp - min(X_cmp)
```

这样 `M >= 0`，就能继续复用原先一阶段的：

- 峰值检测
- 稀疏 support 选择
- 时间方向总变差约束

这个平移不改变峰的位置关系，只是把数值域变成了更适合优化器的形式。

### 4.3 峰值候选 mask

核心函数：

- `peak_mask(...)`

对每一帧 `M[:, t]`，用 `scipy.signal.find_peaks` 沿频率轴做局部峰值检测，得到：

```text
V in {0, 1}^{F x T}
```

其中 `V[f, t] = 1` 表示这个 bin 在该帧上被允许成为 ridge 候选位置。

它支持几类约束：

- `distance`: 峰之间的最小间隔
- `prominence`: 峰值显著性阈值
- `height`: 峰高阈值
- `dilate`: 是否把峰的上下相邻 bin 也并入候选区域

这一步的作用，是先把搜索空间缩小到“看起来像局部峰”的区域，避免后面的稀疏优化在整张谱上无约束游走。

### 4.4 ridge support 优化

核心函数：

- `ridge_optimize(...)`
- `_solve_ridge_torch(...)`
- `_solve_ridge_cvxpy(...)`

对平移后的谱 `M` 和候选 mask `V`，求解一个带稀疏项和时间 TV 的凸问题：

```text
min_R  0.5 * ||R - M||_2^2
     + lam_sparse * ||R||_1
     + lam_tv * sum_t |R[:, t] - R[:, t-1]|

subject to:
  0 <= R <= M ⊙ V
```

这里：

- 数据项 `||R - M||_2^2` 让 `R` 尽量贴近观测谱
- `L1` 项鼓励稀疏，只保留少数 ridge
- 时间方向 TV 项鼓励相邻帧的 ridge 形状变化平滑
- 约束 `R <= M ⊙ V` 强制解只能落在候选峰值位置上

这个优化的输出 `R` 不是最终的“宽 ridge”，而是一个较细、较稀疏的 ridge support 图。它的正值位置定义了潜在的中心线。

实现上有两个后端：

- `torch` 后端：用原始-对偶 proximal 迭代解同一个凸问题
- `cvxpy` 后端：直接交给凸优化求解器

两者目标函数是一致的，差别只在求解方式。

### 4.5 top-k 截断与 refine

核心函数：

- `_topk_per_frame(...)`
- `ridge_optimize(..., ridge_refine=True)`

原始优化结果可能每帧留下超过 `k` 个非零位置，因此会进一步做：

1. 每帧只保留数值最大的 `top-k`
2. 若开启 `ridge_refine`，再用这个更小的 support 重新解一次同样的优化问题

这样做的直觉是：

- 第一次优化更像“粗筛”
- top-k 把每帧的 ridge 数量卡住
- refine 再在缩小后的支持集上回归一次，能让幅度分配更干净

最终返回的 `ridge_support` 仍然是浮点稀疏图，而不是单纯的二值 mask。

### 4.6 从 support 提取离散中心索引

核心函数：

- `extract_center_idx_matrix(...)`

令：

```text
support = ridge_support > 0
```

则可以得到离散中心矩阵：

```text
C = center_idx in Z^{k x T}
```

语义是：

- `C[i, t] >= 0` 表示第 `t` 帧第 `i` 个槽位对应的 STFT 中心 bin
- `C[i, t] = -1` 表示该槽位为空

在当前实现里，索引默认按频率升序存放。函数名里保留了 `mel_asc` 这个历史命名，但在 STFT 版里本质上就是“bin index 升序”。

### 4.7 在 power 域构造加粗模板库

核心函数：

- `build_thickened_stft_basis(...)`

这一步不再用细线 support 直接作为最终输出，而是把每个中心 bin `j` 对应成一个固定宽度模板 `B[:, j]`。最终得到：

```text
B in R^{F x F}
```

其中第 `j` 列表示“以 bin `j` 为中心的加粗 ridge 模板”。

模板的构造方式是：

1. 从单位脉冲 `eye(F)` 出发
2. 沿频率轴做一维高斯模糊
3. 用 `max(eye, blurred)` 融合细中心和模糊外延
4. 按列做 `peak` 归一化

这里的关键参数是：

- `sigma_bins`: 模板在 STFT bin 轴上的加粗宽度

和 mel 版相比，STFT 版模板更直接：

- 不需要先定义在线性频率再映射回 mel
- 整个频率轴使用统一的 bin 宽度
- 对做 harmonic 结构建模更自然

### 4.8 非负强度拟合

核心函数：

- `fit_strength_matrix_nnls(...)`
- `render_from_slots_numpy(...)`
- `render_from_slots_torch(...)`

对每一帧 `t`，取出该帧有效中心位置 `C_t`，从模板库中抽出对应列：

```text
B_t = B[:, C_t]
```

然后在 `power_clamped[:, t]` 上做逐帧 NNLS：

```text
a_t = argmin_{a >= 0} ||B_t a - P_clamped[:, t]||_2^2
```

把所有帧拼起来，就得到：

- `A = slot_strength in R_{>=0}^{k x T}`

并可渲染出：

```text
P_ridge[:, t] = sum_i A[i, t] * B[:, C[i, t]]
```

当前默认 `clip_upper=True`，因此实际渲染时还会做：

```text
P_ridge = min(P_ridge, P_clamped)
```

这样可避免模板叠加把局部能量抬得高于观测谱。因为这里是在功率域拟合，所以这种“模板相加”仍然保留比较合理的物理语义。

### 4.9 回投到训练域

核心函数：

- `compress_whisper_power(...)`

拟合得到 `ridge_power` 后，为了回到原始训练域，需要重新压缩：

```text
ridge_compressed = compress_whisper_power(
    ridge_power,
    reference_max_log10=original_max_log10,
)
```

这里必须复用原样本的 `reference_max_log10`，不能让 `ridge_power` 自己重新取最大值，否则：

- 压缩尺度会漂移
- `ridge_compressed` 就不再和原始 `compressed` 处于同一参考系

代码里还会同时计算：

- `residual_power = power_clamped - ridge_power`
- `residual_compressed`
- `recon_compressed = compress(ridge_power + residual_power)`

由于 `ridge_power` 在默认设置下被裁到不超过 `power_clamped`，理论上：

```text
ridge_power + residual_power = power_clamped
```

因此 `recon_compressed` 应当和输入 `compressed` 基本一致，只剩数值误差。

### 4.10 为什么还能做可微渲染

如果把离散中心 `center_idx` 固定，那么：

```text
(center_idx, slot_strength, basis) -> ridge_power -> ridge_compressed
```

这一段对 `slot_strength` 是可微的。当前实现里专门保留了：

- `render_from_slots_torch(...)`
- `compress_whisper_power_torch(...)`

并在 `run_stage1_pipeline(...)` 里做了一个简单的 autograd smoke test，返回：

- `grad_norm`

它不是训练损失，只是证明“固定中心、只对连续强度反传”这条路在工程上是通的。

需要注意：

- `peak_mask`
- `top-k`
- `center_idx` 提取

这些离散步骤本身都不可微。因此这套一阶段更适合作为：

- 固定前端
- 半参数化中间表示
- 或“先离散检测，后连续渲染”的两阶段系统

而不是一个端到端全可微的离散中心学习器。

## 5. 这套封装里的张量分别是什么意思

常用中间量如下：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `power` | `(F, T)` | 原始 Whisper 风格 power spectrogram |
| `compressed` | `(F, T)` | 上游训练域里的压缩 STFT |
| `power_clamped` | `(F, T)` | 从 `compressed` 反解得到的截断后 power |
| `shifted` | `(F, T)` | `compressed` 平移到非负域后的谱 |
| `peak_mask` | `(F, T)` | 候选峰值布尔 mask |
| `ridge_support` | `(F, T)` | 稀疏 ridge support 浮点图 |
| `center_idx` | `(k, T)` | 每帧最多 `k` 个离散中心 bin |
| `valid_mask` | `(k, T)` | `center_idx` 中哪些槽位有效 |
| `basis` | `(F, F)` | 每个中心 bin 对应一列加粗模板 |
| `slot_strength` | `(k, T)` | 每个槽位的非负强度 |
| `coeff_map` | `(F, T)` | 把 `slot_strength` 散回 bin 轴的稀疏系数图 |
| `ridge_power` | `(F, T)` | 模板化 ridge 的 power 重建 |
| `ridge_compressed` | `(F, T)` | `ridge_power` 压回训练域后的谱 |
| `ridge_audio` | `(N,)` | 对 `ridge_power` 做 Griffin-Lim 得到的波形 |

其中最重要的参数化其实是：

```text
(center_idx, slot_strength, basis) -> ridge_power
```

如果把 `basis` 视为固定先验，那么真正需要保存的核心信息就是：

- `center_idx`
- `slot_strength`

## 6. 为什么这个目录算 self-contained

核心文件 `stage1_stft.py` 已经内置了原先分散在多个模块里的关键逻辑：

- 音频读写
- numpy / torch 转换辅助
- 峰值检测
- ridge 优化
- 中心索引提取
- 模板库构造
- NNLS 强度拟合
- 可微渲染
- Griffin-Lim 重建

因此把这个目录整体复制到别的项目里，不需要继续依赖本仓库其他 Python 模块。

## 7. 目录内容

- `stage1_stft.py`
  - 自包含核心实现
  - 不依赖本仓库其他 Python 模块
  - 包含从音频到 ridge 参数、再到谱和音频重建的整条链路
- `verify_against_legacy.py`
  - 验证脚本
  - 会把当前 self-contained 实现和封装前的 legacy 路径逐项对齐
  - 产出 side-by-side HTML 证明页
- `__init__.py`
  - 便于作为一个小包直接导入

## 8. 依赖要求

基础依赖：

- `numpy`
- `scipy`
- `librosa`
- `soundfile`

优化后端二选一：

- `torch`
- `cvxpy`

其他依赖：

- `matplotlib` 仅验证脚本需要

默认情况下 `ridge_optimize(...)` 会优先用 `torch` 后端；如果没有安装 `torch`，则会退回 `cvxpy`。如果两者都没有，support 优化这一步无法运行。

## 9. 主要 API

### 9.1 直接跑整条一阶段

```python
from stage1_compressed_stft_selfcontained.stage1_stft import run_stage1_pipeline

result = run_stage1_pipeline(
    "input.wav",
    backend="torch",
    device="cpu",
)
```

返回字典里最常用的字段有：

- `compressed`
- `shifted`
- `peak_mask`
- `ridge_support`
- `center_idx`
- `valid_mask`
- `basis`
- `slot_strength`
- `coeff_map`
- `ridge_power`
- `ridge_compressed`
- `residual_power`
- `residual_compressed`
- `recon_compressed`
- `ridge_audio`
- `grad_norm`
- `config`

### 9.2 只复用底层函数

如果你不想从 wav 直接跑整条链，也可以拆开调用：

```python
from stage1_compressed_stft_selfcontained.stage1_stft import (
    compress_whisper_power,
    decompress_whisper_compressed_stft,
    shift_compressed_stft_for_support,
    peak_mask,
    ridge_optimize,
    extract_center_idx_matrix,
    build_thickened_stft_basis,
    fit_strength_matrix_nnls,
)
```

典型拆法是：

1. 自己准备 `compressed`
2. 用 `shift_compressed_stft_for_support(...)` 和 `ridge_optimize(...)` 得到 `center_idx`
3. 用 `decompress_whisper_compressed_stft(...)` 得到 `power_clamped`
4. 在 power 域上做模板拟合和渲染

### 9.3 保存产物

```python
from stage1_compressed_stft_selfcontained.stage1_stft import save_stage1_artifacts

save_stage1_artifacts("out_dir", result)
```

会保存：

- `compressed_stft.npy`
- `shifted_stft.npy`
- `power.npy`
- `power_clamped.npy`
- `ridge_support.npy`
- `center_idx.npy`
- `valid_mask.npy`
- `slot_strength.npy`
- `coeff_map.npy`
- `basis.npy`
- `ridge_power.npy`
- `ridge_compressed.npy`
- `residual_power.npy`
- `residual_compressed.npy`
- `recon_compressed.npy`
- `ridge_griffinlim.wav`

## 10. 关键参数怎么理解

`run_stage1_pipeline(...)` 里最值得关心的参数有这些：

- `n_fft`, `hop_length`, `win_length`, `center`
  - STFT 参数，默认与上游 `vq06-stft` 定义对齐
- `peak_distance`, `peak_prominence`, `peak_height`, `peak_dilate`
  - 控制候选峰值的稀疏程度和局部宽容度
- `k`
  - 每帧最多保留多少条 ridge 中心
- `lam_sparse`
  - ridge support 的稀疏强度
- `lam_tv`
  - 时间方向平滑强度
- `ridge_refine`
  - 是否在 top-k 截断后再做一次 support 回归
- `ridge_stft_sigma`
  - STFT 模板的频率轴加粗宽度
- `backend`, `device`, `solver`
  - 优化后端和运行设备
- `griffin_lim_iters`, `griffin_lim_momentum`
  - 听感重建时的 Griffin-Lim 参数
- `compute_grad_norm`
  - 是否做可微渲染的 autograd smoke test

经验上：

- `k` 越大，表示能力越强，但参数也越多
- `lam_sparse` 越大，support 越瘦
- `lam_tv` 越大，时间上越平滑，也越容易抹掉短瞬态
- `ridge_stft_sigma` 越大，渲染出的 ridge 越宽

## 11. Griffin-Lim 重建在这里的意义

核心函数：

- `reconstruct_griffinlim(...)`

这一步不是算法主体，而是调试和可听化工具。它做的是：

```text
ridge_power
  -> magnitude = sqrt(ridge_power)
  -> Griffin-Lim phase reconstruction
  -> audio
```

需要强调：

- 这里恢复的是“与 `ridge_power` 相匹配的一段相位近似音频”
- 它不能证明相位建模正确
- 它主要用于听这组 ridge 参数大概保留了多少可感知结构

因此 `ridge_griffinlim.wav` 更适合拿来做主观检查，而不是当成高保真重建结果。

## 12. 验证方式

验证脚本会同时跑两条路径：

1. legacy 路径
   - 调用封装前散落在仓库里的原始实现
2. self-contained 路径
   - 调用这个目录里的 `stage1_stft.py`

然后逐项比较：

- `compressed`
- `shifted`
- `ridge_support`
- `basis`
- `slot_strength`
- `coeff_map`
- `ridge_power`
- `ridge_compressed`
- `ridge_audio`
- `center_idx`

并输出 side-by-side HTML，展示：

- legacy ridge 图
- self-contained ridge 图
- 绝对误差图
- legacy / self-contained 音频播放器
- support 图和误差统计

### 运行示例

在仓库根目录执行：

```bash
python stage1_compressed_stft_selfcontained/verify_against_legacy.py \
  --inputs input.wav examples/denoise_prompt.wav \
  --output_dir html/stage1_compressed_stft_selfcontained_verify \
  --backend torch \
  --device cpu
```

验证完成后打开：

`html/stage1_compressed_stft_selfcontained_verify/index.html`

## 13. 适合怎么复用

如果你要把这套方法移植到另一个项目，最稳的方式通常是：

1. 先把这个目录整体复制过去
2. 用 `run_stage1_pipeline(...)` 跑通一条样例，确认输入输出语义一致
3. 再按你的项目需要，拆成三层

比较推荐的拆分方式是：

1. support 选择层
   - 输入 `compressed`
   - 输出 `center_idx`
2. power 域渲染层
   - 输入 `center_idx` 和 `slot_strength`
   - 输出 `ridge_power`
3. 训练域回投层
   - 输入 `ridge_power` 和原始 `reference_max_log10`
   - 输出 `ridge_compressed`

这样最容易保证和当前这份经过验证的实现保持一致。

## 14. 这套方法的边界

这套一阶段方法很适合描述：

- 稳定的窄带 ridge
- 比较清晰的谐波线
- 频率轴上可由固定宽度模板近似的结构

但它并不直接建模：

- 相位
- 宽带噪声纹理
- 强瞬态的复杂时频扩散
- ridge 中心数量本身的可微变化

所以更准确地说，它是一个“针对主导窄带结构的半参数化前端”，不是一整套无损音频建模器。

## 15. 总结

这份 self-contained 封装保留了当前压缩 STFT 一阶段方法的核心语义：

- 输入域对齐 `vq06-stft`
- support 在压缩域上选择
- 强度在 power 域里拟合
- 输出可以重新压回训练域
- 在固定离散中心时保留可微渲染路径

如果你需要的是一份“既能直接复用、又把算法背景说完整”的实现，这个目录就是当前最稳的入口。
