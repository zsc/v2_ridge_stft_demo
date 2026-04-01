# 压缩 STFT 一阶段 Self-contained 封装

这个目录把“压缩 Whisper-STFT 上的一阶段 ridge 参数化”收成了一套可以单独拎走的实现。

## 目录内容

- `stage1_stft.py`
  - 核心实现
  - 不依赖本仓库其他 Python 模块
  - 内含音频读写、峰值检测、ridge support 优化、模板拟合、Griffin-Lim 重建
- `verify_against_legacy.py`
  - 验证脚本
  - 会把这个 self-contained 实现和封装前的 legacy 路径逐项对齐
  - 产出 side-by-side HTML 证明页

## 这套封装保留了什么算法语义

目标定义仍然对齐 `/Users/zhousc6/Downloads/wav2vq0206/VQ06_STFT_PAIR_DATASET_20260401.md` 第 `3.2` 节：

- `n_fft = 400`
- `hop = 160`
- `window = hann`
- `power = |STFT(x)|^2`
- 去掉最后一帧
- `compressed = (max(log10(clamp(power, 1e-10)), max_log10 - 8.0) + 4.0) / 4.0`

一阶段仍然分成两段：

1. 在压缩域上选 support
   - `shifted = compressed - min(compressed)`
   - `peak_mask`
   - `ridge_optimize`
   - 得到每帧 top-k 的离散中心 `center_idx`
2. 在 `power_clamped` 域上拟合强度
   - `power_clamped = 10 ** (4 * compressed - 4)`
   - 用 STFT bin 轴上的固定宽度模板库 `basis`
   - 用 NNLS 拟合连续强度 `slot_strength`

最后如果要回到训练域：

- 把 `ridge_power` 用同一个 `reference_max_log10` 压回 `ridge_compressed`

如果要听结果：

- 对 `ridge_power` 开平方得到 magnitude
- Griffin-Lim 重建成 `ridge_griffinlim.wav`

## 为什么这个目录算 self-contained

核心文件 `stage1_stft.py` 已经内置了原先分散在多个模块里的东西：

- `audio.py`
  - `load_audio`
  - `save_audio`
- `peaks.py`
  - `peak_mask`
- `backend.py`
  - torch / device / numpy 转换辅助
- `ridge_opt.py`
  - ridge support 优化
- `stage1_parametric/*.py`
  - center index、strength、basis、渲染与压缩 STFT 相关逻辑

所以把这个目录复制到别的项目里，只需要满足第三方依赖：

- `numpy`
- `scipy`
- `librosa`
- `soundfile`
- `matplotlib` 仅验证脚本需要
- `torch` 可选，但当前默认会优先用它做 ridge 优化

## 主要 API

### 直接跑整条一阶段

```python
from stage1_compressed_stft_selfcontained.stage1_stft import run_stage1_pipeline

result = run_stage1_pipeline(
    "input.wav",
    backend="torch",
    device="cpu",
)
```

返回结果里包括：

- `compressed`
- `ridge_support`
- `center_idx`
- `slot_strength`
- `ridge_power`
- `ridge_compressed`
- `ridge_audio`

### 保存产物

```python
from stage1_compressed_stft_selfcontained.stage1_stft import save_stage1_artifacts

save_stage1_artifacts("out_dir", result)
```

会保存：

- `.npy` 中间数组
- `ridge_griffinlim.wav`

## 验证方式

验证脚本会做两条路径：

1. legacy 路径
   - 调用封装前散落在仓库里的实现
2. self-contained 路径
   - 调用这个目录里的 `stage1_stft.py`

然后比较：

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

并输出 side-by-side HTML：

- legacy ridge 图
- self-contained ridge 图
- 绝对误差图
- legacy / self-contained 音频播放器

## 运行示例

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

## 适合怎么复用

如果你要把这套方法移植到另一个可微渲染项目，最稳的方式是：

1. 先把这个目录整体复制过去
2. 用 `run_stage1_pipeline(...)` 跑通一条样例
3. 再按你的项目需要，把 `run_stage1_pipeline(...)` 拆成：
   - support 选择层
   - power 域模板渲染层
   - 压缩域回投层

这样最容易保证和现在这份经过验证的结果一致。
