# Ridge + Gaussian Mel 重建 Demo

## 概述

这个项目实现了以下端到端流程：

1. WAV -> STFT（缓存 GT 相位）
2. STFT 幅度 -> Mel 幅度
3. 频率轴峰值检测 -> 峰值掩码
4. 凸优化提取 ridge（L2 + L1 + 时间 TV，硬约束在峰值上）
5. Gaussian balls 拟合残差（凸优化，时间中心等间隔）
6. 反投影回线性频率幅度 + GT 相位 ISTFT
7. 生成 HTML 报告对比（GT / Ridge Only / R Direct / Reconstruction）

## 数学公式（核心）

STFT 与相位：

$$
S = \mathrm{STFT}(x), \quad A = |S|, \quad P = \angle S
$$

Mel 映射：

$$
M = F_{mel} A
$$

Ridge 优化（凸）：

$$
\min_{R} \ \frac{1}{2}\|R - M\|_F^2 + \lambda_{s} \|R\|_1 + \lambda_{tv} \sum_{i,t} |R_{i,t+1}-R_{i,t}|
$$

约束：

$$
R \ge 0, \quad R_{i,t} = 0 \ \text{if} \ V_{i,t}=0, \quad R \le M
$$

每帧 Top-k：对每个时间帧保留最大 k 个 ridge，其余置零。

Gaussian 残差拟合：

$$
E = \max(M - R, 0)
$$

$$
B_j(f,t) = \exp\left(-\frac{1}{2}\left(\frac{(f-f_c)^2}{\sigma_f^2} + \frac{(t-t_c)^2}{\sigma_t^2}\right)\right)
$$

$$
\min_{c \ge 0} \ \frac{1}{2} \|B c - \mathrm{vec}(E)\|_2^2 + \lambda_g \|c\|_1
$$

$$
G = \mathrm{reshape}(B c), \quad M_{hat} = \max(R + G, 0)
$$

逆 Mel 与重建：

$$
A_{hat} \approx F_{mel}^+ M_{hat} \quad \text{(pinv)}
$$

$$
S_{hat} = A_{hat} \cdot e^{jP}, \quad x_{hat} = \mathrm{ISTFT}(S_{hat})
$$

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py \
  --input input.wav \
  --output_dir out \
  --k 5 \
  --lam_sparse 0.05 \
  --lam_tv 0.1 \
  --sigma_t 1.0 \
  --sigma_f_list 1,2,4,8,16 \
  --lam_g 0.01 \
  --mel_inverse pinv \
  --save_intermediates True
```

## 输出

默认写入 `out/`：

- `gt.wav`, `recon.wav`, `ridge.wav`
- `gt_mel.png`, `recon_mel.png`, `ridge_mel.png`, `ridge_direct.png`
- `report.html`
- 以及 `.npy` 中间结果（若 `--save_intermediates True`）

## 参数说明（所有 flag）

### 输入与基础
- `--input`：输入 WAV 路径（必填）
- `--output_dir`：输出目录（必填）
- `--sr`：重采样率，默认 `None`（保持原采样率）

### STFT
- `--n_fft`：FFT 点数，默认 `2048`
- `--hop_length`：帧移，默认 `256`
- `--win_length`：窗长，默认 `2048`
- `--center`：是否中心化（True/False），默认 `True`

### Mel
- `--n_mels`：Mel 维度数，默认 `128`
- `--fmin`：最低频率，默认 `0`
- `--fmax`：最高频率，默认 `sr/2`

### 峰值检测
- `--peak_distance`：峰值最小间距（mel bins），默认 `2`
- `--peak_prominence`：峰值突出度阈值，默认 `None`
- `--peak_height`：峰值高度阈值，默认 `None`
- `--peak_dilate`：是否膨胀峰值掩码（0/1），默认 `0`

### Ridge 优化
- `--k`：每帧保留 ridge 数量上限，默认 `5`
- `--lam_sparse`：L1 稀疏正则系数，默认 `0.05`
- `--lam_tv`：时间 TV 正则系数，默认 `0.1`
- `--solver`：CVXPY 求解器（SCS/ECOS/OSQP），默认 `SCS`
- `--ridge_refine`：是否做 top-k 后再优化（True/False），默认 `True`

### Gaussian Balls
- `--spacing_frames`：时间中心间隔（帧），默认自适应 `T/32`
- `--sigma_t`：时间方向 sigma（帧），默认 `1.0`（小，避免横向变宽）
- `--sigma_f_list`：频率方向 sigma 列表（逗号分隔），默认 `1,2,4,8,16`
- `--lam_g`：Gaussian 系数 L1 正则，默认 `0.01`

### 逆 Mel
- `--mel_inverse`：`pinv`（快速）或 `nnls`（更精确但慢），默认 `pinv`

### 其他
- `--seed`：随机种子，默认 `0`
- `--save_intermediates`：是否保存中间结果（True/False），默认 `True`

## 报告页面顺序

`report.html` 的列顺序：
1. Ground Truth
2. Ridge Only
3. R (Direct)
4. Reconstruction
