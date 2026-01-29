# SPEC: WAV → STFT(保留相位) → Mel → 凸优化抽取脊线(ridges) + 高斯球(gaussian balls) → 反投影回STFT幅度 → 用GT相位ISTFT恢复WAV → 生成HTML对比

## 0. 目标与范围

**目标**：给定一个输入 WAV（视为 GT / ground truth），构建一个 Python 程序完成如下端到端流程：

1. 对 WAV 做 **STFT**，保留并缓存 **GT 相位**。
2. 将 **STFT 幅度谱**映射到 **Mel 幅度谱**（保留 Mel 维度的能量表征）。
3. 在 Mel 幅度谱上做 **沿频率轴的局部峰值检测**，得到“垂直峰值掩码”(vertical peak mask)。
4. 在峰值候选位置上，通过 **凸优化**抽取 **脊线 ridge**：
   - **硬约束**：ridge 只能出现在峰值候选位置上
   - **正则**：稀疏 (L1) + 水平 TV（沿时间轴的 Total Variation）
   - **每帧保留 k 条 ridges（默认 k=5）**
5. 第二阶段：在时间轴上等间隔“种植”**高斯球 gaussian balls**，并优化其系数以最小化对 GT Mel 的 **L2 误差**：
   - 高斯球鼓励**竖向增长（频率方向）**
   - 且**不要过宽（时间方向不要扩散太大）**
6. 将 ridge + gaussian 在 Mel 域的结果 **反投影回线性频率 STFT 幅度**（近似逆 Mel）。
7. 使用第 1 步缓存的 **GT 相位**，与重建幅度组合得到复数 STFT，再 **ISTFT**恢复出重建 WAV。
8. 输出一个 **HTML** 页面，左右并排对比：
   - 两个 WAV 的 Mel 谱图（GT vs 重建）
   - 网页内可直接播放两段音频（`<audio controls>`）

**非目标**（可不做）：
- 不要求 Griffin-Lim 之类相位重建（明确使用 GT phase）。
- 不要求 ridge/gaussian 的物理意义完美，只需流程正确、优化可运行、输出可视化与音频可播放。
- 不强制完全“严格凸”地实现 top-k（可采用“凸优化 + per-frame top-k 投影 + 可选再优化”两阶段近似）。

---

## 1. 输入与输出

### 1.1 输入
- `input_wav`：任意单声道或立体声 WAV 文件（程序需支持自动转单声道）。
- 可选参数：STFT、Mel、优化超参、输出目录等（见 CLI 章节）。

### 1.2 输出（写入 `output_dir/`）
必须产物：
- `gt.wav`：原始音频（拷贝或重采样后保存，便于网页引用）
- `recon.wav`：重建音频
- `report.html`：左右对比页面（含 Mel 谱图与音频播放器）
- `gt_mel.png`、`recon_mel.png`：两张 Mel 谱图图片（供 HTML 引用或内嵌）

建议产物（方便调试与复现）：
- `stft_phase.npy`：GT 相位（`angle(S)`）
- `stft_mag.npy`：GT 幅度（`abs(S)`）
- `mel_mag.npy`：GT Mel 幅度（`M`）
- `peak_mask.npy`：峰值掩码（`V`）
- `ridge.npy`：ridge 结果（`R`）
- `gaussian.npy`：gaussian 贡献（`G`）
- `mel_hat.npy`：重建 Mel（`M_hat`）
- `stft_mag_hat.npy`：重建线性频率幅度（`A_hat`）

---

## 2. 算法管线（必须按顺序实现）

### 2.1 读 WAV & 预处理
- 使用 `soundfile` 或 `librosa.load` 读取：
  - 保留原采样率 `sr`（默认不强制重采样；可提供 `--sr` 参数用于重采样）
- 若多通道：转单声道（取平均）。
- 可选：归一化（例如 peak normalize 到 [-1,1]），但要保持 `gt.wav` 与重建对齐的参考一致性。

### 2.2 STFT（保留相位）
- 参数默认（可 CLI 覆盖）：
  - `n_fft=2048`
  - `hop_length=256`
  - `win_length=2048`
  - `window='hann'`
  - `center=True`
- 计算复数 STFT：`S = stft(x)`
- 缓存：
  - `A = abs(S)`，shape `[n_freq, n_frames]`
  - `P = angle(S)`，shape `[n_freq, n_frames]`  **(GT phase)**

> 注意：后续重建必须用 `P`，即 `S_hat = A_hat * exp(1j*P)`。

### 2.3 幅度映射到 Mel
- 构建 Mel 滤波器矩阵 `F_mel`：
  - shape `[n_mels, n_freq]`
  - 参数默认：
    - `n_mels=128`
    - `fmin=0`
    - `fmax=sr/2`
- Mel 幅度（线性幅度，不是 dB）：
  - `M = F_mel @ A`
  - shape `[n_mels, n_frames]`

---

## 3. 脊线抽取：峰值候选 + 凸优化（核心）

### 3.1 沿频率轴局部峰值检测 → 垂直峰值掩码 V
对每一帧 `t`，对列向量 `M[:, t]` 做局部峰值检测（频率轴/竖直轴）：

- 输出 `V`：bool mask，shape `[n_mels, n_frames]`
- 峰值定义（建议实现之一）：
  - 用 `scipy.signal.find_peaks` 对 `M[:, t]`：
    - `distance=peak_distance`（默认 2 或 3）
    - 可选 `prominence` / `height`（默认不强制，但建议提供参数）
- 将检测到的 peaks index 置 True，其它 False
- 可选：对 `V` 做轻微形态学膨胀（例如 ±1 mel bin）以提高可行性（提供开关参数）

### 3.2 Ridge 凸优化：稀疏 + 水平 TV + 硬约束(只在峰值上)
定义优化变量：
- `R ∈ R^{n_mels×T}`，代表 ridge 的 Mel 强度（非负连续值）

硬约束（必须）：
- `R >= 0`
- `R[i,t] = 0` 当 `V[i,t] == False`（硬约束：ridge 只能出现在峰值候选位置上）
  - 在实现上可写为：`R <= V * big_number` 或直接对变量做 masked 约束
- 建议额外约束（可选，但推荐）：
  - `R <= M`（ridge 不超过原始能量）

目标函数（必须包含）：
- 数据项：`0.5 * ||R - M||_F^2`
- 稀疏项：`λ_sparse * ||R||_1`
- 水平 TV（时间方向）：`λ_tv * Σ_i Σ_t |R[i,t+1] - R[i,t]|`

优化形式（CVXPY）：
- `minimize( 0.5*sum_squares(R - M) + lam_sparse*norm1(R) + lam_tv*tv_time(R) )`
- subject to constraints above

求解器：
- 使用 `cvxpy`，优先尝试：
  - `OSQP`（若可用且建模为 QP/近似 TV）
  - 否则 `SCS` / `ECOS`（TV + L1 通常 SCS 比较稳）
- 必须提供 `--solver {scs,ecos,osqp}` 参数，默认 `scs`

### 3.3 “每帧保留 k 条 ridges”（默认 k=5）
严格的“每帧最多 k 个非零”是非凸的。允许使用以下 **可运行的近似方案（推荐实现）**：

**方案 A（推荐）：凸优化 → per-frame top-k 投影 → 可选再优化**
1. 先解上面的凸优化得到 `R_raw`
2. 对每帧 `t`：
   - 找到 `R_raw[:,t]` 中值最大的 `k` 个 mel bin（且必须在 `V[:,t]` 为 True 的位置）
   - 其余位置置零，得到 `R_topk`
3. 可选 refinement（默认开启，提供开关 `--ridge_refine`）：
   - 用 `R_topk` 的 support 当作新的硬掩码 `V2`（只允许这些 top-k 位置）
   - 再跑一次同样凸优化（或仅最小二乘 + TV），得到更平滑/一致的 `R`

输出 ridge：
- `R = R_topk` 或 refine 后结果

---

## 4. Gaussian balls 阶段：等间隔时间种植 + 优化拟合 residual

### 4.1 目标与 residual
- residual：`E = clip(M - R, min=0)`
- 目的：用一组高斯“球”在 Mel 时频平面上拟合 `E`，得到 `G`，使：
  - `M_hat = R + G` 更接近 GT `M`

### 4.2 等间隔时间中心（必须）
- 设 `T = n_frames`
- 选择 `n_centers` 或 `spacing_frames`：
  - 默认：`spacing_frames = max(1, round(T / 32))`（即大约 32 个中心；或直接默认 `n_centers=32`）
- 时间中心集合：
  - `t_c = [0, spacing, 2*spacing, ...]`（截断到 `<T`）
- 每个中心固定时间位置（**不优化 t_c**）

### 4.3 频率中心初始化（推荐）
每个时间中心 `t_c` 需要一个或多个频率中心 `f_c`（在 mel bin 上）：
- 推荐从 ridge 中取：在 `t=t_c` 的 `R[:,t]` 里取非零位置的 mel index（最多 k 个）
- 若该帧 ridge 为空，则回退到 `M[:,t]` 的 peaks（来自 `V`）或直接取 `argmax(M[:,t])`

### 4.4 高斯基函数设计：鼓励竖向增长、避免横向变宽（必须体现）
采用**字典/基函数**方式，保证整体优化是凸的（系数 NNLS + L1）：

对每个 (time center `t_c`, freq center `f_c`) 生成若干 2D 高斯基函数：
- 时间方向 sigma：`sigma_t` **固定为较小值**（默认 `sigma_t=1.0` 帧 或 2.0 帧）
  - 这样天然“不会太宽”（横向扩散受控）
- 频率方向 sigma：提供一组候选 `sigma_f_list`（默认如 `[1, 2, 4, 8, 16]` mel bins）
  - 允许更大的 `sigma_f`，体现“鼓励竖向增长”
- 2D 基函数（在 mel×time 网格上）：
  - `B_j(f,t) = exp(-0.5*((f-f_c)^2/sigma_f^2 + (t-t_c)^2/sigma_t^2))`
- 将所有基函数堆叠成字典矩阵 `B`：
  - `B` shape `[n_mels*T, n_basis]`（展平时频）

### 4.5 Gaussian 优化（凸）：最小化 Mel L2 loss + 稀疏（必须）
变量：
- `c >= 0`，shape `[n_basis]`，每个基函数一个非负系数

目标：
- `min 0.5 * ||B c - vec(E)||_2^2 + λ_g * ||c||_1`

约束：
- `c >= 0`

输出：
- `G = reshape(B c, [n_mels, T])`
- `M_hat = clip(R + G, min=0)`

可选约束/正则（若实现不复杂可加）：
- 对更“宽”的 `sigma_t` 基函数施加更大 L1 权重（但本 spec 默认 sigma_t 固定，因此不需要）
- 对 `G` 加时间 TV（弱约束）以减少闪烁（可选）

---

## 5. Mel → 线性频率 STFT 幅度（反投影/近似逆 Mel）

### 5.1 基本反投影（必须）
- 已知 `M_hat = F_mel @ A_hat`，要求 `A_hat` 近似解
- 推荐实现两种模式（CLI 可选）：

**模式 1：伪逆（快）**
- `F_pinv = pinv(F_mel)`，shape `[n_freq, n_mels]`
- `A_hat = F_pinv @ M_hat`
- `A_hat = clip(A_hat, min=0)`

**模式 2：逐帧 NNLS（更准但慢，推荐可选）**
对每个时间帧 `t` 解：
- `min ||F_mel a_t - m_hat_t||_2^2  s.t. a_t >= 0`
- 用 `scipy.optimize.nnls` 或 cvxpy 小规模 QP
- 拼回 `A_hat[:,t]`

默认：伪逆（`--mel_inverse pinv`），可选 `--mel_inverse nnls`

---

## 6. 用 GT 相位恢复 WAV（必须）

- 复数 STFT 重建：
  - `S_hat = A_hat * exp(1j * P)`
- ISTFT：
  - `x_hat = istft(S_hat, hop_length, win_length, window, center)`
- 保存 `recon.wav`（采样率同 `gt.wav`）
- 建议输出前做轻微裁剪避免溢出：
  - `x_hat = clip(x_hat, -1, 1)`

---

## 7. HTML 报告（必须：并排谱图 + 可播放音频）

### 7.1 Mel 谱图渲染
- 分别对 `gt.wav` 与 `recon.wav` 计算 Mel 幅度谱（可复用同一 Mel 参数）
- 转 dB 仅用于可视化：
  - `M_db = 20*log10(M + eps)`
- 用 `matplotlib` 画图并保存：
  - `gt_mel.png`
  - `recon_mel.png`

### 7.2 report.html
生成 `report.html`，要求：
- 左右两列布局：
  - 左：GT（图片 + `<audio controls src="gt.wav">`）
  - 右：Reconstruction（图片 + `<audio controls src="recon.wav">`）
- HTML 中引用相对路径资源（同目录可直接打开）
- 可选：增加一行显示参数摘要（n_fft, hop, n_mels, k, λ 等）

---

## 8. CLI 设计（必须）

提供一个入口脚本，例如：
- `python main.py --input input.wav --output_dir out/ [options...]`

必须参数：
- `--input`：输入 wav 路径
- `--output_dir`：输出目录

推荐参数（给默认值）：
### STFT
- `--n_fft 2048`
- `--hop_length 256`
- `--win_length 2048`
- `--center True/False`

### Mel
- `--n_mels 128`
- `--fmin 0`
- `--fmax` 默认 `sr/2`

### Peak detection
- `--peak_distance 2`
- `--peak_prominence` 可选，默认 None
- `--peak_height` 可选，默认 None
- `--peak_dilate 0/1`（0=不膨胀，1=±1 mel bin 膨胀）

### Ridge optimization
- `--k 5`
- `--lam_sparse 0.05`（示例默认）
- `--lam_tv 0.1`（示例默认）
- `--solver scs`
- `--ridge_refine True/False`

### Gaussian stage
- `--spacing_frames`（默认按 T 自适应）
- `--sigma_t 1.0`
- `--sigma_f_list 1,2,4,8,16`
- `--lam_g 0.01`

### Inverse mel
- `--mel_inverse pinv|nnls`

### Misc
- `--seed 0`
- `--save_intermediates True/False`

---

## 9. 项目结构（推荐）

```

project/
main.py
audio.py              # load/save wav
stft_utils.py         # stft/istft wrapper
mel_utils.py          # mel basis + forward + inverse
peaks.py              # peak mask
ridge_opt.py          # cvxpy ridge extraction
gaussian_opt.py       # basis build + convex NNLS
report.py             # png + html
requirements.txt
README.md

```

---

## 10. 依赖（requirements.txt）

最低建议：
- numpy
- scipy
- librosa
- soundfile
- matplotlib
- cvxpy
- tqdm (可选)
- jinja2 (可选，用于 HTML 模板)

---

## 11. 关键实现细节与验收标准

### 11.1 数组维度约定（必须一致）
- `S`: `[n_freq, T]` complex
- `A`, `A_hat`: `[n_freq, T]` float
- `P`: `[n_freq, T]` float (radians)
- `F_mel`: `[n_mels, n_freq]`
- `M`, `R`, `G`, `M_hat`, `E`: `[n_mels, T]`
- `V`: `[n_mels, T]` bool

### 11.2 Ridge 的硬约束必须生效
- 对所有 `(i,t)` 若 `V[i,t]==False`，则输出 `R[i,t]==0`（允许浮点误差，但保存前可强制置零）

### 11.3 k ridges per frame
- 对每帧 `t`：`count_nonzero(R[:,t] > 0) <= k`
- 若 refine 模式，最终也必须满足该条件

### 11.4 Gaussian 阶段必须是“时间等间隔中心 + 横向不宽”
- `t_c` 必须等间隔生成（基于 `spacing_frames` 或 `n_centers`）
- `sigma_t` 默认固定小值，且代码逻辑中要明确体现“横向不宽”的约束/设计

### 11.5 端到端可运行与产物齐全
- 运行一次 CLI 能生成：
  - `gt.wav`, `recon.wav`, `gt_mel.png`, `recon_mel.png`, `report.html`
- `report.html` 用浏览器打开能：
  - 显示两张谱图
  - 播放两段音频

---

## 12. 建议的默认超参（可调整）
- `n_fft=2048, hop=256, win=2048`
- `n_mels=128`
- `peak_distance=2`
- `k=5`
- `lam_sparse=0.05`
- `lam_tv=0.1`
- `spacing_frames ~= T/32`
- `sigma_t=1.0`
- `sigma_f_list=[1,2,4,8,16]`
- `lam_g=0.01`
- `mel_inverse=pinv`

---

## 13. 示例命令（写入 README）
```

python main.py 
--input path/to/input.wav 
--output_dir out 
--k 5 
--lam_sparse 0.05 --lam_tv 0.1 
--sigma_t 1.0 --sigma_f_list 1,2,4,8,16 --lam_g 0.01 
--mel_inverse pinv 
--save_intermediates True

```

---

## 14. 备注：关于“凸优化”的边界说明（允许的实现方式）
- Ridge 主优化：必须用凸优化（cvxpy）实现 L2 + L1 + TV + 硬约束 mask。
- “每帧 top-k”可以采用：
  - 凸优化后投影（top-k）+ 可选再解一次凸优化（支持固定）
  - 不要求用 MILP 强行做严格 top-k（因为实现重且慢）
- Gaussian 阶段：采用“固定字典 + 非负系数 NNLS + L1”的凸问题，实现对 Mel L2 loss 的最小化，并通过固定小 `sigma_t` 达到“不太宽”。


