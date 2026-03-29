# 一阶段 Ridge 参数化封装

这个目录把当前工程里“一阶段 Ridge 加粗拟合”的连续部分拆成了一个可复用、可自包含的参数化表示。

## 结论先说

可以，但需要把“一阶段参数”分成两部分看：

- 一个离散的中心索引矩阵 `C in Z^{k x T}`
- 一个连续的强度矩阵 `A in R_{>=0}^{k x T}`

其中：

- 横轴 `T` 是时间帧
- 纵轴 `k` 是每帧最多保留的 ridge 槽位数
- `C[i, t]` 表示第 `t` 帧第 `i` 个槽位对应的 mel bin 中心
- `A[i, t]` 表示这个槽位的非负强度

当前工程的一阶段并不是“只靠一个 `k x T` 浮点矩阵”就能完整表达，因为每一帧的 top-k 中心位置也在变。所以严格地说，完整参数化是：

`(C, A) -> R`

如果把 `C` 视为固定条件，那么从 `A` 到 mel 谱 `R` 的映射是可微的。

## 和当前实现的对应关系

当前工程里一阶段实际分成三步：

1. `ridge_optimize(...)` 先选出每帧的稀疏细线中心 `R_raw`
2. `R_raw > 0` 给出离散中心集合
3. `fit_topk_thickened_mel_bands(...)` 对这些中心生成“线性频谱固定宽度、映射回 mel 后非均匀加粗”的模板，再逐帧做 NNLS 拟合

这个目录做的事情，是把第 3 步改写成显式的矩阵形式：

- `C[k, T]`: 中心索引矩阵
- `A[k, T]`: 强度矩阵
- `B[n_mels, n_mels]`: 每个 mel 中心对应的一列加粗模板

于是：

`R[:, t] = sum_i A[i, t] * B[:, C[i, t]]`

如果 `C[i, t] = -1`，表示该槽位为空，忽略即可。

## 为什么说它可微

在 `C` 固定时，上式对 `A` 是线性的。

- `render_mel_from_slots_torch(...)` 用的就是纯张量 gather + multiply + sum
- 对 `A` 可反传
- 如果把 `B` 也作为张量参数，它对 `B` 也可反传

注意边界：

- `ridge_optimize` 里的 top-k 支持选择是离散的，不可微
- `fit_strength_matrix_nnls` 里的 NNLS 拟合本身也不是端到端可微训练图的一部分
- legacy 路径里的 `clip_upper=True` 本质是 `min(render, M)`，它是分段线性的，几乎处处可导，但会在饱和区截断梯度

如果你在另一个可微渲染项目里要做真正端到端训练，建议：

- 固定 `C`，只优化 `A`
- 或者把离散中心 `C` 换成 soft assignment / 连续频率参数
- 训练时通常关掉 `clip_upper`，把“不超过观测谱”的约束放到 loss 里

## 文件说明

- `parametric_stage1.py`
  - 自包含核心实现
  - 不依赖本仓库其他模块
- `verify_parametric_stage1.py`
  - 验证脚本
  - 会调用旧版一阶段流程做对照
  - 产出 side-by-side HTML

## 核心 API

### 1. 提取离散中心矩阵

```python
center_idx, valid_mask = extract_center_idx_matrix(ridge_raw, k=3, sort_by="mel_asc")
```

输出：

- `center_idx.shape == (k, T)`
- 空槽位填 `-1`

### 2. 构造加粗模板库

```python
basis = build_thickened_mel_basis(F_mel, sigma_bins=2.0, normalize="peak")
```

输出：

- `basis.shape == (n_mels, n_mels)`
- 第 `j` 列表示“以 mel bin `j` 为中心的加粗模板”

### 3. 拟合强度矩阵

```python
strength, rendered = fit_strength_matrix_nnls(M, center_idx, basis, clip_upper=True)
```

输出：

- `strength.shape == (k, T)`
- `rendered.shape == (n_mels, T)`

### 4. 在可微图里直接渲染

```python
import torch

strength_t = torch.tensor(strength, dtype=torch.float32, requires_grad=True)
render_t = render_mel_from_slots_torch(center_idx, strength_t, basis, clip_upper=None)
loss = render_t.square().mean()
loss.backward()
```

## 验证方式

验证脚本会对每个输入文件执行：

1. 用旧版流程生成 legacy 一阶段结果
2. 用本目录实现重新构造 `C`、`A` 和 `R`
3. 比较：
   - `legacy_ridge` vs `parametric_ridge`
   - legacy 系数图 vs `slot_strength -> coeff_map`
4. 运行一个 torch autograd smoke test
5. 输出 HTML 页面

页面里会展示：

- GT mel
- 旧版一阶段结果
- 新参数化渲染结果
- 绝对误差
- 中心索引矩阵
- 强度矩阵

## 运行示例

在仓库根目录执行：

```bash
python stage1_parametric/verify_parametric_stage1.py \
  --inputs input.wav examples/denoise_prompt.wav \
  --output_dir html/stage1_parametric_verify \
  --backend torch \
  --device cpu
```

运行结束后打开：

`html/stage1_parametric_verify/index.html`

## 集成建议

如果你在另一个 mel 可微渲染项目里复用，推荐拆成两层：

1. 预处理层
   - 负责产生固定的 `center_idx`
   - 可以来自 ridge 检测、启发式规则、外部标注，或者另一个网络
2. 可微渲染层
   - 输入 `center_idx` 和 `strength`
   - 输出 `R`
   - 训练时只对 `strength` 或模板参数反传

这样和当前工程的一阶段语义最接近，也最稳。 
