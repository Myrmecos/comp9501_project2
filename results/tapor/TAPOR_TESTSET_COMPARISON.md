# TAPOR 测试集逐文件对照

## 通俗结论（500 字内）

这三套“测试集”不是同一份名单。A 是你用当前官方 `inference.py` 跑出的 207 文件 / 35,200 帧；B 是 `Logs.zip` 中一个最接近论文数值的历史 TAPOR 记录，102 文件 / 25,472 帧；论文只写了 34,520 帧，没有给文件名单。A 与 B 有 100 个相同文件，且其中每一帧的标签都完全相同，共 24,960 帧。A 多了 107 个文件 / 10,240 帧，主要是环境温度、环境干扰和手温实验；B 多了 2 个 U11 文件 / 512 帧。故 A 比 B 多 9,728 帧正好等于 10,240 减 512，不是同一文件被不同截断。相同帧上两份作者预测的 MPJPE 都是 22.5966 mm，说明 23.5936 与 26.0315 的差异几乎完全由选了哪些文件造成。论文的 34,520 帧目前无法从本地代码、日志或结果精确恢复，不能把 B 当成论文原始测试集。

## 范围、工件与方法

本报告只读分析以下工件，没有运行 GPU inference、训练或改动官方源码：

- A（当前独立推理）：
  - `analysis/inference_results/tapor_type1_weights_tapor.pth_2026111695944_result_dict.pkl`（公开预训练）
  - `analysis/inference_results/tapor_type1_weights_1_20261116103244_tapor.pth_2026111616649_result_dict.pkl`（本次重训）
- B（本报告明确选定的历史记录，不等同论文测试集）：
  - `analysis/official_logs/Logs/LogInference/tapor_type1_weights_77_20242421040_tapor.pth_202431124637_result_dict.pkl`
  - 文件名中记录的模型为 `tapor_type1_weights_77_20242421040_tapor.pth`，记录时间后缀为 `202431124637`。
- 作者 archive 根目录：`analysis/official_logs/`，由本地 `Logs.zip` 只读解压而来。

分析器 `analysis/compare_testsets.py` 对每个 `result_dict.pkl` 的 `infernece_file_name`、`predicts` 和 `labels` 进行逐文件读取。MPJPE/PCK 沿用 `external/TAPOR/utils.py:103-129`：对 21 个关节各自做 XYZ 欧氏距离，再汇总；PCK@4cm 使用 `<0.04` 米。

对同名文件，本审计先比较完整标签数组；若不同，才以每帧 21x3 标签的 SHA-256 精确匹配帧，而不按位置假定相同。结果：100 个共同文件的标签数组均逐元素相同、帧数均相同，因此其 24,960 帧可严格比较。

机器可读完整结果：`analysis/testset_overlap_analysis.json`。每一个文件的名称、类别、两边帧数及三个模型的指标在 `analysis/testset_file_comparison.csv`；按类别、前缀、用户、左右手的加权汇总在 `analysis/testset_category_summary.csv`。

## 三套测试数据究竟是什么

| 名称 | 可直接观察的组成 | 文件 / 实际评估帧 | 是否有逐文件清单 | 能否等同论文 Figure 12(c) |
|---|---|---:|---|---|
| A：当前独立 inference | 当前 `inference.py` 硬编码的 207 个文件；两份 A 结果标签完全一致 | 207 / 35,200 | 有，来自 result_dict | 否，帧数不等于 34,520 |
| B：历史 LogInference 77（本报告对照对象） | 102 个 result_dict 条目，来自作者旧实验结果 | 102 / 25,472 | 有，来自 result_dict | 无法确认 |
| 论文 Figure 12(c) | 论文说 6 名 office 用户的一部分训练、其余测试；总 testing=34,520 | 未给 / 34,520（论文声称） | 没有 | 无法恢复 |

论文 v2 §5.1 明确写“six users in the office”用于训练，剩余数据用于测试，并给出 26,557 / 3,319 / 34,520；它没有列出 pkl 文件名。官方当前 `traintest.py:648-657` 却是把 `Trainset/` 合并后 `random_split` 为 80/10/10；当前 `inference.py:263-314` 另有 207 文件的固定列表。这两个代码路径都没有编码论文所述“六名 office 用户 / 其余测试”的精确文件映射。

论文指标是 TAPOR 2.2636 cm（22.636 mm）和 PCK@4cm=0.9041；论文网页的 §5.1、Figure 12(c) 与 §5.2 可以核对这些数字：[arXiv HTML](https://arxiv.org/html/2501.17585v2)。本地未发现 `TAPOR.pdf`，因此此处核对的是论文公开 HTML v2。

## Logs.zip 中所有可识别 TAPOR 实验

以下是 archive 中文件名含 `tapor` 且可按 pose `result_dict` 解析的全部记录。两份 gesture pickle 不是 pose result_dict，故不纳入。

| 子目录 / 记录 | 成功 / 失败文件 | 帧 | MPJPE (mm) | PCK@4cm (%) | 与论文关系 |
|---|---:|---:|---:|---:|---|
| `Dataset_Ambient_interference_logs/...2024620195042...` | 18 / 1 | 4,544 | 22.5488 | 88.5752 | 场景鲁棒性，不是完整测试 |
| `LogInference/..._76_...202421912457...` | 100 / 0 | 24,960 | 72.7960 | 18.0538 | 旧 / 差模型 |
| `LogInference/..._77_...2024311193111...` | 143 / 0 | 28,096 | 25.3519 | 86.0814 | 历史集合，非 34,520 |
| `LogInference/..._77_...202431124637...`（B） | 102 / 0 | 25,472 | 23.5936 | 89.7785 | 最接近论文，但不能确认 |
| `hand_T_logs/...2024316161834...` | 21 / 13 | 1,344 | 35.4206 | 83.5176 | 手温鲁棒性 |
| `outside_logs/...2024316155544...` | 15 / 9 | 960 | 36.1995 | 69.7371 | 额外条件实验 |
| `temperature_impact_logs/...202439211828...` | 72 / 6 | 4,608 | 138.2762 | 32.8497 | 环境温度鲁棒性 |
| `tracking_logs/...20243819643...` | 44 / 0 | 11,136 | 43.7025 | 79.2034 | tracking 实验 |

没有一个 archive 记录有 34,520 帧，也没有一个同时重算为 22.636 mm / 90.41%。因此 archive 不能恢复 Figure 12(c) 的原始测试清单。

## A 与 B 的逐文件集合关系

| 集合 | 文件数 | 实际评估帧 | 含义 |
|---|---:|---:|---|
| A ∩ B | 100 | 24,960 | 同名、同帧数、标签逐元素相同 |
| A - B | 107 | 10,240 | A 独有；不是 B 的后续帧 |
| B - A | 2 | 512 | B 独有；不是 A 的漏记帧 |
| A 总计 | 207 | 35,200 | 24,960 + 10,240 |
| B 总计 | 102 | 25,472 | 24,960 + 512 |

所以 B **不完全包含于 A**（它有两个 A 不含的 U11 文件）；A 也不包含于 B。共同文件中“同名但帧数不同”的文件数为 **0**。帧差精确分解为 `35,200 - 25,472 = 10,240（A 独有）- 512（B 独有）= 9,728`。

### B 独有的精确文件

| 文件 | 帧 | B 历史 MPJPE (mm) | B PCK@4cm (%) |
|---|---:|---:|---:|
| `P1_U11_R_X_X_0.pkl` | 256 | 89.5437 | 63.6719 |
| `P1_U11_R_X_X_1.pkl` | 256 | 54.8573 | 43.0246 |

它们合计 72.2005 mm / 53.3482%，会把 B 从共同集的 22.5966 mm 拉到全 B 的 23.5936 mm。论文说十名参与者 U1-U10，而该历史记录出现 U11；这是 B 不能被直接视为论文测试集的又一个直接证据。

### A 独有的精确名单

107 个名称、逐文件帧数与指标全部逐行列在 `testset_file_comparison.csv` 中（筛选 `membership=A_only`）。按实际命名类别的精确分组为：

- 覆盖：41 文件 / 2,624 帧，如 `P1_U1_L_thick1_0.pkl`、`P1_U1_cover_medicalGlove_0.pkl`、`P1_UX_cover_nail_0.pkl`。
- 环境温度：33 文件 / 2,112 帧，如 `P2_U1_R_X_X_Temp_17_0.pkl`、`P2_U2_R_X_X_Temp_36_3.pkl`。
- 环境干扰：18 文件 / 4,544 帧，如 `P1_body_R_X_X_0.pkl`、`P1_laptop_R_X_X_0.pkl`、`P1_hotbottle_R_X_X_0.pkl`。
- 手温：11 文件 / 704 帧，如 `Px_U1_R_hot_X_2.pkl`、`Px_U1_R_Warm_X_1.pkl`、`Px_U1_R_vcold_X_4.pkl`。
- 额外光照：4 文件 / 256 帧，如 `Px_U1_R_LightConds7_X_1.pkl`、`Px_U2_R_LightCond6_X_2.pkl`。

完整具体名单不在此处缩写或抽样，避免把示例误当完整清单；CSV 的每一行就是一个可检查文件，含 `a_frames`、`b_frames`、`a_mpjpe_mm`、`a_retrained_mpjpe_mm` 和 `b_mpjpe_mm`。

## 共同测试帧上的公平比较

下表只使用 100 个共同文件、24,960 个标签逐元素匹配的帧。三列之间可严格比较。

| 模型输出来源 | MPJPE (mm) | PCK@4cm (%) |
|---|---:|---:|
| A 中当前公开预训练模型 | 22.5966 | 90.5266 |
| A 中本次重训模型 | 30.7832 | 74.1369 |
| B 中作者历史模型 | 22.5966 | 90.5258 |

历史模型与当前公开预训练模型在共同帧上只相差 **0.000076 mm**、**0.00076 个百分点**；这可能是数值运行环境差异，也可能权重实际相同，但本地没有历史 `tapor.pth` 本体，不能以此证明 state_dict 相同。

样本选择的量化结论：

- B 全部 25,472 帧是 23.5936 mm，其中 B 独有 U11 的 512 帧为 72.2005 mm。
- A 的共同 24,960 帧是 22.5966 mm；加入 A 独有 10,240 帧后变为 26.0315 mm。
- 因而历史 23.5936 与当前 26.0315 的 2.4379 mm 差距，在现有两份 result_dict 中可由文件选择解释；不能归因于在共同数据上模型变差。

## 条件、用户、左右手汇总

以下类别是从真实文件名机械分类，并以 notebook 作交叉验证；并非额外假设。完整按 `category`、`prefix`、`participant`、`hand` 的表在 `testset_category_summary.csv`。

| 文件名类别 | A 文件 / 帧 | A 预训练 MPJPE | A 重训 MPJPE | B 文件 / 帧 | B 历史 MPJPE | A 独有 / B 独有文件 |
|---|---:|---:|---:|---:|---:|---:|
| `standard_x_x` | 52 / 12,672 | 25.8971 | 33.7849 | 54 / 13,184 | 27.6951 | 0 / 2 |
| covering | 68 / 9,536 | 26.2255 | 33.9071 | 27 / 6,912 | 20.0782 | 41 / 0 |
| light condition | 20 / 4,352 | 19.1802 | 18.1628 | 16 / 4,096 | 18.6475 | 4 / 0 |
| distance | 5 / 1,280 | 16.1602 | 16.2923 | 5 / 1,280 | 16.1595 | 0 / 0 |
| hand temperature | 11 / 704 | 22.4663 | 38.6921 | 0 / 0 | — | 11 / 0 |
| ambient temperature | 33 / 2,112 | 54.7440 | 32.6893 | 0 / 0 | — | 33 / 0 |
| ambient interference | 18 / 4,544 | 22.5488 | 31.7375 | 0 / 0 | — | 18 / 0 |

这里的 `standard_x_x` 只是“文件名含 `_X_X_`、且未匹配到其他明确条件词”的审计标签，**不是官方定义的‘普通手势’**。若以它作为最接近“普通条件”的代理，重训为 33.7849 mm，公开权重为 25.8971 mm，差 7.8878 mm；这说明重训差距并非仅由 A 新增的特殊实验拉高。

同时，重训并不是在所有特殊条件均较差：它在 A 独有环境温度文件为 32.6893 mm，优于公开权重的 54.7440 mm；在 A 独有覆盖文件为 28.4673 mm，优于公开权重的 42.4218 mm。但它在手温（38.6921 对 22.4663）、环境干扰（31.7375 对 22.5488）及共同集（30.7832 对 22.5966）明显更差。因此“重训 31.02 主要由特殊条件导致”是**不成立**的；证据指向共同的 `_X_X_` / 覆盖数据及手温数据都有明显退化，且各条件的训练原因仍需受控实验确认。

逐用户与手别也不能直接当作泛化结论，因为 A/B 的条件组成不同。例如 A 的 U1 是 14,656 帧 / 26.8281 mm，B 的 U1 是 10,880 帧 / 21.9013 mm；这首先反映文件选择不同。完整的 U1-U11/UX、L/R、P1-Px 汇总见 CSV，不能只比较这些总体均值下结论。

## 文件名可验证含义与不能确认的含义

| 名称部分 | 可证实解释 | 证据 / 限制 |
|---|---|---|
| `U1` ... `U10` | 用户 ID | `z_results_visualization.ipynb` cell 16 从第二 token 提取 `U*`；论文 §5.1 说十名参与者 U1-U10。`UX` 与 U11 未在论文定义。 |
| `L` / `R` | 左 / 右手标识 | `dataset.py:20,33-36` 注释并编码左 `[1,0]`、右 `[0,1]`。实现仅检查文件名是否含 `L`，很脆弱，但表达了命名意图。 |
| `P1`、`P2` 等 | 环境分组前缀 | notebook cell 18 将 P1/P2/P3/P5/P6 的文件组画成 `E1`--`E5`；论文说有 5 环境。没有代码把 P1 明确映射到 office/bathroom/bedroom/kitchen/meeting room，因此不能写成确定地点。 |
| `Cover` / `cover` / `thick` / glove / nail | 手部覆盖物实验 | notebook cell 25 明确把这些 token 映射为 flour、foam、cool water、warm water、oil、ring、nail、plastic glove、medical glove、thin/thick glove；论文 Figure 21 是 hand coverings。 |
| `LightCond` | 光照条件 | notebook cell 22 按 `LightCond0`--`3` 分组；论文 Figure 20 是 light conditions。 |
| `Dist` | 手-传感器距离条件 | notebook cell 20 用这些数据作距离曲线；论文 §5.3讨论 hand-sensor distance。 |
| `Temp_17` 等 | 环境温度条件 | notebook cell 29 是 `temperature_impact_logs`；论文 §5.3 讨论 ambient temperature。 |
| `hot/Warm/normal/scold/vcold` | 手温条件 | notebook cell 27 为 hand temperature；论文说明热/温/冷/冰水实验。 |
| `body/laptop/display/hotbottle/light` | 环境热干扰条件 | notebook cell 31；论文 §5.3 明确列 laptop、display、hot water bottle、body、light tubes。 |
| `empty` | notebook 的 “w/o interference” 组 | **不能据此断言“没有手”**：当前 inference 使用 `fo=1`，而这些 result_dict 仍有 3D pose 标签。更安全的解释是该 notebook 把它作为无外界热干扰对照。 |
| `X_X` | 未有正式定义 | 仅是频繁出现的命名模式；无法从代码/论文确认“普通条件”。 |

## 34,520 的来源审计

需要区分四层数量：

1. **原始测试帧数**：每个 Testset pkl 中 thermal map 的总帧。当前 Mac 没有原始 Testset pkl，不能给出论文版本总数。
2. **有效标注帧数**：`dataset.py:22-37` 先跳过每文件开头 20 帧；`:302-307` 在 `fo=1` 时保留 2D/3D 标签均非空的帧。
3. **送入模型的帧数**：`inference.py:418-421` 每文件建 dataset，再以 batch 64 加载。
4. **指标帧数**：同一行 DataLoader 设置 `drop_last=True`，每文件剩余不足 64 的 batch 被丢弃；保存的 result_dict 只含这些帧。

当前 A 的 35,200 是第 4 层。它的 207 文件帧分布为 110 x 256、90 x 64、6 x 192、1 x 128，全是 64 的倍数。34,520 不是 64 的倍数，故不可能是本 commit 当前逐文件 `drop_last=True` 所保存的完整推理输出。B 的 25,472 也是 64 的倍数。所有本地 Logs 记录的帧数也不等于 34,520。

结论：**论文 34,520 的精确名单目前无法恢复。** 所需的最小额外证据是论文版本的 Testset pkl archive（或内容 hash）、论文版逐文件清单、每文件“跳过 20 / 标签过滤 / drop_last”前后的计数，以及 Figure 12(c) 所用 checkpoint 和 result_dict。没有这些证据，不应从 A/B 选择、也不应从 35,200 中删除 680 帧来凑数。

## 复现性影响

本报告不改变先前审计结论：目前可以声称“按当前 `inference.py` 207 文件协议复现并评估”，不能声称“严格复现论文 Figure 12(c)”。对当前重训模型，最有价值的下一步不是立即再跑一次 200 Epoch，而是固定 A 共同集和 A 独有集分别报告，并取得论文版文件清单后再决定是否需要按论文协议受控重训。
