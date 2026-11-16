# TAPOR Reproduction Audit

审计日期：2026-11-17。审计范围为只读分析；未修改 `external/TAPOR/`、checkpoint、原始日志或数据集，也没有启动训练。本报告以官方仓库 commit `09b8e7cf76744a6973cb4249f76332005be5891e` 为代码基线。

## Executive Summary

**目前不能宣称已严格复现论文 Figure 12(c) 的 TAPOR 结果。** 已确认复现了官方仓库当前 `inference.py` 所定义的 207 文件、35,200 帧评估流程，并能够用保存的原始预测和标签可重复地计算指标；但该流程的预训练模型全局结果是 **26.0315 mm MPJPE / 85.8502% PCK@4cm**，不等于论文的 **22.636 mm / 90.41%**。论文声称的 34,520 个测试样本也不能由本 commit 的逐文件 `batch_size=64, drop_last=True` 输出得到，因为 34,520 不是 64 的倍数，而实际结果为 35,200。

重训 checkpoint 在**完全相同的 35,200 标签**上为 **31.0186 mm / 74.1246%**，比预训练 checkpoint 差 **4.9871 mm / 11.7256 个百分点**。这是真实性能差异，不是把逐文件指标误当全局指标、文件排序或标签不一致所致。

结论：**现在不应立即再训练一次 200 Epoch。** 已有一次完整 200 Epoch 训练；先解决“论文测试协议 / 数据版本 / 作者 checkpoint 的对应关系”这一不可比问题。只有在确认相同的训练数据版本、相同评估清单及官方 checkpoint 架构后，才应做一次受控重训。仅重新评估即可完成的工作列在最后一节。

## 证据与方法

新增的两份独立 Testset 结果已只读解压在 `analysis/inference_results/`：

| 模型 | 文件 | 成功文件 | 失败文件 | 帧数 |
|---|---|---:|---:|---:|
| 作者预训练 `tapor.pth` | `tapor_type1_weights_tapor.pth_2026111695944_result_dict.pkl` | 207 | 0 | 35,200 |
| 本次重训 Epoch 194 | `tapor_type1_weights_1_20261116103244_tapor.pth_2026111616649_result_dict.pkl` | 207 | 0 | 35,200 |

`analysis/audit_inference_results.py` 对两份 `result_dict.pkl` 只读计算。实现严格对应 `external/TAPOR/utils.py:103-111` 的 MPJPE 和 `:113-129` 的 PCK：每个 `(frame, joint)` 先计算三维欧氏距离，再对所有 21 个关节、全部帧求平均/比例。坐标保留为米，表中距离转换为 mm；PCK 阈值是 0.04 m。它还按 `utils.py:81-100` 计算 root drift，仅把 root-aligned 数值作为诊断，绝不混入官方 MPJPE/PCK。

分析输出的可追溯机器可读证据是 `analysis/audit_evidence.json`。运行命令：

```bash
python3 analysis/audit_inference_results.py \
  --pretrained analysis/inference_results/tapor_type1_weights_tapor.pth_2026111695944_result_dict.pkl \
  --retrained analysis/inference_results/tapor_type1_weights_1_20261116103244_tapor.pth_2026111616649_result_dict.pkl \
  --official-logs analysis/official_logs \
  --output analysis/audit_evidence.json
```

## 四方结果比较

详表见 `analysis/TAPOR_RESULTS_COMPARISON.csv`。论文数值来自任务中给出的 Figure 12(c)；其余数值均从本地工件或代码直接取得。

| 来源 | 测试协议 | 样本 | MPJPE (mm) | PCK@4cm (%) | 可严格比较？ |
|---|---|---:|---:|---:|---|
| 论文 Figure 12(c) | 论文称 34,520 testing samples；细节未随本地结果提供 | 34,520 | 22.636 | 90.41 | 否 |
| 作者 `Logs.zip`，TAPOR 77 的 102 文件结果 | archive `result_dict.pkl` | 25,472 | 23.5936 | 89.7785 | 否 |
| 作者预训练 checkpoint，本次独立推理 | 当前 `inference.py` 协议 | 35,200 | 26.0315 | 85.8502 | 与下一行可比 |
| 本次重训 checkpoint，本次独立推理 | 同上 | 35,200 | 31.0186 | 74.1246 | 是 |
| 本次重训的训练集内部随机 test split | `traintest.py`，不是独立 Testset | 日志未保存精确帧数 | 26.9242 | 85.2858 | 否 |

### 实际聚合口径比较

逐帧平均和对全部 frame-joint pair 的全局平均在这里数学上相同：每帧固定有 21 个关节。逐关节平均再对 21 个关节平均也相同。只有给每个文件相同权重时不同，因为结果文件的长度并不相同。

| 模型 | 全局 / 逐帧 / 逐关节 MPJPE (mm) | 全局 / 逐帧 / 逐关节 PCK@4cm (%) | 等权逐文件 MPJPE (mm) | 等权逐文件 PCK (%) | Root drift (mm) |
|---|---:|---:|---:|---:|---:|
| 预训练 | 26.0315 | 85.8502 | 31.8609 | 78.5232 | 25.6584 |
| 重训 | 31.0186 | 74.1246 | 31.3072 | 73.8116 | 26.2996 |

把预训练模型改成“等权逐文件”会得到 **31.8609 mm**，离论文更远；root alignment 的诊断 MPJPE 是 20.6943 mm，虽更低但不是官方 `calculate_mpjpe` 调用的口径。因此没有任何合理且已证实的聚合切换能把当前预训练结果合法地变成 22.636 mm / 90.41%。

## 35,200 与 34,520

已证实：

1. 当前 `inference.py:263-314` 硬编码 207 个独立 Testset 文件；结果中的文件名顺序与该清单匹配，且两模型相同。
2. `inference.py:255` 固定 batch size 64，`:418-421` 对每一个文件单独构建 `iHand_dataset` 和 `DataLoader(..., drop_last=True)`。保存结果中的帧分布为 110 x 256、90 x 64、6 x 192、1 x 128，共 35,200；每个文件输出均是 64 的倍数。
3. `dataset.py:22-37` 先丢弃每个源文件的前 20 帧；`dataset.py:302-307` 在 `fo=1` 时仅保留 2D 与 3D 标签均非空的帧；`dataset.py:358-363` 定义 fragment 长度。当前推理使用 `fs=1`，因此不因 fragment 合并帧，但仍会受上述前两项及 `drop_last` 影响。
4. 34,520 除以 64 的余数是 24。故它不可能是**当前代码中每个文件均 `drop_last=True` 后保存的完整输出帧数**。

结论（高置信度）：差异不是应当“删掉 680 帧”来修正的问题。它至少意味着以下之一：论文统计的是 DataLoader 前的样本数、论文使用了不同的文件清单/数据版本/过滤规则，或论文文字/图表计数存在不一致。由于本地没有原始 Testset pkl 的逐帧记录，无法区分前三种可能性；不能从 35,200 人为选择 34,520 帧。

可在 GPU 服务器上执行的只读核验（不训练）：

```bash
cd /workspace/TAPOR
python /path/to/thermal-hand-depth/analysis/audit_inference_results.py --help
# 对每个官方清单文件，在 dataset.py 过滤和 DataLoader drop_last 前后记录长度；不要改文件。
```

上述命令仅说明可用分析器；要精确定位 680 帧，还需要一个只读逐文件计数脚本在服务器原始 Testset 上运行，或作者提供论文版本的原始文件清单。当前 Mac 上没有 Testset pkl，不能假装完成该检查。

## 作者 Experimental Results（`Logs.zip`）

`analysis/official_logs/Logs/` 含 13 个 pickle、一个 `.DS_Store` 和 macOS 元数据；不含 checkpoint、TensorBoard event、训练文本日志或单独的汇总 CSV。6 个 `LogInference/` 文件分别对应 baseline3d、MANO、MediaPipe 及 3 个 TAPOR 结果；另外有 ambient、hand-T、outside、temperature、tracking 等场景结果。两个 gesture pickle 不是 pose `result_dict` 架构。

所有可解析 pose `result_dict` 都只有 `infernece_file_name`、`failed_file_names`、`predicts`、`labels`。它们允许重算指标，但没有直接记录作者训练的 best epoch、训练 loss 或 validation loss。TAPOR 相关已重算的全局结果包括：

| 归档子集 | 成功文件 / 帧 | MPJPE (mm) | PCK@4cm (%) |
|---|---:|---:|---:|
| `LogInference/..._76_...` | 100 / 24,960 | 72.7960 | 18.0538 |
| `LogInference/..._77_..._2024311193111` | 143 / 28,096 | 25.3519 | 86.0814 |
| `LogInference/..._77_..._202431124637` | 102 / 25,472 | 23.5936 | 89.7785 |
| ambient interference | 18 / 4,544 | 22.5488 | 88.5752 |

**没有一个本地 `Logs.zip` 的原始 result_dict 同时重算为 22.636 mm 与 90.41%，也没有任何一个有 34,520 帧。** 因而它不含足以从原始预测/标签直接复现 Figure 12(c) 的完整数据。最接近的 102 文件 TAPOR archive 仍不应被擅自认定为 Figure 12(c) 的来源。

## 训练协议核查

| 项目 | 证据 | 状态 |
|---|---|---|
| 训练命令 | README `:40` 与 `metadata/manifest.json` 一致：`-m tapor -e 200 -b 48 -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1` | 已证实 |
| 模型 | `traintest.py:581-616` 选择 base1；`tapor_model_config.py:2-41`；`fs=1` 在 `:603` 将 `num_history` 设为 0 | 已证实 |
| 优化器 / scheduler | `traintest.py:620-639`：默认 Adam，`lr=1e-4`，`s=0` 时无 scheduler | 已证实 |
| 损失 | `traintest.py:292-305`：`jb` 为 joint MSE 加 bone-length loss | 已证实 |
| 划分 | `traintest.py:648-657`：Trainset 合并后随机 80/10/10，而不是按文件固定 train/val；DataLoader 三组都 `drop_last=True` | 已证实 |
| 实际总样本 | 服务器 console 保存为 `The size of the dataset is: 33197`；因此 int(0.8N)=26,557，int(0.1N)=3,319，余数 test=3,321 | 已证实 |
| 内部 test 评分数量 | batch 48 + `drop_last=True`，3,321 理论输出为 3,312；日志只留下数值、未记录 shape | 有代码支持，未由保存数组直接确认 |
| best checkpoint | `traintest.py:324-339` 按 validation **coordinate MAE** 最小选择；TensorBoard 最小 validation MAE 为 Epoch 194 的 0.0100431，匹配保存日志 | 已证实 |
| 训练曲线 | TensorBoard 有完整 200 个 epoch；validation MSE 最小 Epoch 154，validation MAE 最小 Epoch 194，末 epoch MAE 0.0130139 | 已证实 |
| checkpoint 结构 | 本次 checkpoint 是 476 tensors、868,099 parameters 的 raw state dict。预训练 `.pth` 本体不在本地，无法做 state_dict 逐键比对 | 部分证实 |

注意论文的 26,557 / 3,319 与本次运行的 33,197 总 Trainset 样本精确吻合代码切分；它们不是独立 Testset 34,520 的代码路径。训练集与测试集有 11 个同名文件，但 `dataset_inspection_report.json` 的 `verified_train_test_content_overlap` 为空、哈希错误为空，故**没有内容重叠证据，也不能仅凭同名宣称泄漏**。

## 指标定义与可比性

- 单位：标签数值和模型输出按米处理；训练日志的 `0.026924232` 因此是 26.924232 mm。高置信度，来自 `utils.py` 中 0.04 PCK 阈值与数据检查中约 0.1 m 的关节坐标范围。
- MPJPE / PCK：三维欧氏距离，非按坐标 MAE；包括全部 21 个关节且包括 root joint；没有无效标签 mask；未 root-align。
- root drift / MPJRPE：`calculate_shift_error` 先算 joint 0 的欧氏漂移，再仅为“shift error”将预测根对齐。推理打印但未将此对齐应用于 MPJPE / PCK。
- 当前独立推理：`inference.py:432-471` 每个文件打印一次原始 MPJPE/PCK，`:478-493` 仅保存参数日志和 `result_dict`，不输出全局汇总。因此此前的 22.19 / 22.17 只能是最后文件的指标，不能解释为全局结果。
- 对每个模型，本审计确认所有 35,200 个标签有限、没有全零 3D 标签帧。跨模型文件名及每个标签数组都完全相同。

## 性能差异的证据边界

### 26.03（当前预训练）对 22.64（论文）

| 结论 | 证据级别 | 置信度 |
|---|---|---|
| 当前预训练结果不是论文 Figure 12(c) 的严格复现 | 直接数据与代码 | 高 |
| 不是“最后一个文件的 22.xx mm”造成的全局误读 | 代码直接证实 | 高 |
| 不是将全局改为等权文件平均即可解决 | 直接重算 | 高 |
| 论文与当前运行使用的 sample protocol 或数据版本至少有一项不同，或论文计数/报告不一致 | 强证据但不能区分原因 | 高 |
| 具体是数据版本、文件清单、标签修订还是图表误差 | 当前不可判断 | 低 |

### 31.02（重训）对 26.03（预训练）

| 结论 | 证据级别 | 置信度 |
|---|---|---|
| 重训模型在同标签同文件协议上确实较差 | 直接重算 | 高 |
| 差异不是按文件聚合、样本数、标签或推理文件顺序造成 | 直接数据比对 | 高 |
| 预训练与重训 checkpoint 权重本身不同 | 由预测不同及训练事实支持 | 高 |
| 随机初始化、CUDA 非确定性、训练数据版本、预处理环境可能贡献差异 | 源码与环境证据支持，但未隔离实验 | 中 / 低 |
| 存在训练/测试内容泄漏 | 现有哈希审计反证同名文件泄漏；其他泄漏目前无证据 | 低 |

`traintest.py:525-529` 固定 Torch / CUDA / NumPy seed，但未设置 Python `random.seed`、cudnn deterministic 或 DataLoader worker generator；这构成可重复性风险。不过 fs=1 时 `dataset.py:367-370` 的 `random.randint(0, 0)` 恒为 0，不能据此断言它是本次 4.99 mm 差异的根因。

## 后续行动（按优先级）

1. **只读复核当前 35,200 帧结果。** 运行本审计脚本；成本约数秒，无需重训。目的：任何后续实验都固定报告 pooled 口径及 result_dict 哈希。
2. **在服务器做逐文件计数审计。** 读取原始 207 个 Testset pkl，记录“原始帧、跳过前 20、2D+3D 过滤后、drop_last 后”四个数；成本分钟级，无需重训。此步骤是解释 680 帧差异的必要条件。
3. **向作者索取 Figure 12(c) 对应的 checkpoint、207/论文版清单、原始 result_dict 和评价脚本 commit。** 无计算成本，无需重训。这是确认 22.636 来源的唯一直接证据。
4. **加载并逐键比较预训练与重训 `.pth`。** 需要把作者 `tapor.pth` 复制到本地或服务器；成本秒级，无需重训。检查 key、shape、参数总数、base1/fs1 设置，不比较权重数值来判断“好坏”。
5. **受控重训（仅在 1--4 完成后）。** 固定数据 archive hash、官方 commit、conda/PyTorch、命令、seed 与确定性设置；200 Epoch 约 18,708 秒（本次约 5.2 小时），需要重训。若第 2--4 步发现协议或数据不匹配，先修正协议；否则新的 200 Epoch 也不会回答论文 22.636 的可比性问题。

## 未执行事项

没有在 Mac 或服务器重新训练、重新推理、下载数据、安装 CUDA/依赖、修改模型、改写官方源码或删除任何数据。也没有因 35,200 与 34,520 的差异删除或筛选任何帧。
