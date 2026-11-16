# TAPOR Result Figure Summary

All figures in this directory are derived read-only from local reproduction artifacts. They do not rerun inference or training. Main result comparisons use the independent Testset inference results; the `traintest.py` random internal split is not used in any headline performance chart.

## Figures

| File | What it shows | Exact data source(s) | Why useful | Placement |
|---|---|---|---|---|
| `figures/training_loss_curve.png` | Average training loss over 200 epochs. | `reproduction_results/reproduced_training/events.out.tfevents.*`, tag `train loss (average)`. | Documents that the recorded run executed the intended schedule. | Analysis |
| `figures/validation_mse_curve.png` | Validation MSE over 200 epochs. | Same event file, tag `validation mse loss (average)`. | Shows optimization progression; this is not independent Testset performance. | Analysis |
| `figures/validation_mae_curve.png` | Validation coordinate MAE over 200 epochs. | Same event file, tag `validation mae loss (average)`. | Explains the saved Epoch 194 selection criterion. | Analysis |
| `figures/pck_curve_official_vs_ours.png` | PCK from 0.5 cm to 9 cm for official versus reproduced checkpoint. | Both files in `inference_results/`; all 35,200 frames. | More informative than a single PCK threshold. | README |
| `figures/mpjpe_comparison_overall.png` | Paper, historical archive, current official inference and reproduction MPJPE. | Paper reference value; `audit_evidence.json`. | Establishes the reproduction gap while preserving protocol caveats. | README |
| `figures/pck4cm_comparison_overall.png` | The corresponding PCK@4cm comparison. | Paper reference value; `audit_evidence.json`. | Gives the complementary accuracy view. | README |
| `figures/evaluated_frame_count_comparison.png` | 34,520 paper-reported samples versus 25,472 historical and 35,200 current evaluated frames. | Paper reference value; `audit_evidence.json`. | Makes protocol non-equivalence visible. | README |
| `figures/shared_subset_performance.png` | MPJPE on the exact shared 24,960 labeled frames. | `testset_overlap_analysis.json`. | Fair three-model comparison unaffected by file-list selection. | README |
| `figures/shared_subset_pck4cm.png` | PCK@4cm on the exact shared subset. | `testset_overlap_analysis.json`. | Complements the shared-subset MPJPE. | Analysis |
| `figures/testset_overlap_summary.png` | Shared, A-only, and B-only frame counts. | `testset_overlap_analysis.json`. | Explains why historical and current all-set results differ. | README |
| `figures/testset_category_frame_counts.png` | A/B frame counts by audited filename category. | `testset_category_summary.csv`. | Reveals which condition groups are absent from the historical record. | Analysis |
| `figures/category_mpjpe_official_vs_ours.png` | Official/reproduced MPJPE by filename category. | `testset_category_summary.csv`, A independent inference. | Locates condition-dependent differences without inventing a new test protocol. | Analysis |
| `figures/category_pck4cm_official_vs_ours.png` | Official/reproduced PCK@4cm by filename category. | `testset_category_summary.csv`, A independent inference. | Shows thresholded accuracy by condition. | Analysis |
| `figures/joint_error_comparison.png` | Mean Euclidean error of all 21 joints. | Both current `result_dict.pkl` files; all 35,200 frames. | Shows whether error is joint-specific. | Analysis |
| `figures/per_user_mpjpe.png` | MPJPE grouped by filename user token. | `testset_category_summary.csv`, A independent inference. | Exploratory user-token view; condition composition differs by user. | Analysis |
| `figures/framewise_mpjpe_histogram_official.png` | Distribution of official model framewise MPJPE. | Official current `result_dict.pkl`; all 35,200 frames. | Shows dispersion hidden by a global mean. | Analysis |
| `figures/framewise_mpjpe_histogram_ours.png` | Distribution of reproduced model framewise MPJPE. | Reproduced current `result_dict.pkl`; all 35,200 frames. | Shows whether degradation is broad or tail-heavy. | Analysis |
| `figures/per_file_mpjpe_boxplot.png` | Distribution of per-file MPJPE for both current models. | Both current `result_dict.pkl` files; 207 files each. | Summarizes variation across recordings. | Analysis |

## Reading Notes

- “Paper” is a reported reference (22.636 mm, 90.41%, 34,520 samples), not recomputed from local predictions.
- “Author historical” is specifically `Logs/LogInference/tapor_type1_weights_77_20242421040_tapor.pth_202431124637_result_dict.pkl`, with 25,472 evaluated frames. It is not asserted to be the paper Figure 12(c) test protocol.
- “Official pretrained” and “Our reproduction” are independently evaluated on the same current 207-file / 35,200-frame result protocol.
- `standard_x_x` is a mechanical filename category used by the audit, not a documented claim that samples are normal or easy.

## Interpretation in Chinese

1. **现有证据不支持“结果较差主要因为没有正确使用 200 epochs”。** 保存的训练 manifest 使用了官方命令的 `-e 200 -b 48`，TensorBoard 也记录了 200 个 epoch；最终 checkpoint 按官方代码的 validation MAE 标准选择 Epoch 194。不能从现有数据把 31.02 mm 的差距归因于 epoch 数量。

2. **论文和官方 notebook 都把 200 epochs 作为训练超参数使用，而没有给出选择 200 的单独理论论证或消融理由。** 论文 §5.1 与 README/notebook 中的命令均采用 200 epochs；它是实验设置，不是已经证实的性能差异根因。

3. **更可能的差距来源是协议与训练复现细节，而非 epoch 数。** 审计已证实论文、历史 Logs 与当前 inference 的文件选择/帧数不同；当前官方预训练与历史模型在共同 24,960 帧上几乎相同，说明全局 23.59 对 26.03 的变化主要由测试集选择造成。重训模型在同一共同子集仍显著较差，因此后续应核对训练数据版本、精确训练协议、权重/配置兼容性和可重复性设置；不能仅靠再训练一次 200 epochs 解决或解释差距。

## Skipped Figures

None. All 18 requested figures were generated from available local artifacts.
