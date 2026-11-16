# Recommended GitHub Figure Selection

Use these seven figures in the main README. They tell a complete, cautious reproduction story without overloading readers.

1. `figures/mpjpe_comparison_overall.png` — headline result and gap.
2. `figures/pck4cm_comparison_overall.png` — complementary pose-accuracy measure.
3. `figures/shared_subset_performance.png` — fairest direct author/current/reproduction comparison.
4. `figures/pck_curve_official_vs_ours.png` — performance across thresholds.
5. `figures/evaluated_frame_count_comparison.png` — immediate protocol caveat.
6. `figures/testset_overlap_summary.png` — explains the historical/current dataset difference.
7. `figures/category_mpjpe_official_vs_ours.png` — condition-level diagnostic.

Place the remaining figures in a deeper `analysis/` section or link to [GITHUB_FIGURE_SUMMARY.md](GITHUB_FIGURE_SUMMARY.md):

- Training curves: `training_loss_curve.png`, `validation_mse_curve.png`, `validation_mae_curve.png`.
- Shared-subset PCK and category/frame-count diagnostics: `shared_subset_pck4cm.png`, `testset_category_frame_counts.png`, `category_pck4cm_official_vs_ours.png`.
- Error distributions: `joint_error_comparison.png`, `per_user_mpjpe.png`, both framewise histograms, and `per_file_mpjpe_boxplot.png`.

Do not present the paper, historical, and current all-set values as a strict leaderboard: their evaluated frame counts differ. Pair the overall MPJPE/PCK figures with the frame-count and shared-subset figures.
