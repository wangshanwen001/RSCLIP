import numpy as np
from scipy import stats
import os

def perform_statistical_analysis(our_path, sota_path):

    if not os.path.exists(our_path) or not os.path.exists(sota_path):
        print(f"错误: 找不到文件 {our_path} 或 {sota_path}")
        print("请确保先运行 eval.py 导出数据。")
        return

    our_ious = np.load(our_path)
    sota_ious = np.load(sota_path)

    if len(our_ious) != len(sota_ious):
        print(f"警告: 样本量不一致! (我们的: {len(our_ious)}, SOTA: {len(sota_ious)})")
        min_len = min(len(our_ious), len(sota_ious))
        our_ious = our_ious[:min_len]
        sota_ious = sota_ious[:min_len]

    mean_our = np.mean(our_ious)
    mean_sota = np.mean(sota_ious)
    improvement = mean_our - mean_sota

    t_stat, p_val = stats.ttest_rel(our_ious, sota_ious)

    print("-" * 50)
    print(f"{'指标 (Metric)':<20} | {'数值 (Value)':<15}")
    print("-" * 50)
    print(f"{'Our mIoU':<20} | {mean_our:.4f}")
    print(f"{'SOTA mIoU':<20} | {mean_sota:.4f}")
    print(f"{'Improvement':<20} | {improvement:+.4f}")
    print(f"{'T-Statistic':<20} | {t_stat:.4f}")
    print(f"{'P-Value':<20} | {p_val:.6e}") # 使用科学计数法显示
    print("-" * 50)

    if p_val < 0.05:
        significance = "具有显著统计学意义 (Statistically Significant)"
        if p_val < 0.01:
            significance += " (Highly Significant, p < 0.01)"
    else:
        significance = "不具有统计学显著性 (Not Statistically Significant)"

if __name__ == "__main__":

    perform_statistical_analysis('rsclip_ious.npy', 'sota_per_image_ious.npy')