import os
import time
import torch
import cv2
import numpy as np
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.analysis import get_model_complexity_info

import time

import segmentor
import custom_datasets
from utils import append_experiment_result

def measure_fps(runner, input_shape=(1, 3, 448, 448), warm_up=10, iterations=100):

    model = runner.model
    model.eval()
    
    dummy_input = torch.randn(*input_shape).cuda()
    
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_input) 
            
    torch.cuda.synchronize() 
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = iterations / total_time
    return fps

def compute_boundary_f1_per_class(gt, pred, num_classes=7, dilation_ratio=0.02):
    def get_boundary(mask):
        mask = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        return mask - eroded

    class_f1_results = {}
    dist = int(max(gt.shape) * dilation_ratio)
    if dist < 1: dist = 1
    kernel = np.ones((dist, dist), np.uint8)

    for i in range(num_classes):
        gt_i = (gt == i).astype(np.uint8)
        pred_i = (pred == i).astype(np.uint8)

        if gt_i.sum() == 0:
            continue

        gt_b = get_boundary(gt_i)
        pred_b = get_boundary(pred_i)

        gt_b_d = cv2.dilate(gt_b, kernel)
        pred_b_d = cv2.dilate(pred_b, kernel)

        precision = np.sum(pred_b * gt_b_d) / (np.sum(pred_b) + 1e-6)
        recall = np.sum(gt_b * pred_b_d) / (np.sum(gt_b) + 1e-6)

        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        class_f1_results[i] = f1

    return class_f1_results


def eval_fine_grained(runner):
    print(">>> 开始评估（BF1）...")
    model = runner.model
    loader = runner.test_dataloader
    model.eval()

    num_classes = 7
    # 用于计算每个类的平均 BF1
    class_bf1_sums = np.zeros(num_classes)
    class_bf1_counts = np.zeros(num_classes)
    
    all_image_ious = []
    s_inter = 0
    s_union = 0
    SMALL_THRESHOLD = 1024
    total_imgs = len(loader)

    with torch.no_grad():
        for i, data in enumerate(loader):
            outputs = model.test_step(data)
            pred = outputs[0].pred_sem_seg.data.cpu().numpy()[0]
            gt = outputs[0].gt_sem_seg.data.cpu().numpy()[0]

            img_bfs = compute_boundary_f1_per_class(gt, pred, num_classes=num_classes)
            for cls_idx, f1_val in img_bfs.items():
                class_bf1_sums[cls_idx] += f1_val
                class_bf1_counts[cls_idx] += 1

            intersect = np.logical_and(pred == gt, gt > 0).sum()
            union = np.logical_or(pred > 0, gt > 0).sum()
            iou_this_img = intersect / (union + 1e-6)
            all_image_ious.append(iou_this_img)

            for cls_idx in range(num_classes):
                gt_mask = (gt == cls_idx)
                if 0 < gt_mask.sum() < SMALL_THRESHOLD:
                    pred_mask = (pred == cls_idx)
                    s_inter += np.logical_and(gt_mask, pred_mask).sum()
                    s_union += np.logical_or(gt_mask, pred_mask).sum()

            if (i + 1) % 100 == 0:
                print(f"进度: [{i+1}/{total_imgs}]")

    valid_mask = class_bf1_counts > 0
    class_avg_bf1 = np.zeros(num_classes)
    class_avg_bf1[valid_mask] = class_bf1_sums[valid_mask] / class_bf1_counts[valid_mask]
    
    mBF1 = np.mean(class_avg_bf1[valid_mask]) if any(valid_mask) else 0.0
    sIoU = s_inter / (s_union + 1e-6)

    np.save('rsclip_ious.npy', np.array(all_image_ious))

    print("\n--- 类别级 BF1 结果 ---")
    for c in range(num_classes):
        print(f"Class {c}: {class_avg_bf1[c]:.4f}")
    print("----------------------\n")

    res = {'mBF1': mBF1, 'sIoU': sIoU}
    for c in range(num_classes):
        res[f'BF1_Class_{c}'] = class_avg_bf1[c]
        
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='RSCLIP Evaluation Script')
    parser.add_argument('--config', default='./configs/cfg_potsdam.py')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if 'work_dir' not in cfg:
        cfg._cfg_dict['work_dir'] = args.work_dir
    
    cfg.model.model_type = 'RSCLIP'
    
    runner = Runner.from_cfg(cfg)

    input_shape = (3, 448, 448)
    try:
        complexity = get_model_complexity_info(runner.model, input_shape)
        flops_str = complexity.get('flops_str', 'N/A')
    except:
        flops_str = 'N/A'
        
    try:
        fps_val = measure_fps(runner, input_shape=(1, 3, 448, 448))
    except Exception as e:
        print(f"FPS 测量失败: {e}")
        fps_val = 0.0
        
    results = runner.test()
    
    fine_results = eval_fine_grained(runner)

    results.update(fine_results)
    results.update({
        'FLOPs': flops_str,
        'FPS': f"{fps_val:.2f}", 
        'VIT': cfg.model.get('vit_type', 'N/A'),
        'CLIP': cfg.model.get('clip_type', 'N/A'),
        'MODEL': 'RSCLIP',
        'Dataset': cfg.dataset_type
    })

    if runner.rank == 0:
        append_experiment_result('results.xlsx', [results])
        print("-" * 30)
        print(f"最终结果汇总:")
        print(f"mIoU:  {results['mIoU']:.2f}")
        print(f"mBF1:  {results['mBF1']:.4f}")
        print(f"S-IoU: {results['sIoU']:.4f}")
        print(f"FLOPs: {results['FLOPs']}")
        print(f"FPS:   {results['FPS']}") 
        print("-" * 30)

if __name__ == '__main__':
    main()
