"""
图像标签GT 可视化工具
======================

这个脚本用于将现有的mask文件与原图混合，生成语义分割可视化结果。

主要功能：
- 支持输出正方形或保持原始比例
- 透明覆盖效果（可调透明度）
- 轮廓线绘制
- 文本标签自动添加
- 颜色图例生成
- 多种输出格式

使用方法：
1. 修改main()函数中的文件路径
2. 设置MAKE_SQUARE=True输出正方形，False保持原比例
3. 设置STRETCH_MODE=True拉伸短边，False裁剪图片
4. 设置TARGET_SIZE指定正方形尺寸，None为自动
5. 运行脚本

输出文件：
- mask_segmentation_comparison.png: 四宫格对比图
- mask_overlay_with_contours_and_labels.png: 带轮廓和标签
- mask_overlay_with_labels_only.png: 只有标签
- 其他多种格式的输出文件
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import cv2

def make_square_image(img, target_size=None, stretch_mode=True):
    """
    将图像调整为正方形
    
    Args:
        img: PIL图像对象
        target_size: 目标尺寸，如果为None则使用原图的最长边长
        stretch_mode: True为拉伸模式（拉伸短边），False为裁剪模式（裁剪长边）
    
    Returns:
        square_img: 正方形PIL图像对象
    """
    width, height = img.size
    
    if target_size is None:
        if stretch_mode:
            # 拉伸模式：使用较长的边长作为正方形的边长
            target_size = max(width, height)
        else:
            # 裁剪模式：使用较短的边长作为正方形的边长
            target_size = min(width, height)
    
    if stretch_mode:
        # 拉伸模式：直接resize到目标尺寸
        square_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    else:
        # 裁剪模式：中心裁剪
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        square_img = img.crop((left, top, right, bottom))
    
    return square_img

def load_mask_and_image(image_path, mask_path, make_square=True, target_size=None, stretch_mode=True):
    """
    加载原图和mask文件
    
    Args:
        image_path: 原图路径
        mask_path: mask文件路径
        make_square: 是否调整为正方形
        target_size: 目标正方形尺寸，如果为None则使用原图的最长边长（拉伸模式）或最短边长（裁剪模式）
        stretch_mode: True为拉伸模式，False为裁剪模式
    
    Returns:
        original_img: PIL图像对象
        mask_array: numpy数组，包含分割mask
    """
    # 加载原图
    original_img = Image.open(image_path)
    
    # 加载mask
    mask_img = Image.open(mask_path)
    
    # 如果mask是RGB图像，转换为灰度图
    if len(np.array(mask_img).shape) == 3:
        mask_img = mask_img.convert('L')
    
    # 调整为正方形
    if make_square:
        original_img = make_square_image(original_img, target_size, stretch_mode)
        mask_img = make_square_image(mask_img, target_size, stretch_mode)
    
    mask_array = np.array(mask_img)
    
    return original_img, mask_array

def create_colored_segmentation(seg_pred, color_palette):
    """
    将分割结果转换为自定义颜色的RGB图像
    
    Args:
        seg_pred: 分割预测结果 (H, W)
        color_palette: 颜色调色板，格式为 [[R,G,B], [R,G,B], ...]
    
    Returns:
        colored_seg: RGB图像 (H, W, 3)
    """
    h, w = seg_pred.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_classes = np.unique(seg_pred)
    
    for class_id in unique_classes:
        if class_id < len(color_palette):
            mask = (seg_pred == class_id)
            colored_seg[mask] = color_palette[class_id]
        else:
            # 如果类别超出颜色调色板范围，使用白色
            mask = (seg_pred == class_id)
            colored_seg[mask] = [255, 255, 255]
    
    return colored_seg

def create_overlay(original_img, seg_pred, color_palette, alpha=0.5):
    """
    创建分割结果与原图的透明覆盖

    Args:
        original_img: 原始图像 (PIL Image 或 numpy array)
        seg_pred: 分割预测结果 (H, W)
        color_palette: 颜色调色板
        alpha: 透明度 (固定为0.5)

    Returns:
        overlay_img: 覆盖后的图像 (numpy array)
    """
    # 确保原图是numpy数组格式
    if isinstance(original_img, Image.Image):
        original_array = np.array(original_img)
    else:
        original_array = original_img.copy()

    # 调整原图尺寸匹配分割结果
    if original_array.shape[:2] != seg_pred.shape:
        original_pil = Image.fromarray(original_array)
        original_pil = original_pil.resize((seg_pred.shape[1], seg_pred.shape[0]), Image.Resampling.LANCZOS)
        original_array = np.array(original_pil)

    # 创建彩色分割图
    h, w = seg_pred.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    mask_overlay = np.zeros((h, w), dtype=bool)  # 用于标记需要覆盖的区域

    unique_classes = np.unique(seg_pred)

    for class_id in unique_classes:
        # 忽略背景类别（最后一个类别）
        if class_id == len(color_palette) - 1:
            continue

        if class_id < len(color_palette):
            class_mask = (seg_pred == class_id)
            colored_seg[class_mask] = color_palette[class_id]
            mask_overlay = mask_overlay | class_mask

    # 创建覆盖图像
    overlay_img = original_array.astype(np.float32)

    # 在有分割结果的区域进行alpha混合 (alpha=0.5)
    for i in range(3):  # RGB三个通道
        overlay_img[mask_overlay, i] = (
            alpha * original_array[mask_overlay, i] +
            (1 - alpha) * colored_seg[mask_overlay, i]
        )

    return overlay_img.astype(np.uint8)

def add_contours_to_overlay(overlay_img, seg_pred, contour_color=(255, 255, 255), contour_thickness=2):
    """
    在透明覆盖图上添加轮廓线

    Args:
        overlay_img: 透明覆盖图像
        seg_pred: 分割预测结果
        contour_color: 轮廓线颜色 (R, G, B)
        contour_thickness: 轮廓线粗细

    Returns:
        overlay_with_contours: 带轮廓的覆盖图像
    """
    overlay_with_contours = overlay_img.copy()
    unique_classes = np.unique(seg_pred)

    # 为每个类别添加轮廓线（排除背景）
    for class_id in unique_classes:
        if class_id == len(unique_classes) - 1:  # 跳过背景
            continue
        class_mask = (seg_pred == class_id).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        cv2.drawContours(overlay_with_contours, contours, -1, contour_color, contour_thickness)

    return overlay_with_contours

def add_text_labels(ax, seg_pred, name_list, min_area_threshold=200):
    """
    在分割图上添加文本标签
    
    Args:
        ax: matplotlib的轴对象
        seg_pred: 分割预测结果 (H, W)
        name_list: 类别名称列表
        min_area_threshold: 最小区域面积阈值，小于此值的区域不显示标签
    """
    unique_classes = np.unique(seg_pred)
    
    for class_id in unique_classes:
        if class_id >= len(name_list):
            continue
            
        # 创建当前类别的mask
        class_mask = (seg_pred == class_id)
        
        # 计算区域面积，过滤太小的区域
        area = np.sum(class_mask)
        if area < min_area_threshold:
            continue
        
        # 使用连通组件分析，找到最大的连通区域
        labeled_array, num_features = ndimage.label(class_mask)
        
        if num_features > 0:
            # 找到最大的连通组件
            component_sizes = ndimage.sum(class_mask, labeled_array, range(1, num_features + 1))
            largest_component = np.argmax(component_sizes) + 1
            largest_mask = (labeled_array == largest_component)
            
            # 计算最大连通组件的质心
            center_of_mass = ndimage.center_of_mass(largest_mask)
            
            if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
                y, x = center_of_mass
                
                # 获取类别名称，处理可能的逗号分隔
                class_name = name_list[class_id]
                if ',' in class_name:
                    class_name = class_name.split(',')[0]  # 取第一个名称
                
                # 添加文本标签
                ax.text(x, y, class_name, 
                       color='white', 
                       fontsize=25,
                       fontweight='bold',
                       ha='center', 
                       va='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', 
                                alpha=0.7,
                                edgecolor='white',
                                linewidth=1))

def create_color_legend(name_list, custom_colors, save_path='color_legend.png'):
    """创建颜色图例"""
    fig, ax = plt.subplots(figsize=(8, len(name_list) * 0.5))
    
    for i, (name, color) in enumerate(zip(name_list, custom_colors)):
        # 清理类别名称
        clean_name = name.split(',')[0] if ',' in name else name
        
        # 绘制颜色块
        rect = plt.Rectangle((0, i), 1, 0.8, facecolor=np.array(color)/255.0)
        ax.add_patch(rect)
        
        # 添加文本
        ax.text(1.1, i + 0.4, f"{i}: {clean_name}", 
                va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, len(name_list))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Segmentation Color Legend', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"颜色图例已保存为 '{save_path}'")

def main():
    # 文件路径
    image_path = 'demo/demo_1.JPG'
    mask_path = 'demo/demo_1.png'
    
    # ===== 输出尺寸设置 =====
    MAKE_SQUARE = True   # 设置为True输出正方形，False保持原始比例
    STRETCH_MODE = True  # 设置为True拉伸短边，False裁剪长边
    TARGET_SIZE = None   # 正方形边长设置：
                        # None: 拉伸模式使用最长边，裁剪模式使用最短边
                        # 512: 输出512x512的正方形
                        # 1024: 输出1024x1024的正方形
                        # 等等...
    
    # 类别名称列表（与原代码保持一致）
    name_list = ['Vegetation','Building','Road','Vehicle','Background']

    # 自定义颜色调色板（与原代码保持一致）
    custom_colors = [
        [107,142,35], # vegetation - 橄榄绿
        [102, 102, 156], # building - 蓝灰色
        [128, 64, 128],  # road - 紫色
        [0, 0, 142],     # vehicle - 深蓝色
        [0, 0, 0]        # background - 黑色
    ]
    
    # 加载图像和mask
    print(f"加载图像: {image_path}")
    print(f"加载mask: {mask_path}")
    if MAKE_SQUARE:
        mode_text = "拉伸模式（拉伸短边）" if STRETCH_MODE else "裁剪模式（裁剪长边）" 
        print(f"将调整为正方形输出，模式: {mode_text}，目标尺寸: {'自动' if TARGET_SIZE is None else TARGET_SIZE}")
    
    try:
        original_img, mask_array = load_mask_and_image(image_path, mask_path, 
                                                      make_square=MAKE_SQUARE, 
                                                      target_size=TARGET_SIZE,
                                                      stretch_mode=STRETCH_MODE)
        print(f"图像尺寸: {original_img.size}")
        print(f"Mask尺寸: {mask_array.shape}")
        print(f"Mask中的唯一值: {np.unique(mask_array)}")
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return
    
    # 创建彩色分割图
    colored_segmentation = create_colored_segmentation(mask_array, custom_colors)
    
    # 创建透明覆盖
    overlay_img = create_overlay(original_img, mask_array, custom_colors, alpha=0.5)
    
    # 创建带轮廓的覆盖图
    overlay_with_contours = add_contours_to_overlay(overlay_img, mask_array,
                                                   contour_color=(255, 255, 255),
                                                   contour_thickness=32)
    
    # 创建对比图：原图、纯分割结果、无轮廓覆盖、带轮廓覆盖
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))  # 改为正方形布局
    
    # 原图
    ax[0, 0].imshow(original_img)
    ax[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    ax[0, 0].axis('off')
    ax[0, 0].set_aspect('equal')  # 确保显示比例为1:1
    
    # 纯分割结果
    ax[0, 1].imshow(colored_segmentation)
    ax[0, 1].set_title('Pure Segmentation', fontsize=14, fontweight='bold')
    ax[0, 1].axis('off')
    ax[0, 1].set_aspect('equal')
    
    # 透明覆盖（无轮廓）
    ax[1, 0].imshow(overlay_img)
    ax[1, 0].set_title('Overlay (Alpha=0.5)', fontsize=14, fontweight='bold')
    ax[1, 0].axis('off')
    ax[1, 0].set_aspect('equal')
    
    # 透明覆盖（带轮廓）
    ax[1, 1].imshow(overlay_with_contours)
    ax[1, 1].set_title('Overlay with Contours (Alpha=0.5)', fontsize=14, fontweight='bold')
    ax[1, 1].axis('off')
    ax[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('mask_segmentation_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 生成两个最终版本
    
    # 版本1: 带轮廓和文本标签
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))  # 改为正方形
    ax1.imshow(overlay_with_contours)
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title('Overlay with Contours and Labels', fontsize=16, fontweight='bold', pad=20)
    
    # 添加文本标签
    add_text_labels(ax1, mask_array, name_list, min_area_threshold=200)
    
    plt.tight_layout()
    plt.savefig('mask_overlay_with_contours_and_labels.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 版本2: 无轮廓但有文本标签
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))  # 改为正方形
    ax2.imshow(overlay_img)
    ax2.axis('off')
    ax2.set_aspect('equal')
    ax2.set_title('Overlay with Labels (No Contours)', fontsize=16, fontweight='bold', pad=20)
    
    # 添加文本标签
    add_text_labels(ax2, mask_array, name_list, min_area_threshold=200)
    
    plt.tight_layout()
    plt.savefig('mask_overlay_with_labels_only.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存纯图像版本（无matplotlib边框）
    overlay_pil = Image.fromarray(overlay_img)
    overlay_pil.save('mask_overlay_no_contours.png')
    
    overlay_contours_pil = Image.fromarray(overlay_with_contours)
    overlay_contours_pil.save('mask_overlay_with_contours.png')
    
    # 保存彩色分割结果
    custom_seg_image = Image.fromarray(colored_segmentation)
    custom_seg_image.save('mask_custom_colored_segmentation.png')
    
    # 创建颜色图例
    create_color_legend(name_list, custom_colors, 'mask_color_legend.png')
    
    mode_text = "拉伸模式" if STRETCH_MODE else "裁剪模式"
    print(f"\n结果已保存（正方形格式 - {mode_text}）:")
    print("- 'mask_segmentation_comparison.png': 四宫格对比图")
    print("- 'mask_overlay_with_contours_and_labels.png': 带轮廓和标签的最终图")
    print("- 'mask_overlay_with_labels_only.png': 只有标签的最终图")
    print("- 'mask_overlay_with_contours.png': 纯带轮廓覆盖图像")
    print("- 'mask_overlay_no_contours.png': 纯无轮廓覆盖图像")
    print("- 'mask_custom_colored_segmentation.png': 彩色分割结果")
    print("- 'mask_color_legend.png': 颜色图例")
    print(f"图像尺寸: {original_img.size[0]}x{original_img.size[1]} (正方形)")
    print(f"处理模式: {mode_text}")
    print(f"透明度: 0.5, 轮廓颜色: 白色, 轮廓粗细: 2px")
    
    print(f"\n检测到的类别: {np.unique(mask_array)}")
    print("对应的类别名称和颜色:")
    for class_id in np.unique(mask_array):
        if class_id < len(name_list) and class_id < len(custom_colors):
            print(f"  {class_id}: {name_list[class_id]} - RGB{custom_colors[class_id]}")

if __name__ == "__main__":
    main()