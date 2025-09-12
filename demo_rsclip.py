"""
RSCLIP 可视化工具
======================

这个脚本用于生成各种形式的语义分割可视化结果。

主要功能：
- 透明覆盖效果（可调透明度）
- 轮廓线绘制
- 文本标签自动添加
- 颜色图例生成
- 多种输出格式
- 多种对比格式

使用方法：
1. 修改文件路径
2. 设置数据集类别提示词
3. create_colored_segmentation函数中设置数据集的RGB颜色映射
4. 设置min_area_threshold类别文本显示的最小阈值面积
5. 运行脚本

输出文件：
- segmentation_comparison.png: 四宫格对比图
- segmentation_overlay_with_contours_and_labels.png: 带轮廓和标签
- segmentation_overlay_with_labels_only.png: 只有标签
- 其他多种格式的输出文件
"""

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segmentor import RSCLIPSegmentation

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
img = Image.open('demo/demo_1.JPG')

name_list = ['Vegetation','Building','Road','Vehicle','Background'] #UDD5

with open('./configs/cls_udd5.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
    writers.close()
#此处为CLIP模型的标准化参数，不可随意更改
img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = RSCLIPSegmentation(
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='RSCLIP',
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    name_path='./configs/cls_udd5.txt',
    prob_thd=0.1,
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

# ===== 自定义颜色映射 =====
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

custom_colors = [
    [107,142,35], #1
    [102, 102, 156],#1
    [128, 64, 128],  #1
    [0, 0, 142],
    [0, 0, 0]
]

# 创建彩色分割图
colored_segmentation = create_colored_segmentation(seg_pred, custom_colors)

# ===== 文本标签添加代码 =====
def add_text_labels(ax, seg_pred, name_list, min_area_threshold=100):
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
                       fontsize=20,
                       fontweight='bold',
                       ha='center', 
                       va='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', 
                                alpha=0.7,
                                edgecolor='white',
                                linewidth=1))

# 保存自定义颜色的分割结果为单独的图像
from PIL import Image as PILImage

# ===== 可选：创建颜色图例 =====
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

# 创建颜色图例
create_color_legend(name_list, custom_colors)
#
# ===== 透明覆盖功能 =====
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
    if isinstance(original_img, PILImage.Image):
        original_array = np.array(original_img)
    else:
        original_array = original_img.copy()

    # 调整原图尺寸匹配分割结果
    if original_array.shape[:2] != seg_pred.shape:
        original_pil = PILImage.fromarray(original_array)
        original_pil = original_pil.resize((seg_pred.shape[1], seg_pred.shape[0]), PILImage.Resampling.LANCZOS)
        original_array = np.array(original_pil)

    # 创建彩色分割图
    h, w = seg_pred.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    mask_overlay = np.zeros((h, w), dtype=bool)  # 用于标记需要覆盖的区域

    unique_classes = np.unique(seg_pred)

    for class_id in unique_classes:
        if class_id <= len(color_palette):
            class_mask = (seg_pred == class_id)
            colored_seg[class_mask] = color_palette[class_id]
            mask_overlay = mask_overlay | class_mask

    # 创建覆盖图像
    overlay_img = original_array.astype(np.float32)

    # 在有分割结果的区域进行alpha混合 (alpha=0.5)
    for i in range(3):  # RGB三个通道
        overlay_img[mask_overlay, i] = (
            0.5 * original_array[mask_overlay, i] +
            0.5 * colored_seg[mask_overlay, i]
        )

    return overlay_img.astype(np.uint8)

# ===== Contours功能 =====
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
    import cv2

    overlay_with_contours = overlay_img.copy()
    unique_classes = np.unique(seg_pred)

    for class_id in unique_classes:
        class_mask = (seg_pred == class_id).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        cv2.drawContours(overlay_with_contours, contours, -1, contour_color, contour_thickness)

    return overlay_with_contours

# ===== 主要可视化代码 =====
# 创建0.5倍透明覆盖
overlay_img = create_overlay(img, seg_pred, custom_colors, alpha=0.5)

# 创建带轮廓的覆盖图
overlay_with_contours = add_contours_to_overlay(overlay_img, seg_pred,
                                               contour_color=(255, 255, 255),
                                               contour_thickness=2)

# 创建对比图：原图、纯分割结果、无轮廓覆盖、带轮廓覆盖
fig, ax = plt.subplots(2, 2, figsize=(20, 16))

# 原图
ax[0, 0].imshow(img)
ax[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
ax[0, 0].axis('off')

# 纯分割结果
ax[0, 1].imshow(colored_segmentation)
ax[0, 1].set_title('Pure Segmentation', fontsize=14, fontweight='bold')
ax[0, 1].axis('off')

# 透明覆盖（无轮廓）
ax[1, 0].imshow(overlay_img)
ax[1, 0].set_title('Overlay (Alpha=0.5)', fontsize=14, fontweight='bold')
ax[1, 0].axis('off')

# 透明覆盖（带轮廓）
ax[1, 1].imshow(overlay_with_contours)
ax[1, 1].set_title('Overlay with Contours (Alpha=0.5)', fontsize=14, fontweight='bold')
ax[1, 1].axis('off')

plt.tight_layout()
plt.savefig('segmentation_comparison.png', bbox_inches='tight', dpi=300)
# plt.show()

# ===== 生成两个最终版本 =====

# 版本1: 带轮廓和文本标签
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.imshow(overlay_with_contours)
ax1.axis('off')
ax1.set_title('Overlay with Contours and Labels', fontsize=16, fontweight='bold', pad=20)

# 添加文本标签
add_text_labels(ax1, seg_pred, name_list, min_area_threshold=100)

plt.tight_layout()
plt.savefig('segmentation_overlay_with_contours_and_labels.png', bbox_inches='tight', dpi=300)
plt.close(fig1)

# 版本2: 无轮廓但有文本标签
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
ax2.imshow(overlay_img)
ax2.axis('off')
ax2.set_title('Overlay with Labels (No Contours)', fontsize=16, fontweight='bold', pad=20)

# 添加文本标签
add_text_labels(ax2, seg_pred, name_list, min_area_threshold=100)

plt.tight_layout()
plt.savefig('segmentation_overlay_with_labels_only.png', bbox_inches='tight', dpi=300)
plt.close(fig2)

# 保存纯图像版本（无matplotlib边框）
overlay_pil = PILImage.fromarray(overlay_img)
overlay_pil.save('overlay_no_contours.png')

overlay_contours_pil = PILImage.fromarray(overlay_with_contours)
overlay_contours_pil.save('overlay_with_contours.png')

print("结果已保存:")
print("- 'segmentation_comparison.png': 四宫格对比图")
print("- 'segmentation_overlay_with_contours_and_labels.png': 带轮廓和标签的最终图")
print("- 'segmentation_overlay_with_labels_only.png': 只有标签的最终图")
print("- 'overlay_with_contours.png': 纯带轮廓覆盖图像")
print("- 'overlay_no_contours.png': 纯无轮廓覆盖图像")
print(f"透明度: 0.5, 轮廓颜色: 白色, 轮廓粗细: 2px")