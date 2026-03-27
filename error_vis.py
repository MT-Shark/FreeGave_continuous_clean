from PIL import Image, ImageChops
import matplotlib.pyplot as plt

# def visualize_image_difference(image1_path, image2_path):
#     # 打开两张图像
#     image1 = Image.open(image1_path)
#     image2 = Image.open(image2_path)
    
#     # 确保图像具有相同的模式和大小
#     if image1.mode != image2.mode or image1.size != image2.size:
#         raise ValueError("图像必须具有相同的模式和大小")
    
#     # 计算图像之间的差异
#     diff = ImageChops.difference(image1, image2)
    
#     # 将差异转换为灰度图像
#     diff_gray = diff.convert('L')
    
#     # 只展示差异部分
#     plt.imshow(diff_gray, cmap='gray')
#     plt.title('差异')
#     plt.axis('off')
#     plt.show()

# # 示例用法：
# image1 = r'output\pen1_shortwarmup_7000_threshold_0.0002\increment_20_100_noStaticMask\test\0024\render\0198.png'
# # image1 = r'C:\Users\Public\Downloads\Phd\Vlag_baselines\Def3G_nvfi_incremental_controlNode\output\pen1_end2end_controlNode_shortwarmup_0.0002_20\increment_20_100_noStaticMask\test\0024\render\0198.png'
# image2 = r'output\pen1_shortwarmup_7000_threshold_0.0002\increment_20_100_noStaticMask\test\0024\gt\0198.png'

# visualize_image_difference(image1, image2)

import os
from PIL import Image, ImageChops

import os
from PIL import Image, ImageChops

test_dir = r'output\box_controlNode_shortwarmup_0.0002_20\increment_20_100_noStaticMask\test'

# 遍历 test_dir，获取所有子目录
for i , root_dir in enumerate(os.listdir(test_dir)):
    # if i > 25:
    #     break
    # if root_dir != '0024':
    #     continue
    print(root_dir)
    root_path = os.path.join(test_dir, root_dir)
    
    if os.path.isdir(root_path):
        # 创建 error vis 文件夹
        error_vis_dir = os.path.join(root_path, 'error_vis')
        os.makedirs(error_vis_dir, exist_ok=True)

        render_dir = os.path.join(root_path, 'render')
        gt_dir = os.path.join(root_path, 'gt')

        # 遍历 render 文件夹，获取文件名称
        for image_name in os.listdir(render_dir):
            image1_path = os.path.join(render_dir, image_name)
            image2_path = os.path.join(gt_dir, image_name)
            
            # 打开两张图像
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            
            # 确保图像具有相同的模式和大小
            if image1.mode != image2.mode or image1.size != image2.size:
                raise ValueError("图像必须具有相同的模式和大小")
            
            # 计算图像之间的差异
            diff = ImageChops.difference(image1, image2)
            
            # 将差异转换为灰度图像
            diff_gray = diff.convert('L')
            
            # 保存差异图像到 error vis 文件夹
            diff_image_path = os.path.join(error_vis_dir, image_name)
            diff_gray.save(diff_image_path)