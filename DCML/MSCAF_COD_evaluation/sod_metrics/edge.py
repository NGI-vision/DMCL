import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
from PVT_Model.pvtmodel import PvtNet, Edge_Module  # 请替换为你的模型导入路径

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = PvtNet(args=None).to(device)  # 需要根据实际情况传递参数
model.eval()  # 切换到推理模式

# 定义输入图像的预处理（调整大小，标准化等）
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像的函数
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度并转移到 GPU
    return image

# 加载 RGB 和 Depth 图像
rgb_image_path = "./dataset/test_data/CAMO/RGB/camourflage_00120.jpg"  # 替换为你的 RGB 图像路径
depth_image_path = "./dataset/test_data/CAMO/T/camourflage_00120.png"  # 替换为你的 Depth 图像路径
rgb_image = load_image(rgb_image_path)
depth_image = load_image(depth_image_path)

x2_rgb, x4_rgb, x5_rgb = rgb_image, rgb_image, rgb_image  # 模拟 RGB 特征层
x2_depth, x4_depth, x5_depth = depth_image, depth_image, depth_image  # 模拟 Depth 特征层

# 将 RGB 和 Depth 图像传入模型进行推理
with torch.no_grad():
    edge_pred = model.edge_layer(
        rgb_image, 
        x2_rgb, x4_rgb, x5_rgb, 
        x2_depth, x4_depth, x5_depth
    )

# 获取 RGB 和 Depth 边缘图
rgb_edge_pred = edge_pred[0].squeeze().cpu().numpy()  # RGB-edge
depth_edge_pred = edge_pred[1].squeeze().cpu().numpy()  # Depth-edge


output_folder = "output_edges"  
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  


rgb_edge_path = os.path.join(output_folder, "rgb_edge_pred.png")
depth_edge_path = os.path.join(output_folder, "depth_edge_pred.png")


plt.imsave(rgb_edge_path, rgb_edge_pred, cmap='gray')
plt.imsave(depth_edge_path, depth_edge_pred, cmap='gray')

print(f"RGB-edge and Depth-edge saved as '{rgb_edge_path}' and '{depth_edge_path}'.")

