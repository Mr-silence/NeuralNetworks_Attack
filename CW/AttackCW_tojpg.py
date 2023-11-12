import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import torchattacks

# 定义图像加载和转换
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

# 从文件加载图像
image_path = 'C:\\Users\\1\\Desktop\\CW\\attackimage\\ILSVRC2012_val_00000003.JPEG'
image = load_image(image_path)
image = image.unsqueeze(0)  # 添加 batch 维度

# 加载ResNet-18模型
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
model.eval()

# 创建CW攻击对象
atk = torchattacks.CW(model)

# 设置攻击参数
atk.targeted = False  # 非目标攻击
atk.c = 0.1  # 控制攻击的距离
atk.iterations = 100 #迭代

# 提供虚拟标签（dummy label）
dummy_labels = torch.zeros(1, dtype=torch.long).to(image.device)

adv_images = atk(image, dummy_labels)

# 定义保存攻击后图像的目录
output_dir = 'C:\\Users\\1\\Desktop\\CW\\save'
os.makedirs(output_dir, exist_ok=True)

# 保存攻击后的图像
for i, adv_image in enumerate(adv_images):
    adv_image = adv_image.squeeze(0)  # 移除 batch 维度
    adv_image = transforms.ToPILImage()(adv_image)  # 转换为PIL图像
    adv_image_path = os.path.join(output_dir, f'adversarial_image_{i}.jpg')
    adv_image.save(adv_image_path)
    print(f'Saved adversarial image {i} to {adv_image_path}')
