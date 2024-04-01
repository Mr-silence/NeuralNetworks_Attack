import sys
import os
sys.path.insert(0, '..')
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from torchattacks import PGD
def preprocess_image(image_path, transform):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def magenet_data(image_path, transform):
    image = preprocess_image(image_path, transform)
    return image

image_path = 'C:\\Users\\1\\Desktop\\PGD\\attackimage\\ILSVRC2012_val_00000003.JPEG'
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])
images = magenet_data(image_path, transform)

print('[Data loaded]')
labels=torch.tensor([432])
device = "cpu"
model = models.resnet18(pretrained=True).to(device).eval()

#acc = get_accuracy(model, [(images.to(device), labels.to(device))])
print('[Model loaded]')
#print('Acc: %2.2f %%'%(acc))


atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(atk)
# random labels as target labels.
atk.set_mode_targeted_random()
adv_images = atk(images, labels)


from utils import imshow, get_pred
idx = 0
pre = get_pred(model, adv_images[idx:idx+1], device)
imshow(adv_images[idx:idx+1], title="True:%d, Pre:%d"%(labels[idx], pre))


def imshow(img , save_path=None):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    # 如果指定了保存路径，则保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建保存图像的文件夹，如果不存在
        plt.imsave(save_path, np.transpose(npimg, (1, 2, 0)))  # 保存图像为 JPEG 文件
        print(f'Image saved to: {save_path}')


# 保存攻击后的图像到固定路径的文件夹
save_folder = 'C:\\Users\\1\\Desktop\\PGD\\save'
os.makedirs(save_folder, exist_ok=True)  # 创建文件夹，如果不存在
adv_images = torchvision.utils.make_grid(adv_images.cpu().data, normalize=True)
for i, adv_image in enumerate(adv_images):
    save_path = os.path.join(save_folder, f'adversarial_image_{idx}.jpeg')
    print(save_path)
    imshow(adv_images,save_path)

print('Adversarial images saved successfully!')

