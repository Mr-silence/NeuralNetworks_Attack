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
from torchattacks import CW

image_folder = 'J:\\神经网络攻击数据\\imagenet\\ILSVRC2012_img_val\\n01443537'
save_folder = 'C:\\Users\\1\\Desktop\\save\\n01443537'

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])

labels=torch.tensor([1])
device = "cpu"
model = models.vgg16(pretrained=True).to(device).eval()
atk = CW(model, c=1, kappa=0, steps=50, lr=0.01)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
atk.set_mode_targeted_random()
def preprocess_image(image_path, transform):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def magenet_data(image_path, transform):
    image = preprocess_image(image_path, transform)
    return image

def img_save(img , save_path=None):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    if save_path:
        plt.imsave(save_path, np.transpose(npimg, (1, 2, 0)))

for filename in os.listdir(image_folder):
    img_path = os.path.join(image_folder, filename)
    print(filename)
    images = magenet_data(img_path, transform)
    print(filename+'[Data loaded]')
    adv_images = atk(images, labels)
    os.makedirs(save_folder, exist_ok=True)
    adv_images = torchvision.utils.make_grid(adv_images.cpu().data, normalize=True)
    save_path = os.path.join(save_folder, f'Adversarial_{filename}.jpeg')
    img_save(adv_images, save_path)
    print(f'Adversarial_{filename}.jpeg')

print('Adversarial images saved successfully!')