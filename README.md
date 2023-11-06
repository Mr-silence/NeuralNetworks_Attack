# NeuralNetworks_Attack

alexnet的github地址：https://github.com/Lornatang/AlexNet-PyTorch

# Deepfool文件
tojpg是攻击后输出jpg文件的代码
tonpy是攻击后输出npy文件的代码
两个代码都是从文件夹中读取图片，攻击后输出到目标文件夹，攻击前和攻击后的文件名是相同的

# PGD文件
使用前先配置torchattacks包，可在终端使用“pip install torchattacks”进行下载
只有输出jpg文件的代码
输入是确定的一张照片的路径，攻击后输出到目标文件夹
