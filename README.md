# NeuralNetworks_Attack
attack文件中demo文件夹中包含了4种攻击算法的代码，名字分别为gpd、CW、MIN、DeepFool。请按照需要更改代码使用的device为cpu还是gpu。并更改输入图片的路径和输出图片的文件夹。
labels是图片的原标签，即原图片按照imagenet的1000种分类方法的分类情况，取值为0-999.

file_的py文件是用于攻击imagenet验证集测试成功率的代码，系统小组无需使用。

demo文件夹中的attack_CW.py文件是用于攻击一整个文件夹中的图片，并全部输出到一个文件夹的代码，使用方法和上述一致。
