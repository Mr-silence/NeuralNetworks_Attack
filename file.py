import os
import shutil

# 指定大文件夹路径和导出文件夹路径
大文件夹路径 = 'H:\\imagenet\\imagenets'
导出文件夹路径 = 'H:\\imagenet\\daochu'

# 确保大文件夹存在
if not os.path.exists(大文件夹路径):
    raise FileNotFoundError(f"大文件夹 '{大文件夹路径}' 不存在")

# 确保导出文件夹存在，如果不存在则创建
if not os.path.exists(导出文件夹路径):
    os.makedirs(导出文件夹路径)

# 遍历大文件夹中的每个小文件夹
for 子文件夹名称 in os.listdir(大文件夹路径):
    子文件夹路径 = os.path.join(大文件夹路径, 子文件夹名称)

    # 确保子文件夹是一个目录
    if os.path.isdir(子文件夹路径):
        # 遍历子文件夹中的每个文件
        for 文件名称 in os.listdir(子文件夹路径):
            文件路径 = os.path.join(子文件夹路径, 文件名称)

            # 确保文件是照片（可以根据文件扩展名来判断）
            if 文件名称.endswith(('.jpg', '.JPEG', '.png', '.gif', '.bmp')):
                # 尝试将照片复制到导出文件夹
                目标路径 = os.path.join(导出文件夹路径, 文件名称)
                try:
                    shutil.copy(文件路径, 目标路径)
                except Exception as e:
                    raise Exception(f"无法复制文件 '{文件路径}' 到 '{目标路径}': {str(e)}")

print('照片已成功导出到', 导出文件夹路径)

