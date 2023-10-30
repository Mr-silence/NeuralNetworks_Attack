import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

# 读取Excel文件
excel_file = 'C:\\Users\\1\\Desktop\\副本.xls'
df = pd.read_excel(excel_file)

# 选择要读取的列
column_name_child = '身份证号码'
column_name_mother = '母亲身份证号码'

# 提取出生年、月、日信息
df['孩子出生年'] = df[column_name_child].str[6:10]
df['孩子出生月'] = df[column_name_child].str[10:12]
df['孩子出生日'] = df[column_name_child].str[12:14]

df['母亲出生年'] = df[column_name_mother].str[6:10]
df['母亲出生月'] = df[column_name_mother].str[10:12]
df['母亲出生日'] = df[column_name_mother].str[12:14]

# 创建一个空的列表来存储日期
child_birthdates = []
mother_birthdates = []

# 逐行检查日期数据并转换
for index, row in df.iterrows():
    try:
        child_birthdate = parse(str(row['孩子出生年']) + str(row['孩子出生月']) + str(row['孩子出生日']))
        mother_birthdate = parse(str(row['母亲出生年']) + str(row['母亲出生月']) + str(row['母亲出生日']))
        child_birthdates.append(child_birthdate)
        mother_birthdates.append(mother_birthdate)
    except (ValueError, TypeError):
        # 日期无效或数据类型错误，跳过该行
        child_birthdates.append(None)
        mother_birthdates.append(None)

# 添加日期列到 DataFrame，并将其转换为datetime格式
df['孩子出生日期'] = pd.to_datetime(child_birthdates, format='%Y%m%d', errors='coerce')
df['母亲出生日期'] = pd.to_datetime(mother_birthdates, format='%Y%m%d', errors='coerce')



# 计算年龄差值
df['年龄差'] = (df['孩子出生日期'] - df['母亲出生日期']).dt.days / 365

# 处理父亲出生日期的类似问题
column_name_father = '父亲身份证号码'
df['父亲出生年'] = df[column_name_father].str[6:10]
df['父亲出生月'] = df[column_name_father].str[10:12]
df['父亲出生日'] = df[column_name_father].str[12:14]

father_birthdates = []

for index, row in df.iterrows():
    try:
        father_birthdate = parse(str(row['父亲出生年']) + str(row['父亲出生月']) + str(row['父亲出生日']))
        father_birthdates.append(father_birthdate)
    except (ValueError, TypeError):
        # 日期无效或数据类型错误，跳过该行
        father_birthdates.append(None)

# 添加日期列到 DataFrame，并将其转换为datetime格式
df['父亲出生日期'] = pd.to_datetime(father_birthdates, format='%Y%m%d', errors='coerce')

# 计算父亲在孩子怀孕时的年龄
df['父亲在孩子怀孕时的年龄'] = (df['孩子出生日期'] - df['父亲出生日期']).dt.days / 365

# 打印父亲在孩子怀孕的年龄
df['父亲在孩子怀孕时的年龄'] = df['父亲在孩子怀孕时的年龄'] - 0.75
print(df['父亲在孩子怀孕时的年龄'])

# 将数据列写入到 Excel 文件
output_file = 'C:\\Users\\1\\Desktop\\副本2.xlsx'  # 输出 Excel 文件名
df[['父亲在孩子怀孕时的年龄']].to_excel(output_file, index=False, sheet_name='Sheet1', header=['父亲在孩子怀孕时的年龄'])
