import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
# 读取Excel文件
excel_file = 'C:\\Users\\1\\Desktop\\副本2.xlsx'
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


chird_birthdates = []

for index, row in df.iterrows():
    try:
        father_birthdate = parse(str(row['孩子出生年']) + str(row['孩子出生月']) + str(row['孩子出生日']))
        chird_birthdates.append(father_birthdate)
    except (ValueError, TypeError):
        # 日期无效或数据类型错误，跳过该行
        chird_birthdates.append(None)

# 添加日期列到 DataFrame
df['孩子出生日期'] = chird_birthdates
print(df['孩子出生日期'])

mother_birthdates = []

for index, row in df.iterrows():
    try:
        father_birthdate = parse(str(row['母亲出生年']) + str(row['母亲出生月']) + str(row['母亲出生日']))
        mother_birthdates.append(father_birthdate)
    except (ValueError, TypeError):
        # 日期无效或数据类型错误，跳过该行
        mother_birthdates.append(None)

# 添加日期列到 DataFrame
df['母亲出生日期'] = mother_birthdates
print(df['母亲出生日期'])


# 将"母亲出生日期"列转换为datetime类型
df['母亲出生日期'] = pd.to_datetime(df['母亲出生日期'], errors='coerce')
print(df['母亲出生日期'])

# 将"孩子出生日期"列转换为datetime类型
df['孩子出生日期'] = pd.to_datetime(df['孩子出生日期'], errors='coerce')
print(df['孩子出生日期'])




# 计算孩子出生日期和母亲出生日期的差值
df['年龄差'] = (df['孩子出生日期'] - df['母亲出生日期']).dt.days / 365


# 打印母亲分娩的年龄
print(df[['年龄差']])

# 打印母亲怀孕的年龄
df['怀孕年龄'] = (df[['年龄差']]-0.75)
print(df['怀孕年龄'])

# 将数据列写入到 Excel 文件
output_file = 'C:\\Users\\1\\Desktop\\副本2.xlsx'  # 输出 Excel 文件名
df[['怀孕年龄']].to_excel(output_file, index=False, sheet_name='Sheet1', header=['怀孕年龄'])

