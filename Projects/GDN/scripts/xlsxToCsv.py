import pandas as pd

# 读取 xlsx 文件
df = pd.read_excel('swat_train.xlsx')  # 若有多工作表，需指定 sheet_name

# 保存为 csv 文件
df.to_csv('swat_train.csv', index=False)  # index=False 表示不保存行索引