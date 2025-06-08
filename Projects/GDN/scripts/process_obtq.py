"""
整体流程：原始 CSV 数据（train/test） → 缺失值填充 → 标准化（Min-Max） → 降采样（downsample） → 生成新的 train.csv 和 test.csv

"""

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):  # 数据归一化函数

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret

# downsample by 2
def downsample(data, labels, down_len):  # 降采样函数
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()  # 转置数据。shape: (F, T)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)  # 按时间窗口 reshape 并取中位数（每 down_len 时间步一个片段）

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))  # 如果一个时间窗口内 存在任意一个异常点，整个窗口就被视为异常


    d_data = d_data.transpose()  # 转置回来。 shape: (T/down_len, F)

    return d_data.tolist(), d_labels.tolist()


def main():

    test = pd.read_csv('./obtq2_test.csv', index_col=0)
    train = pd.read_csv('./obtq2_train.csv', index_col=0)


    test = test.iloc[:, 1:]  # 丢弃时间戳列
    train = train.iloc[:, 1:]

    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)

    # trim column names
    train = train.rename(columns=lambda x: x.strip())  # 列名去空格
    test = test.rename(columns=lambda x: x.strip())

    # print(len(test.columns),test.columns)
    # print(len(train.columns),train.columns)


    train_labels = train.attack
    test_labels = test.attack

    train = train.drop(columns=['attack'])  # 分离标签列
    test = test.drop(columns=['attack'])


    x_train, x_test = norm(train.values, test.values)  # 归一化特征


    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]


    d_train_x, d_train_labels = downsample(train.values, train_labels, 2)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 2)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)

    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    # train_df = train_df.iloc[2160:]

    # print(train_df.values.shape)
    # print(test_df.values.shape)


    train_df.to_csv('./train.csv')
    test_df.to_csv('./test.csv')

    f = open('./list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
