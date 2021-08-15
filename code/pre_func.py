import pandas as pd
import numpy as np
import datatime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 数据压缩
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# 缺失值处理，对连续型变量用均值填充， 离散型变量用众数填充
def missing_value(df, numerical_var, category_var):
    df_copy = df.copy()
    for col in numerical_var:
        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    for col in category_var:
        df_copy[col].fillna(pd.Series(df_copy[col]).mode()[0], inplace=True)
    return df_copy

# 异常值处理，箱线图
def outlier(df, col_name, scale = 1.5):
    def iqr_outlier(df_ser, iqr_scale = scale):
        iqr = iqr_scale * (df_ser.quantile(0.75) - df_ser.quantile(0.25))
        val_low = df_ser.quantile(0.25) - iqr
        val_high = df_ser.quantile(0.25) + iqr
        rule_low = (df_ser < val_low)
        rule_high = (df_ser > val_high)
        return (rule_low, rule_high), (val_low, val_high)
    df_copy = df.copy()
    for col in col_name:
        df_serices = df_copy[col]
        rule, val = iqr_outlier(df_serices, scale)
        tag = rule[0].astype('int')
        new = tag * df_serices
        df_serices = new.replace(0, df_serices.quantile(0.995))
        tag = rule[1].astype('int')
        new = tag * df_serices
        df_serices = new.replace(0, df_serices.quantile(0.07))
        df_copy[col] = df_serices
    return df_copy

#定义交叉特征统计
def cross_feature(df,fea_Col1,fea_Col2):
    for f1 in tqdm(fea_Col1):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(fea_Col2):
            feat = g[f2].agg(['max','min','median','sum','mad','std']).rename(columns={'max':f'{f1}_{f2}_max','min':f'{f1}_{f2}_min',
                                                                                      'median':f'{f1}_{f2}_median', 'sum':f'{f1}_{f2}_sum',
                                                                                      'mad':f'{f1}_{f2}_mad', 'std':f'{f1}_{f2}_std'
                                                                                      })
            df = df.merge(feat, on=f1, how='left')
    return df

# 数据分箱
def cut_bins(df, col_name, num_bins = 50):
    for col in col_name:
        df[col+'_bin'] = pd.cut(df[col], num_bins, labels = False)
    return df

# 日期处理
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])
    if month < 1:
        month = 1
    date = datetime(year, month, day)
    return date

# 删除相关性高的变量
def filter_corr(corr, cutoff=0.9):
    cols = []
    for i,j in feature_group:
        if corr.loc[i, j] > cutoff:
            print(i,j,corr.loc[i, j])
            i_avg = corr[i][corr[i] != 1].mean()
            j_avg = corr[j][corr[j] != 1].mean()
            if i_avg >= j_avg:
                cols.append(i)
            else:
                cols.append(j)
    return set(cols)

if __name__ == '__main__':
    pass


