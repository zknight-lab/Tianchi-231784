import pre_func as pf

train = pf.reduce_mem_usage(pd.read_csv('../data/used_car_train_20200313.csv', sep = ' '))
test = pf.reduce_mem_usage(pd.read_csv('../data/used_car_testB_20200421.csv', sep = ' '))

concat_data = pd.concat([train, test])

# 对相同名称的车进行计数
concat_data['name_count'] = concat_data.groupby(['name'])['SaleID'].transform('count')

#删除极端偏差的数据和难以提取信息的数据
del concat_data['name']
del concat_data['seller']
del concat_data['offerType']
del concat_data['regionCode']

#根据题意，对power进行截尾处理
concat_data['power'][concat_data['power'] > 600] = 600
concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].replace('-',0).astype('float16')

concat_data['regDate'] = concat_data['regDate'].apply(date_process)
concat_data['creatDate'] = concat_data['creatDate'].apply(date_process)
concat_data['regDate_year'] = concat_data['regDate'].dt.year
concat_data['regDate_month'] = concat_data['regDate'].dt.month
concat_data['regDate_day'] = concat_data['regDate'].dt.day
concat_data['creatDate_year'] = concat_data['creatDate'].dt.year
concat_data['creatDate_month'] = concat_data['creatDate'].dt.month
concat_data['creatDate_day'] = concat_data['creatDate'].dt.day
concat_data['car_age_day'] = (concat_data['creatDate'] - concat_data['regDate']).dt.days
concat_data['car_age_year'] = round(concat_data['car_age_day'] / 365, 1)

numerival_val = ['car_age_day', 'car_age_year']
category_val = ['model', 'bodyType', 'fuelType', 'gearbox']
concat_data = pf.missing_value(concat_data, numerival_val, category_val)
concat_data = pf.cut_bins(concat_data.reset_index(), ['power'], 30)
concat_data = pf.cut_bins(concat_data.reset_index(), ['model'], 25)
concat_data.drop(['creatDate', 'regDate'], axis=1, inplace=True)

#特征工程
feature_col1 = ['model', 'brand', 'kilometer', 'bodyType', 'fuelType']
feature_col2 = ['price', 'car_age_day', 'car_age_year', 'power']
concat_data = pf.cross_feature(concat_data, feature_col1, feature_col2)

for i in ['v_' + str(i) for i in range(14)]:
    for j in ['v_' + str(i) for i in range(14)]:
        concat_data[str(i)+'+'+str(j)] = concat_data[str(i)]+concat_data[str(j)]
for i in ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage']:
    for j in ['v_' + str(i) for i in range(14)]:
        concat_data[str(i)+'*'+str(j)] = concat_data[i]*concat_data[j]

drop_cols = list(pf.filter_corr(corr, cutoff=0.95))
cat_list = ['fuelType', 'bodyType','model','gearbox','notRepairedDamage']
feature_list = ['car_age_day', 'car_age_year', 'v_1+v_3', 'v_0+v_10', 'name_count', 'v_14', 'v_0+v_12', 'power*v_11', 'creatDate_day', 'v_10+v_12', 'kilometer*v_6', 'regDate_year', 'kilometer*v_1', 'bodyType*v_11', 'regDate_day', 'v_3+v_4', 'v_1+v_12', 'model_car_age_day_median', 'power*v_1', 'kilometer*v_8', 'regDate_month', 'v_1+v_11', 'power*v_3', 'v_0+v_4', 'power*v_6', 'v_0+v_3', 'v_1+v_4', 'v_6', 'kilometer*v_11', 'power*v_10', 'power*v_9', 'v_0+v_13']

feature_list = [i for i in feature_list if i not in drop_cols]

for i in list(train.columns):
    if i not in feature_list and i not in ['SaleID','name','seller','offerType','regionCode','creatDate', 'regDate']:
        feature_list.append(i)

concat_data.iloc[:len(train),:][feature_list].to_csv('../data/train_tree.csv', index=0, sep=' ')
concat_data.iloc[:len(train),:][feature_list].to_csv('../data/test_tree.csv', index=0, sep=' ')
