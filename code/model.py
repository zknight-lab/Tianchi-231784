import mod_func as mf

xgb_params = {'n_estimators':100000,
             'learning_rate':0.01,
             'subsample':0.8,
             'colsample_bytree':0.9,
             'max_depth':7,
             'n_jobs':-1
             }

lgb_params = {'boosting_type': 'gbdt',
             'num_leaves': 64,
             'max_depth': 7,
             "lambda_l2": 2,  # 防止过拟合
             'min_data_in_leaf': 20,  # 防止过拟合，好像都不用怎么调
             'objective': 'regression_l1',
             'learning_rate': 0.01,
             "min_child_samples": 20,
             "feature_fraction": 0.8,
             "metric": 'mae',
             }

Train_data = pd.read_csv('../data/train_tree.csv', sep=' ')
TestA_data = pd.read_csv('../data/test_tree.csv', sep=' ')
numerical_cols = Train_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price','SaleID']]
## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
X_test = TestA_data[feature_cols]

X_data = np.array(X_data)
X_test = np.array(X_test)
Y_data = np.array(Train_data['price'])

x_train,x_val,y_train,y_val = train_test_split(X_data,np.expm1(Y_data),test_size=0.3)

model_xgb = mf.build_model_xgb(xgb_params,x_train,y_train)
val_xgb = model_xgb.predict(x_val)
subA_xgb = model_xgb.predict(X_test)

model_lgb = mf.build_model_lgb(lgb_params,x_train,y_train)
val_lgb = model_lgb.predict(x_val)
subA_lgb = model_lgb.predict(X_test)

# 模型融合
## 第一层
train_lgb_pred = model_lgb.predict(x_train)
train_xgb_pred = model_xgb.predict(x_train)

Strak_X_train = pd.DataFrame()
Strak_X_train['Method_1'] = np.log1p(train_lgb_pred)
Strak_X_train['Method_2'] = np.log1p(train_xgb_pred)

Strak_X_val = pd.DataFrame()
Strak_X_val['Method_1'] = np.log1p(val_lgb)
Strak_X_val['Method_2'] = np.log1p(val_xgb)

Strak_X_test = pd.DataFrame()
Strak_X_test['Method_1'] = np.log1p(subA_lgb)
Strak_X_test['Method_2'] = np.log1p(subA_xgb)

# 由于预测值存在负数，用零修正
Strak_X_train=Strak_X_train.fillna(0)
Strak_X_val=Strak_X_val.fillna(0)
Strak_X_test=Strak_X_test.fillna(0)

# 第二层
model_lr_Stacking = mf.build_model_lr(Strak_X_train,np.log1p(y_train))
# 训练集
train_pre_Stacking = model_lr_Stacking.predict(Strak_X_train)
print('MAE of Stacking-LR:',mean_absolute_error(y_train,np.expm1(train_pre_Stacking)))

# 验证集
val_pre_Stacking = model_lr_Stacking.predict(Strak_X_val)
print('MAE of Stacking-LR:',mean_absolute_error(y_val,np.expm1(val_pre_Stacking)))

# 预测集
print('Predict Stacking-LR...')
subA_Stacking = np.expm1(model_lr_Stacking.predict(Strak_X_test))

# 对预测值小于10的进行修正
subA_Stacking[subA_Stacking<10] = 10

sub = pd.DataFrame()
sub['SaleID'] = [i for i in range(200000, 250000)]
sub['price'] = subA_Stacking
sub.to_csv('../data/sub.csv',index=False)
