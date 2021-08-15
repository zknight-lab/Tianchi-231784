import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold, train_test_split
# 自定义损失函数
def myFeval(preds, train):
    label = train.get_label()
    score = mean_absolute_error(label, preds)
    return 'myFeval', score, False
def myFeval_min(preds, train):
    label = train.get_label()
    return 'myFeval', mean_absolute_error(label, preds)

# 创建模型
def build_model_xgb(params, X_data,Y_data, n_splits=5, random_state=2021):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_xgb = np.zeros(len(X_data))
    scores = []
    model = xgb.XGBRegressor(**params)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):
        print("fold n°{}".format(fold_ + 1))
        clf = model.fit(X_data[trn_idx], Y_data[trn_idx],eval_set=[(X_data[val_idx], Y_data[val_idx])], eval_metric=myFeval_min, early_stopping_rounds=300, verbose=300)
        oof_xgb[val_idx] = clf.predict(X_data[val_idx])
        scores.append(mean_absolute_error(Y_data[val_idx], oof_xgb[val_idx]))
        print(scores[-1])
    print("xgboost score: {:<8.8f}".format(mean_absolute_error(oof_xgb, Y_data)))
    return clf

def build_model_lgb(params, X_data,Y_data, n_splits=5, random_state=2021):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_lgb = np.zeros(len(X_data))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_data[trn_idx], Y_data[trn_idx])
        val_data = lgb.Dataset(X_data[val_idx], Y_data[val_idx])
        num_round = 100000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=300,
                        early_stopping_rounds=300, feval = myFeval)
        oof_lgb[val_idx] = clf.predict(X_data[val_idx], num_iteration=clf.best_iteration)

    print("lightgbm score: {:<8.8f}".format(mean_absolute_error(oof_lgb, Y_data)))
    return clf

def build_model_lr(x_train,y_train):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x_train,y_train)
    return reg_model

if __name__ == '__main__':
    pass
