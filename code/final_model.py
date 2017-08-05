# coding: utf-8


import pandas as pd
import numpy as np
from scipy.stats import boxcox

import xgboost as xgb
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

def read_data(filename):
    filepath = './../../data/processed/' + filename
    return pd.read_csv(filepath, sep=',')

# read data
train = read_data('train_features.csv')
test = read_data('test_features.csv')

# filter required columns
train_cols = ['col_1', 'col_18', 'col_76', 'col_110', 'col_128', 'col_210', 'col_228', 'spatial_coord_1', 'spatial_coord_2', 'class', 'coord1_col_1_std','coord_diff_1','coord_diff_2','coords_combined']
test_cols = ['col_1', 'col_18', 'col_76', 'col_110', 'col_128', 'col_210', 'col_228', 'spatial_coord_1', 'spatial_coord_2', 'coord1_col_1_std','coord_diff_1','coord_diff_2','coords_combined']
train_df = train[train_cols]
test_df = test[test_cols]

train_X = train_df.drop('class', axis=1).fillna(-1)
train_y = train_df['class']

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
train_test = pd.concat((train_X, test_df)).reset_index(drop=True)

def boxcox_fit(series):
    series_min = np.min(series)-1
    series_boxcox,_lambda = boxcox(series-series_min)
    return series_boxcox#,_lambda,y_min

# apply box cox transformation
train_cols = train_test.columns.values.tolist()
for col in train_cols:
    skew = train_test[col].skew()
    if abs(skew) > 0.25:
        train_test[col] = boxcox_fit(train_test[col].values)

train_X = train_test.iloc[:ntrain, :]
test_X = train_test.iloc[ntrain:,:]


scaler = StandardScaler()
# Transform train_X
train_X_columns = train_X.columns.values.tolist()
train_X = pd.DataFrame(scaler.fit_transform(train_X))
train_X.columns = train_X_columns

# Transform test_X
test_X_columns = test_X.columns.values.tolist()
test_X = pd.DataFrame(scaler.transform(test_X))
test_X.columns = test_X_columns



def xgb_model(tr_X, tr_y, te_X=None, te_y=None, seed_val=8, num_rounds=5000):
    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.05 #0.1
    param['gamma'] = 0.5
    param['max_depth'] = 4
    param['silent'] = 1
    param['num_class'] = 10
    param['min_child_weight'] = 1
    param['eval_metric'] ='mlogloss'
    param['subsample'] = 0.75
#     param['colsample_bytree'] = 0.4
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())


    plst = list(param.items())
    xgtrain = xgb.DMatrix(tr_X, label=tr_y)
    xgtest = xgb.DMatrix(te_X)
    model = xgb.train(plst, xgtrain, num_rounds)
    print("Completed training")
    y_pred = model.predict(xgtest)
    return model, y_pred

# run model
model, preds = xgb_model(train_X, train_y, te_X=test_X, num_rounds=best_iteration, model_type="model")
preds = [float(p) for p in preds]

# write predictions to text file
output = open('./../../data/processed/testing_class.txt', 'w')
for p in preds:
    output.write("%s\n" % p)

output.close()
