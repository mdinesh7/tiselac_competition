# coding: utf-8

import pandas as pd
import numpy as np

def read_data(filename):
    filepath = './../../data/raw/' + filename
    return pd.read_csv(filepath, sep=',', header=None)

def read_test(filename):
    filepath = './../../data/raw/' + filename
    return pd.read_csv(filepath, sep='\t', header=None)

def get_noncollinear(data):
    non_multi_coll = ['col_1','col_18','col_76','col_110','col_128','col_210','col_228']
    return data[non_multi_coll]

def extract_features(data):
    cols = ['col_1', 'col_18', 'col_76', 'col_110', 'col_128', 'col_210', 'col_228']
    for c in cols:
        colname_mean = 'coord1_' + c + '_mean'
        colname_std = 'coord1_' + c + '_std'
        colname_median = 'coord1_' + c + '_median'
        temp = data.groupby('spatial_coord_1')[c].agg({colname_mean: np.mean, colname_std: np.std, colname_median: np.median}).reset_index()
        data = pd.merge(data, temp, on='spatial_coord_1', how='left')

    data['coord_diff_1'] = data['spatial_coord_1'] - data['spatial_coord_2']
    data['coord_diff_2'] = data['spatial_coord_2'] - data['spatial_coord_1']
    data['coords_combined'] = (data['spatial_coord_1'].apply(str) + data['spatial_coord_2'].apply(str)).apply(int)
    return data

train = read_data("training.txt")
train_coords = read_data("coord_training.txt")
train_class = read_data("training_class.txt")
test = read_test('test.txt')
test_coords = read_data('coord_test.txt')
train_cols = train.columns.values.tolist()
test_cols = test.columns.values.tolist()

train.columns = ['col_' + str(col) for col in train_cols]
test.columns = ['col_' + str(col) for col in test_cols]

# filter non collinear columns
train = get_noncollinear(train)
test = get_noncollinear(test)

train_coords.columns = ['spatial_coord_1','spatial_coord_2']
train_class.columns = ['class']
train_df = pd.concat([train, train_coords, train_class], axis=1)
train_df['class'] = train_df['class'].apply(np.int32)

test_coords.columns = ['spatial_coord_1','spatial_coord_2']

test_df = pd.concat([test, test_coords], axis=1)

# Remove outliers
train_df = train_df[(train_df['col_1'] < 121) & (train_df['col_18'] > -963) & (train_df['col_18'] < 573) & (train_df['col_76'] < 190) & (train_df['col_110'] < 106) & (train_df['col_128'] < 625) & (train_df['col_210'] < 119) & (train_df['col_228'] < 634) & (train_df['col_228'] > -862)]

# Extract features
ntrain = train_df.shape[0]
train_test = pd.concat((train_df, test_df)).reset_index(drop=True)
train_test = extract_features(train_test).fillna(-1)
train_df = train_test.iloc[:ntrain, :]
test_df = train_test.iloc[ntrain:,:]

test_df.drop('class', axis=1, inplace=True)

# save data to data/processed folder
train_df.to_csv('./../../data/processed/train_features.csv', index=False)
test_df.to_csv('./../../data/processed/test_features.csv', index=False)
