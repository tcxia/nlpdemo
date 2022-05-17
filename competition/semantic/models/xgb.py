# -*- coding: utf-8 -*-
'''
# Created on 2021/03/26 11:14:38
# @filename: xgb.py
# @author: tcxia
'''


import xgboost as xgb
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df_train = pd.read_table(
    "/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
    names=['q1', 'q2', 'label']).fillna("0")
df_test = pd.read_table(
    '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
    names=['q1', 'q2']).fillna("0")


label = df_train['label'].values
df = pd.concat([df_train, df_test], ignore_index=True)
df['text'] = df['q1'] + " " + df['q2']

tfidf = TfidfVectorizer(ngram_range=(1, 5))
tfidf_feature = tfidf.fit_transform(df['text'])
svd_feature = TruncatedSVD(n_components=100).fit_transform(tfidf_feature)
# print(svd_feature.shape)

train_df = svd_feature[:-len(df_test)]
# print(train_df.shape)
# print(df_train['label'].values.shape)


test_df = svd_feature[-len(df_test):]


dtrain = xgb.DMatrix(train_df, label=df_train['label'].values)
dtest = xgb.DMatrix(test_df)

params = {
    'booster': 'gbtree',  #  tree-based models
    'objective': 'binary:logistic',
    'eta': 0.1,  # Same to learning rate
    'gamma': 0,  # Similar to min_impurity_decrease in GBDT
    'alpha': 0,  # L1 regularization term on weight (analogous to Lasso regression)
    'lambda': 2,  # L2 regularization term on weights (analogous to Ridge regression)
    'max_depth': 3,  # Same as the max_depth of GBDT
    'subsample': 1,  # Same as the subsample of GBDT
    'colsample_bytree': 1,  # Similar to max_features in GBM
    'min_child_weight': 1,  # minimum sum of instance weight (Hessian) needed in a child
    'nthread': 1,
}

num_pround = 10

bst = xgb.train(params, dtrain, num_pround)

ypred = bst.predict(dtest)
print(type(ypred))

# pd.DataFrame(ypred).to_csv(
#     '/data/nlp_dataset/oppo_breeno_round1_data/result_xgb.csv', index=False, header=False)
