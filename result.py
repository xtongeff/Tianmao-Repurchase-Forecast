import os
os.getcwd()
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gc
import warnings


from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

test_data = pd.read_csv('./data/data_format1/test_format1.csv')
train_data = pd.read_csv('./data/data_format1/train_format1.csv')

user_info = pd.read_csv('./data/data_format1/user_info_format1.csv')
user_log = pd.read_csv('./data/data_format1/user_log_format1.csv')


# # reduce memory
# def reduce_mem_usage(df, verbose=True):
#     start_mem = df.memory_usage().sum() / 1024 ** 2
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#
#     for col in df.columns:
#         col_type = df[col].dtypes
#         if col_type in numerics:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#
#     end_mem = df.memory_usage().sum() / 1024 ** 2
#     print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
#     return df
#
# train_data = reduce_mem_usage(train_data)
# test_data = reduce_mem_usage(test_data)
#
# user_info = reduce_mem_usage(user_info)
# user_log = reduce_mem_usage(user_log)
#
# # 数据探索
# print(train_data.head())
# print(test_data.head())
# print(user_info.head())
# print(user_log.head())
#
# # 缺失值查看
# print(train_data.isnull().sum())
# print(test_data.isnull().sum())
# print(user_info.isna().sum()/user_info.shape[0])
# print(user_log.isna().sum()/user_log.shape[0])
#
# # 重复值查看
# print(train_data.duplicated().sum())
# print(test_data.duplicated().sum())
# print(user_info.duplicated().sum())
# print(user_log.duplicated().sum())
#
# # 可视化数据
# # 正负样本比例分布
# label_gp = train_data.groupby('label')['user_id'].count()
# print('正负样本的数量：\n', label_gp)
# _,axe = plt.subplots(1,2,figsize=(12,6))
# train_data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, explode=[0, 0.1], ax=axe[0])
# # 修改此处，移除多余的参数传递
# sns.countplot(x='label', data=train_data, ax=axe[1])
# plt.show()
#
# # 购买次数前5的店铺
# print('选取top5店铺\n店铺\t购买次数')
# print(train_data['merchant_id'].value_counts().head(5))
# train_data_merchant = train_data.copy()
# train_data_merchant['TOP5'] = train_data_merchant['merchant_id'].map(lambda x: 1 if x in [4044, 3828, 4173, 1102, 4976] else 0)
# train_data_merchant = train_data_merchant[train_data_merchant['TOP5'] == 1]
#
# plt.figure(figsize=(8, 6))
# plt.title('Merchant VS Label')
# ax = sns.countplot(x='merchant_id', hue='label', data=train_data_merchant)
# for p in ax.patches:
#     height = p.get_height()
#     ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
#                 ha='center', va='center', xytext=(0, 5), textcoords='offset points')
#
# # 复购概率分布
# user_repeat_buy = [rate for rate in train_data.groupby(['user_id'])['label'].mean() if 0 < rate <= 1]
#
# plt.figure(figsize=(8, 6))
#
# ax = plt.subplot(1, 2, 1)
# # 直方图
# sns.histplot(user_repeat_buy, kde=True)
# ax = plt.subplot(1, 2, 2)
# # qq图
# res = stats.probplot(user_repeat_buy, plot=plt)
#
# train_data_user_info = train_data.merge(user_info, on=['user_id'], how='left')
# plt.figure(figsize=(8, 8))
# plt.title('Gender VS Label')
# ax = sns.countplot(x='gender', hue='label', data=train_data_user_info)
# for p in ax.patches:
#     height = p.get_height()
#     ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
#                 ha='center', va='center', xytext=(0, 5), textcoords='offset points')
#
# plt.show()
#
# # 性别与复购分布
# repeat_buy_gender = train_data_user_info.groupby(['gender'])['label'].mean().tolist()
#
# plt.figure(figsize=(8, 4))
#
# ax = plt.subplot(1, 2, 1)
# # 使用 sns.histplot 替代已弃用的 sns.distplot
# sns.histplot(repeat_buy_gender, kde=True)
# ax = plt.subplot(1, 2, 2)
# res = stats.probplot(repeat_buy_gender, plot=plt)
#
# plt.figure(figsize=(8, 8))
# plt.title('Age VS Label')
# ax = sns.countplot(x='age_range', hue='label', data=train_data_user_info)
# # 为柱子添加高度标注
# for p in ax.patches:
#     height = p.get_height()
#     ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
#                 ha='center', va='center', xytext=(0, 5), textcoords='offset points')
#
# # 年龄与复购分布
# repeat_buy_age = train_data_user_info.groupby(['age_range'])['label'].mean().tolist()
#
# plt.figure(figsize=(8, 4))
#
# ax = plt.subplot(1, 2, 1)
# # 使用 sns.histplot 替代已弃用的 sns.distplot
# sns.histplot(repeat_buy_age, kde=True)
# ax = plt.subplot(1, 2, 2)
# res = stats.probplot(repeat_buy_age, plot=plt)
#
# # 训练集不同时间段的复购用户
# all_data_1 = user_log.merge(train_data, on=['user_id'], how='left')
# print(all_data_1[all_data_1['label'].notnull()].head())
#
# all_data_2 = all_data_1[all_data_1['label'].notnull()]
# all_data_2_sum = all_data_2.groupby(['time_stamp'])['label'].sum().reset_index()
# print(all_data_2_sum.head())
#
# all_data_2_sum['time_stamp'] = all_data_2_sum['time_stamp'].astype(str)
# all_data_2_sum['label'] = all_data_2_sum['label'].astype(int)
#
# # 提取月份信息优化
# all_data_2_sum['month'] = all_data_2_sum['time_stamp'].apply(lambda x: x[0] if len(x) == 3 else x[:2]).astype(int)
#
# plt.figure(figsize=(20, 8))
# c = 5
# for i in range(1, 8):
#     plt.subplot(3, 3, i)
#     b = all_data_2_sum[all_data_2_sum["month"] == c]
#     plt.plot(b['time_stamp'], b['label'], linewidth=1, color="orange", marker="o", label="Mean value")
#     plt.title(f'Month {c}')
#     c += 1
#
# c = all_data_2_sum.groupby(['month'])['label'].sum().reset_index()
# plt.figure(figsize=(8, 4))
# plt.plot(c['month'], c['label'], linewidth=1, color="orange", marker="o", label="Total by month")
# plt.title('Total label sum by month')
# plt.xlabel('Month')
# plt.ylabel('Total label sum')
# plt.legend()
#
# # 显示所有图片
# plt.show()
#
# # 特征工程
# del test_data['prob']
# train_data['target'] = 1
# test_data['target']=-1
# all_data = train_data.append(test_data)
# all_data = all_data.merge(user_info,on=['user_id'],how='left')
# del train_data, test_data, user_info
# gc.collect()  # Python 中用于手动触发垃圾回收机制的函数。
#
# print(all_data.head())
#
# all_data.dropna(subset=['age_range','gender'],inplace=True)
# print(all_data.isnull().sum())
#
# # 用户店铺数.“用户店铺数” 指的是每个用户在平台上关联的不同店铺的数量。seller_id代表不同的店铺.
# sell= user_log.groupby(['user_id'])['seller_id'].count().reset_index()
#
# all_data = all_data.merge(sell,on=['user_id'],how='inner')
# all_data.rename(columns={'seller_id': 'sell_sum'},inplace=True)
# print(all_data.head())
#
#
# def nunique_k(data, sigle_name, new_name_1):  # 主要功能是对 user_log 数据集中的数据按照 user_id 进行分组，
#     # 统计每个用户在指定列中不同值的数量，然后将统计结果合并到输入的数据集中，并对合并后的列进行重命名。
#     #  nunique 方法统计每个用户在 sigle_name 列中不同值的数量。
#     data1 = user_log.groupby(['user_id'])[sigle_name].nunique().reset_index()
#
#     data_union = data.merge(data1, on=['user_id'], how='inner')
#     data_union.rename(columns={sigle_name: new_name_1}, inplace=True)
#     return data_union
#
# # 不同店铺个数
# all_data=nunique_k(all_data,'seller_id','seller_id_unique')
#
# # 不同品类个数
# all_data=nunique_k(all_data,'cat_id','cat_id_unique')
#
# # 不同品牌个数
# all_data=nunique_k(all_data,'brand_id','brand_id_unique')
#
# # 不同商品个数
# all_data=nunique_k(all_data,'item_id','item_id_unique')
#
# # 活跃天数
# all_data=nunique_k(all_data,'time_stamp','time_stamp_unique')
#
# # 不用行为种数
# all_data=nunique_k(all_data,'action_type','action_type_unique')
#
# print(all_data.head())
#
# print(user_log.head())
#
# # 活跃天数方差
# std=user_log.groupby(['user_id'])['time_stamp'].std().reset_index()
# all_data = all_data.merge(std,on=['user_id'],how='inner')
# all_data.rename(columns={'time_stamp':'time_stamp_std'},inplace=True)
# print(all_data.head())
#
# def most_love(data_1,most_name,new_name_2):  # 找出每个用户在 user_log 数据集中对某一事物（由 most_name 指定的列）的 “最爱”，
#     # 也就是出现次数最多的那个元素，然后将这些结果合并到输入的数据集中，并对合并后的列进行重命名。
#     # Counter(x) 是 collections 模块中的一个工具，它会统计 x 中每个元素出现的次数，返回一个字典，键为元素，值为该元素出现的次数。
#     # most_common(1) 会从 Counter 对象中找出出现次数最多的前 1 个元素，返回一个包含元素及其次数的元组列表，这里只取出现次数最多的那个元素。
#     # [0][0] 从 most_common(1) 返回的列表中取出元组，再从元组中取出元素本身。
#     data2=user_log.groupby(['user_id'])[most_name].apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
#     data_union_1=data_1.merge(data2,on=['user_id'],how='inner')
#     data_union_1.rename(columns={most_name:new_name_2},inplace=True)
#     return data_union_1
#
# # 用户最喜欢的店铺
# all_data=most_love(all_data,'seller_id','sell_id_most')
#
# # 最喜欢的类目
# all_data=most_love(all_data,'cat_id','cat_id_most')
#
# # 最喜欢的品牌
# all_data=most_love(all_data,'brand_id','brand_id_most')
#
# # 最常见的行为动作
# all_data=most_love(all_data,'action_type','action_type_most')
#
# print(all_data.head())
#
# def most_love_cnt(data_1,most_name,new_name_2):
#     data2=user_log.groupby(['user_id'])[most_name].apply(lambda x: Counter(x).most_common(1)[0][1]).reset_index()
#     data_union_1=data_1.merge(data2,on=['user_id'],how='inner')
#     data_union_1.rename(columns={most_name:new_name_2},inplace=True)
#     return data_union_1
#
# # 用户最喜欢的店铺 行为次数
# all_data=most_love_cnt(all_data,'seller_id','seller_id_most_cnt')
#
# # 最喜欢的类目 行为次数
# all_data=most_love_cnt(all_data,'cat_id','cat_id_most_cnt')
#
# # 最喜欢的品牌 行为次数
# all_data=most_love_cnt(all_data,'brand_id','brand_id_most_cnt')
#
# # 最常见的行为动作 行为次数
# all_data=most_love_cnt(all_data,'action_type','action_type_most_cnt')
#
# print(all_data.head())
#
#
# user_id_union=list(set(all_data['user_id']))
#
# def action_type_select(data,num,new_name):
#     d=user_log.groupby(['user_id'])['action_type'].apply(lambda x: Counter(x))
#     e=dict(d)  # 字典的键是 user_id，值是对应 user_id 的 action_type 统计结果。
#     k=[]
#     for i in user_id_union:
#         try:
#             k.append(e[(i,num)])
#         except KeyError:
#             k.append(0)
#     # 包含两列：user_id 和新列（列名由 new_name 指定），其中新列存储了每个用户特定 action_type 的出现次数。
#     data3=pd.DataFrame({'user_id':user_id_union,new_name:k})
#     data_union_2=data.merge(data3,on=['user_id'],how='inner')
#     return data_union_2
#
# # 点击次数
# all_data=action_type_select(all_data,0,'action_type_sum_0')
#
# # 加购次数
# all_data=action_type_select(all_data,1,'action_type_sum_1')
#
# # 购买次数
# all_data=action_type_select(all_data,2,'action_type_sum_2')
#
# # 收藏次数
# all_data=action_type_select(all_data,3,'action_type_sum_2')
#
# print(all_data.head())
#
# train = all_data[all_data['target'] == 1].reset_index(drop = True)
# test = all_data[all_data['target'] == -1].reset_index(drop = True)
#
# train.drop(['target'],axis=1,inplace=True)
# test.drop(['target'],axis=1,inplace=True)
#
# train.fillna(0,inplace=True)
#
# test.drop(['label'],axis=1,inplace=True)
#
# test.fillna(0,inplace=True)  # 将 test 数据集中所有的缺失值（NaN）填充为 0。
#
# print(test.head())
#
# train.to_csv('train_all_k.csv',header=True,index=False)
# test.to_csv('test_all_k.csv',header=True,index=False)
#
# print(pd.read_csv('train_all_k.csv').head())


# 建模
train_data=pd.read_csv('train_all_k.csv')
print(train_data.head())

test_data=pd.read_csv('test_all_k.csv')
print(test_data.head())

print(train_data['label'].value_counts())

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from functools import partial

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn import  metrics
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, log_loss,  auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def model_clf(model):
    model.fit(X_train, y_train)
    y_train_pred = model.predict_proba(X_train)
    y_train_pred_pos = y_train_pred[:,1]

    y_test_pred = model.predict_proba(X_test)
    y_test_pred_pos = y_test_pred[:,1]

    auc_train = roc_auc_score(y_train, y_train_pred_pos)#AUC评分
    auc_test = roc_auc_score(y_test, y_test_pred_pos)

    print(f"Train AUC Score {auc_train}")
    print(f"Test AUC Score {auc_test}")

    fpr, tpr, _ = roc_curve(y_test,y_test_pred_pos)#绘制ROC曲线
    return fpr,tpr

target = train_data['label']
train_data_1 = train_data.drop(['label'],axis=1)

# # 逻辑回归
# stdScaler = StandardScaler()
#
# X = stdScaler.fit_transform(train_data_1)
# X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=0)
# # Split the data into a training set and a test set
#
# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
# model_clf(clf)
#
# # knn 模型
# stdScaler = StandardScaler()
# X = stdScaler.fit_transform(train_data_1)
# X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=5)
# model_clf(clf)
#
# # 决策树模型
# X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, random_state=0)
# clf = tree.DecisionTreeClassifier()
# model_clf(clf)
#
# # bagging模型
# X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, random_state=0)
# clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
# model_clf(clf)
#
# # 随机森林
# X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, random_state=0)
# clf = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=12, random_state=0)
# model_clf(clf)
#
# # Adaboost
# X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, random_state=0)
# clf = AdaBoostClassifier(n_estimators=10)
# model_clf(clf)
#
# # GBDT
# X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, random_state=0)
# clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)
# model_clf(clf)

# # lgb
# X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, test_size=0.4, random_state=0)
# X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
# clf = lgb
# train_matrix = clf.Dataset(X_train, label=y_train)
# test_matrix = clf.Dataset(X_test, label=y_test)
# params = {
#           'boosting_type': 'gbdt',
#           #'boosting_type': 'dart',
#           'objective': 'multiclass',
#           'metric': 'multi_logloss',
#           'min_child_weight': 1.5,
#           'num_leaves': 2**5,
#           'lambda_l2': 10,
#           'subsample': 0.7,
#           'colsample_bytree': 0.7,
#           'colsample_bylevel': 0.7,
#           'learning_rate': 0.03,
#           'tree_method': 'exact',
#           'seed': 2017,
#           "num_class": 2,
#           'silent': True,
#           }
# num_round = 10000
# # early_stopping_rounds = 100
# callbacks = [log_evaluation(period=10), early_stopping(stopping_rounds=100)]
# model = lgb.train(params,
#                   train_matrix,
#                   num_boost_round=num_round,
#                   valid_sets=[test_matrix],
#                   callbacks=callbacks
#                   )
#
# y_train_pred = model.predict(X_train,num_iteration=model.best_iteration)
# y_train_pred_pos = y_train_pred[:,1]
#
# y_test_pred = model.predict(X_valid,num_iteration=model.best_iteration)
# y_test_pred_pos = y_test_pred[:,1]
#
# auc_train = roc_auc_score(y_train, y_train_pred_pos)#AUC评分
#
# auc_test = roc_auc_score(y_valid, y_test_pred_pos)
#
# print(f"Train AUC Score {auc_train}")
# print(f"Test AUC Score {auc_test}")
#
# fpr, tpr, _ = roc_curve(y_valid,y_test_pred_pos)#绘制ROC曲线
#
#
def plot_auc_curve(fpr, tpr, auc):
    plt.figure(figsize = (16,6))
    plt.plot(fpr,tpr,'b+',linestyle = '-')
    plt.fill_between(fpr, tpr, alpha = 0.5)
    plt.ylabel('True Postive Rate')
    plt.xlabel('False Postive Rate')
    plt.title(f'ROC Curve Having AUC = {auc}')
    plt.show()
#
# plot_auc_curve(fpr, tpr, auc_test)
#
#
# # xgb
X_train, X_test, y_train, y_test = train_test_split(train_data_1, target, test_size=0.4, random_state=0)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

clf = xgb
train_matrix = clf.DMatrix(X_train, label=y_train, missing=-1)
test_matrix = clf.DMatrix(X_test, label=y_test, missing=-1)
z = clf.DMatrix(X_valid, label=y_valid, missing=-1)
params = {'booster': 'gbtree',
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'gamma': 1,
          'min_child_weight': 1.5,
          'max_depth': 5,
          'lambda': 100,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.03,
          'tree_method': 'exact',
          'seed': 2017,
          "num_class": 2
          }
score_dict={}
num_round = 10000
early_stopping_rounds = 100
watchlist = [(train_matrix, 'train'),
             (test_matrix, 'eval')
             ]
model = clf.train(params,
                  train_matrix,
                  num_boost_round=num_round,#训练的轮数，即生成的树的数量
                  evals=watchlist,#训练时用于评估模型的数据集合，用户可以通过验证集来评估模型好坏
                  early_stopping_rounds=early_stopping_rounds,
                  #激活早停止，当验证的错误率early_stopping_rounds轮未下降，则停止训练。
                  #如果发生了早停止，则模型会提供三个额外的字段：bst.best_score, bst.best_iteration 和 bst.best_ntree_limit
                  evals_result =score_dict
                  )
y_train_pred = model.predict(train_matrix, iteration_range=(0, model.best_iteration))#ntree_limit  用于预测的树的数量
y_train_pred_pos = y_train_pred[:,1]

y_test_pred = model.predict(z, iteration_range=(0, model.best_iteration))
y_test_pred_pos = y_test_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)#AUC评分

auc_test = roc_auc_score(y_valid, y_test_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")

fpr, tpr, _ = roc_curve(y_valid,y_test_pred_pos)#绘制ROC曲线

plot_auc_curve(fpr, tpr, auc_test)


plt.figure(figsize = (15,8))
plt.plot(score_dict['train']['mlogloss'], 'r-+', label = 'Training Loss')
plt.plot(score_dict['eval']['mlogloss'], 'b-', label = 'Test Loss')
plt.show()

# xgb调参
features_columns = [col for col in train_data.columns if col not in ['user_id','label']]
train_data_1 = train_data[features_columns]
test_data_1 = test_data[features_columns]
target = train_data['label']

def model(train_1,target_1):
    X_train, X_val, y_train, y_val = train_test_split(train_1, target_1, test_size = 0.2 ,random_state = 42)

    clf = XGBClassifier()

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict_proba(X_train)
    y_train_pred_pos = y_train_pred[:,1]

    y_val_pred = clf.predict_proba(X_val)
    y_val_pred_pos = y_val_pred[:,1]

    auc_train = roc_auc_score(y_train, y_train_pred_pos)
    auc_test = roc_auc_score(y_val, y_val_pred_pos)

    print(f"Train AUC Score {auc_train}")
    print(f"Test AUC Score {auc_test}")

    fpr, tpr, _ = roc_curve(y_val, y_val_pred_pos)
    return fpr,tpr,clf,auc_test

fpr_1,tpr_1,clf_1,auc_test_1=model(train_data_1,target)#0.6153997

def plot_auc_curve(fpr, tpr, auc):
    plt.figure(figsize = (16,6))
    plt.plot(fpr,tpr,'b+',linestyle = '-')
    plt.fill_between(fpr, tpr, alpha = 0.5)
    plt.ylabel('True Postive Rate')
    plt.xlabel('False Postive Rate')
    plt.title(f'ROC Curve Having AUC = {auc}')
    plt.show()

plot_auc_curve(fpr_1, tpr_1, auc_test_1)


def plot_learning_cuve(model, X, Y, num):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=11)
    train_loss, test_loss = [], []

    for m in range(num, len(x_train), num):
        model.fit(x_train.iloc[:m, :], y_train[:m])
        y_train_prob_pred = model.predict_proba(x_train.iloc[:m, :])
        train_loss.append(log_loss(y_train[:m], y_train_prob_pred))

        y_test_prob_pred = model.predict_proba(x_test)
        test_loss.append(log_loss(y_test, y_test_prob_pred))

    plt.figure(figsize=(15, 8))
    plt.plot(train_loss, 'r-+', label='Training Loss')
    plt.plot(test_loss, 'b-', label='Test Loss')
    plt.xlabel('Number Of Batches')
    plt.ylabel('Log-Loss')
    plt.legend(loc='best')
    plt.show()

plot_learning_cuve(XGBClassifier(), train_data_1, target,5000)

smote = SMOTE(random_state = 402)
X_smote, Y_smote = smote.fit_resample(train_data_1,target)
sns.countplot(Y_smote, edgecolor = 'black')
plt.title("Count Plot of Y_smote after First SMOTE")
plt.show()  # 显示第一次过采样后的计数图

fpr_2,tpr_2,clf_2,auc_test_2=model(X_smote,Y_smote)#0.6022299

plot_learning_cuve(XGBClassifier(),X_smote, Y_smote,9600)
# # 代码的目的是对少数类样本进行过采样，以平衡数据集中不同类别的样本数量。SMOTE 算法通过合成新的少数类样本来增加少数类样本的数量，以此改善模型的性能。
# smote = SMOTE(random_state = 446)
# X_smote1, Y_smote1 = smote.fit_resample(train_data_1,target)
#
# # 合并两次过采样的数据
# X_final = pd.concat([X_smote, X_smote1], axis = 0).reset_index(drop = True)
# Y_final = pd.concat([Y_smote, Y_smote1], axis = 0).reset_index(drop = True)
#
# # 报错MemoryError了。
# # sns.countplot(Y_final, edgecolor = 'black')  # 这行代码使用了 seaborn 库中的 countplot 函数来绘制分类变量的计数图
# # plt.title("Count Plot of Y_final after Combining SMOTE Results")
# # plt.show()  # 显示合并后的计数图
#
# fpr_5,tpr_5,clf_5,auc_test_5=model(X_final,Y_final)#0.6027413

# 特征优化
# train：这是一个包含所有特征的训练数据集，通常是一个二维数组或矩阵，每一行代表一个样本，每一列代表一个特征。
# train_sel：这是经过特征选择后的训练数据集，同样是二维数组或矩阵，不过特征数量可能比 train 少。
# target：这是对应的目标标签，通常是一个一维数组，每个元素表示对应样本的类别。
def feature_selection(train, train_sel, target):  # 对比在使用全部特征和经过特征选择后的特征进行训练时，模型的交叉验证准确率。
    clf = XGBClassifier()
    # cross_val_score 是 sklearn 库中的一个函数，用于进行交叉验证。cv = 5 表示使用 5 折交叉验证，即将数据集分成 5 份，
    # 依次将其中 4 份作为训练集，1 份作为测试集进行训练和评估，重复 5 次，最终得到 5 个评估分数。
    scores = cross_val_score(clf, train, target, cv=5)

    scores_sel = cross_val_score(clf, train_sel, target, cv=5)
    # scores.mean() 计算 5 次交叉验证分数的平均值，scores.std() * 2 计算 5 次交叉验证分数的标准差的两倍，用于表示估计的误差范围。
    print("No Select Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Features Select Accuracy: %0.2f (+/- %0.2f)" % (scores_sel.mean(), scores_sel.std() * 2))
    return scores.mean(), scores_sel.mean()


def select(train, goal):
    a_score = []
    b_score = []
    feature_names = train.columns.tolist()  # 新增行：获取特征名称列表
    selected_features_dict = {}  # 新增行：存储每次选择的特征

    for i in range(2, train.shape[1] + 1):
        sel = SelectKBest(mutual_info_classif, k=i)
        sel = sel.fit(train, goal)
        train_sel = sel.transform(train)

        # 新增代码块：获取并打印被选中的特征
        selected_indices = sel.get_support(indices=True)
        selected_features = [feature_names[idx] for idx in selected_indices]
        selected_features_dict[i] = selected_features

        print(f'\n特征数量: {i}')
        print('训练数据未特征筛选维度', train.shape)
        print('训练数据特征筛选维度后', train_sel.shape)
        print(f'选中的特征: {selected_features}')

        mean_train, mean_test = feature_selection(train, train_sel, goal)
        a_score.append(mean_train)
        b_score.append(mean_test)

    x = list(range(2, train.shape[1] + 1))

    plt.figure(figsize=(12, 6))  # 修改行：调整图表大小
    plt.plot(x, a_score, marker='o', markersize=5, label='No Select')  # 修改行：增加标记大小
    plt.plot(x, b_score, marker='o', markersize=5, label='Features Select')  # 修改行：增加标记大小

    # 新增代码块：在曲线上标记每个点的特征数量和性能
    for i, (x_val, a_val, b_val) in enumerate(zip(x, a_score, b_score)):
        plt.annotate(f'{i + 2} features\n{a_val:.3f}',
                     (x_val, a_val),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)
        plt.annotate(f'{b_val:.3f}',
                     (x_val, b_val),
                     textcoords="offset points",
                     xytext=(0, -15),
                     ha='center',
                     fontsize=8)

    plt.xticks(x)
    plt.xlabel('Number of Features')  # 新增行：添加x轴标签
    plt.ylabel('Cross-Validation Accuracy')  # 新增行：添加y轴标签
    plt.title('Feature Selection Performance')  # 新增行：添加标题
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # 新增行：添加网格线
    plt.tight_layout()  # 新增行：优化布局
    plt.show()

    return a_score, b_score, selected_features_dict  # 修改行：增加返回选中的特征


    # for i in range(2, train.shape[1] + 1):  # 使用 for 循环，从选择 2 个特征开始，逐步增加到选择所有特征（train.shape[1] 表示训练数据的特征数量）。
    #     sel = SelectKBest(mutual_info_classif, k=i)
    #     sel = sel.fit(train, goal)
    #     train_sel = sel.transform(train)
    #
    #     print('训练数据未特征筛选维度', train.shape)
    #     print('训练数据特征筛选维度后', train_sel.shape)
    #     mean_train, mean_test = feature_selection(train, train_sel, goal)
    #     a_score.append(mean_train)
    #     b_score.append(mean_test)
    # x = list(range(2, train.shape[1] + 1))
    # # 创建一个列表 x，包含从 2 到训练数据特征数量的所有整数，作为折线图的 x 轴数据。
    # # 使用 plt.plot 函数分别绘制未进行特征选择和经过特征选择后的交叉验证平均准确率随特征选择数量变化的折线图，并添加数据点，设置点的大小为 3。
    # plt.plot(x, a_score, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(x, b_score, marker='o', markersize=3)
    # plt.xticks(x)
    # plt.show()
    # return a_score, b_score

a_score,b_score=select(train_data_1,target)

x=list(range(2, train_data_1.shape[1]+1))
plt.plot(x, a_score, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, b_score, marker='o', markersize=3)
plt.xticks(x)
plt.show()

def select_fin(i,train_1,target_1):  # 使用 SelectKBest 方法进行特征选择。SelectKBest 是 sklearn 库中的一个类，用于根据指定的评分函数选择 k 个最佳特征。
    sel = SelectKBest(mutual_info_classif, k=i)
    sel = sel.fit(train_1, target_1)
    train_sel = sel.transform(train_1)
    test_sel = sel.transform(test_data_1)
    return train_sel,test_sel

train_sel,test_sel=select_fin(4,train_data_1,target)  # 选择 4 个最佳特征，并将处理后的训练集和测试集分别赋值给 train_sel 和 test_sel。

fpr_5,tpr_5,clf_5,auc_test_5=model(train_sel,target)#k=2 0.6419733。
# 注释中给出了不同 k 值下的 AUC 测试结果，但实际上这里的 k 值均为 4，因为使用的是 select_fin(4, ...) 处理后的 train_sel。

fpr_6,tpr_6,clf_6,auc_test_6=model(train_sel,target)#k=3 0.6365505

fpr_7,tpr_7,clf_7,auc_test_7=model(train_sel,target)#k=4 0.6323718

a_score,b_score=select(X_smote,Y_smote)


train_sel,test_sel=select_fin(9,X_smote,Y_smote)

fpr_8,tpr_8,clf_8,auc_test_8=model(train_sel,Y_smote)#k=9  0.5940987

pred = clf_8.predict_proba(test_sel)  # 训练好的分类器模型 clf_8
# 这里创建了一个空列表 pred_list，用于存储每个测试样本属于正类（通常索引为 1 的类别）的概率。
# 通过 for 循环遍历 pred 数组的每一行，将每行中索引为 1 的元素（即正类概率）添加到 pred_list 中。
pred_list=[]
for i in range(len(pred)):
    pred_list.append(pred[i][1])
df_out = pd.DataFrame()
df_out['user_id'] = test_data['user_id'].astype(int)
df_out['merchant_id']=test_data['merchant_id'].astype(int)
df_out['predict_prob'] = pred_list
df_out.to_csv('prediction.csv',index=0)

# 源数据调参
def tun_parameters(train_x, train_y):  # 通过这个函数，确定树的个数
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                         colsample_bytree=0.8, objective='binary:logistic', scale_pos_weight=1, seed=27)
    modelfit(xgb1, train_x, train_y)


def modelfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # useTrainCV：布尔型参数，默认为 True，表示是否使用交叉验证来确定最优的树的数量。
    # cv_folds：整数类型，默认为 5，表示交叉验证的折数。
    if useTrainCV:
        xgb_param = alg.get_xgb_params()  # 参数
        xgtrain = xgb.DMatrix(X, label=y)  # 训练数据
        # 使用 xgb.cv 函数进行交叉验证，通过指定的参数进行 num_boost_round 轮迭代，使用 nfold 折交叉验证，
        # 以 auc 作为评估指标，并设置早停轮数为 early_stopping_rounds。
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        # num_boost_round:树的个数 nfold 几折交叉验证 early_stopping_rounds 早停值
        alg.set_params(n_estimators=cvresult.shape[0])
        # 输出树的个数
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds)
    # # Fit the algorithm on the data
    # alg.fit(X, y, eval_metric='auc')  # 使用更新后的参数对 XGBoost 分类器进行训练，以 auc 作为评估指标。

    # Predict training set:
    dtrain_predictions = alg.predict(X)  # 对训练数据进行预测，得到预测的类别标签。
    dtrain_predprob = alg.predict_proba(X)[:, 1]  # 对训练数据进行预测，得到属于正类的概率。

    # Print model report:
    print("\nModel Report")  # 打印模型的评估报告，包括准确率（Accuracy）和训练集的 AUC 分数（AUC Score (Train)）。
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
    # 获取 XGBoost 模型中每个特征的重要性得分，并将其存储在 pandas 的 Series 对象中，按重要性得分降序排序。
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    # bosster()改为get_bosster()
    feat_imp.plot(kind='bar', title='Feature Importances')  # 绘制特征重要性的柱状图。
    plt.ylabel('Feature Importance Score')
    plt.show()
    print('n_estimators=', cvresult.shape[0])

tun_parameters(train_data_1,target)



# 树的最大深度 最小叶子节点样本权重
param_test1 = {
  'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}
# GridSearchCV 是 sklearn 库中的一个类，用于进行网格搜索以找到最优的超参数组合。
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=333, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,\
 objective= 'binary:logistic', nthread=8,scale_pos_weight=1, seed=27),
 param_grid = param_test1,scoring='roc_auc',n_jobs=-1, cv=5)
gsearch1.fit(train_data_1,target)  # fit 方法会对 param_test1 中定义的所有超参数组合进行遍历，使用 5 折交叉验证对每个组合进行评估，并记录每个组合的性能。
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch3 = GridSearchCV(
    estimator=XGBClassifier(learning_rate=0.1, n_estimators=333, max_depth=5, min_child_weight=2, gamma=0,
                            subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=8,
                            scale_pos_weight=1, seed=27), param_grid=param_test3, scoring='roc_auc', n_jobs=-1,
     cv=5)
gsearch3.fit(train_data_1,target)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}

gsearch4 = GridSearchCV(
    estimator=XGBClassifier(learning_rate=0.1, n_estimators=333, max_depth=5, min_child_weight=2, gamma=0.0,
                            subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=8,
                            scale_pos_weight=1, seed=27), param_grid=param_test4, scoring='roc_auc', n_jobs=-1,
    cv=5)

gsearch4.fit(train_data_1, target)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)


param_test6 = {
 'reg_alpha':[1e-5,1e-4,1e-3, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=333, max_depth=5, min_child_weight=2,
    gamma=0.0, subsample=0.9, colsample_bytree=0.7, objective= 'binary:logistic', nthread=8,
    scale_pos_weight=1,seed=27), param_grid = param_test6, scoring='roc_auc',n_jobs=-1,cv=5)
gsearch6.fit(train_data_1,target)
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


X_train, X_val, y_train, y_val = train_test_split(train_data_1,target, test_size = 0.2 ,random_state = 42)
#0.6305415


clf = XGBClassifier(
    learning_rate =0.1,
    n_estimators=333,max_depth=5, min_child_weight=2,
    gamma=0.0, subsample=0.9, colsample_bytree=0.7, objective= 'binary:logistic', nthread=8,
    scale_pos_weight=1,seed=27,reg_alpha=1
                    )

clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)
y_train_pred_pos = y_train_pred[:,1]

y_val_pred = clf.predict_proba(X_val)
y_val_pred_pos = y_val_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)
auc_test = roc_auc_score(y_val, y_val_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")

fpr, tpr, _ = roc_curve(y_val, y_val_pred_pos)


pred = clf.predict_proba(test_data_1)
pred_list=[]
for i in range(len(pred)):
    pred_list.append(pred[i][1])
df_out = pd.DataFrame()
df_out['user_id'] = test_data['user_id'].astype(int)
df_out['merchant_id']=test_data['merchant_id'].astype(int)
df_out['predict_prob'] = pred_list
df_out.to_csv('prediction.csv',index=0)


# 特征选择后的模型调参
train_sel,test_sel=select_fin(2,train_data_1,target)#0.653329

tun_parameters(train_sel,target)

#树的最大深度 最小叶子节点样本权重
param_test1 = {
  'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,\
 objective= 'binary:logistic', nthread=8,scale_pos_weight=1, seed=27),
 param_grid = param_test1,scoring='roc_auc',n_jobs=-1, cv=5)
gsearch1.fit(train_sel,target)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch3 = GridSearchCV(
    estimator=XGBClassifier(learning_rate=0.1, n_estimators=2000, max_depth=8, min_child_weight=1, gamma=0,
                            subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=8,
                            scale_pos_weight=1, seed=27), param_grid=param_test3, scoring='roc_auc', n_jobs=-1,
     cv=5)
gsearch3.fit(train_sel,target)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}

gsearch4 = GridSearchCV(
    estimator=XGBClassifier(learning_rate=0.1, n_estimators=2000, max_depth=8, min_child_weight=1, gamma=0.2,
                            subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=8,
                            scale_pos_weight=1, seed=27), param_grid=param_test4, scoring='roc_auc', n_jobs=-1,
    cv=5)

gsearch4.fit(train_sel, target)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

param_test6 = {
 'reg_alpha':[1e-5,1e-4,1e-3, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=2000, max_depth=8, min_child_weight=1,
    gamma=0.2, subsample=0.9, colsample_bytree=0.6, objective= 'binary:logistic', nthread=8,
    scale_pos_weight=1,seed=27), param_grid = param_test6, scoring='roc_auc',n_jobs=-1,cv=5)
gsearch6.fit(train_sel,target)
print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)


X_train, X_val, y_train, y_val = train_test_split(train_sel,target, test_size = 0.2 ,random_state = 42)
#0.65


clf = XGBClassifier(
    learning_rate =0.1,
    n_estimators=2000,max_depth=8, min_child_weight=1,
    gamma=0.2, subsample=0.9, colsample_bytree=0.6, objective= 'binary:logistic', nthread=8,
    scale_pos_weight=1,seed=27,reg_alpha=0.01
                    )

clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)
y_train_pred_pos = y_train_pred[:,1]

y_val_pred = clf.predict_proba(X_val)
y_val_pred_pos = y_val_pred[:,1]

auc_train = roc_auc_score(y_train, y_train_pred_pos)
auc_test = roc_auc_score(y_val, y_val_pred_pos)

print(f"Train AUC Score {auc_train}")
print(f"Test AUC Score {auc_test}")

fpr, tpr, _ = roc_curve(y_val, y_val_pred_pos)


pred = clf.predict_proba(test_sel)
pred_list=[]
for i in range(len(pred)):
    pred_list.append(pred[i][1])
df_out = pd.DataFrame()
df_out['user_id'] = test_data['user_id'].astype(int)
df_out['merchant_id']=test_data['merchant_id'].astype(int)
df_out['predict_prob'] = pred_list
df_out.to_csv('prediction.csv',index=0)