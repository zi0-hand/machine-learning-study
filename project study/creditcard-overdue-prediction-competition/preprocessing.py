import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import os
os.chdir('C:/Users/sgsgk/Documents/머신러닝 프로젝트 스터디/creditcart-overdue-prediction-study/open')

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')


def day_to_year(x):
    return (x*-1)//365

def minus(x):
    return -1*x


# 이상치 처리 (보충 필요)
def outlier_process(data):

    # 자녀수가 7 이상인 데이터는 이상치라 판단
    data = data[data['child_num'] < 7]

    # 가족수가 1이면서 아이가 2인 데이터는 이상치라 판단
    #data = data[(data['family_size']==1) & (data['child_num']==2)].index
    return data

# 이진 범주형 변수 처리 (라벨 인코딩 있으면 요거는 필요 없음)
def binary_cate_process(data):
    data['gender'] = data['gender'].replace(['F', 'M'], [0, 1])
    data['car'] = data['car'].replace(['N', 'Y'], [0, 1])
    data['reality'] = data['reality'].replace(['N', 'Y'], [0, 1])
    return data


# 수치형 변수 처리
def numeric_process(data):

    # 출생일 연단위로 전처리
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].apply(day_to_year)

    # 고용일 연단위로 전처리
    data.loc[data['DAYS_EMPLOYED'] >= 0, 'DAYS_EMPLOYED'] = 0
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(day_to_year)

    return data

# 직업 유형 변수 처리 (NaN값에 대한 보충 필요)
def occuyp_process(data):
    data['occyp_type'] = data['occyp_type'].fillna('NaN')
    data.loc[(data['DAYS_EMPLOYED']==0) & (data['occyp_type']=='NaN'), 'occyp_type'] = 'no_work'
    return data

# 필요없는 열 제거
def not_use_column(data):
    data.drop(['index','FLAG_MOBIL'], axis=1, inplace=True)
    #data.drop(['index','FLAG_MOBIL','occyp_type'], axis=1, inplace=True)

    return data

# 라벨 인코딩
def label_encoding(data):

    features = ['gender','car','reality','income_type','edu_type','family_type','house_type','occyp_type']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])

    return data


# 함수 실행 함수
def function_run(data):
    data = outlier_process(data)
    data = numeric_process(data)
    data = occuyp_process(data)
    data = not_use_column(data)
    data = label_encoding(data)
    return data


train = outlier_process(train)

#train = binary_cate_process(train)
#test = binary_cate_process(test)

train = numeric_process(train)
test = numeric_process(test)

train = occuyp_process(train)
test = occuyp_process(test)

train = not_use_column(train)
test = not_use_column(test)

train = label_encoding(train)
test = label_encoding(test)


train_x = train.drop('credit', axis=1)
train_y = train[['credit']]
test_x = test

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y,
                                                  stratify=train_y, test_size=0.25, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_val)

print(f"log_loss: {log_loss(y_val['credit'], y_pred)}")
# 값이 작을수록 잘 예측한 것임

###다음에 할 거###
# 1. 이상치 제거 함수 만들기
# 2. occuyp_process 함수 보충하기
# 해결 3. 필요없는 열 제거 함수 만들기 -> data.drop('FLAG_MOBIL', axis=1, inplace=True)
# 4. occupy_type 결측값 예측하는 함수 만들기