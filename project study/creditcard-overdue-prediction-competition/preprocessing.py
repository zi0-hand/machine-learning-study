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

    # 자녀수가 가족수보다 같거나 큰 경우 제외 (근데 이렇게 하면 오히려 성능 떨어짐)
    #data = data[data['child_num'] < data['family_size']]

    return data

'''
# 이진 범주형 변수 처리 (라벨 인코딩 있으면 요거는 필요 없음)
def binary_cate_process(data):
    data['gender'] = data['gender'].replace(['F', 'M'], [0, 1])
    data['car'] = data['car'].replace(['N', 'Y'], [0, 1])
    data['reality'] = data['reality'].replace(['N', 'Y'], [0, 1])
    return data
'''

# 수치형 변수 처리
def numeric_process(data):

    # 출생일 연단위로 전처리
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].apply(day_to_year)

    # 고용일 연단위로 전처리
    data.loc[data['DAYS_EMPLOYED'] >= 0, 'DAYS_EMPLOYED'] = 0
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(day_to_year)

    # 카드 생성 전처리
    data['begin_month'] = data['begin_month'].apply(minus) # (요거 넣어주면 오히려 떨어짐)

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
def label_encoding(data1, data2):

    features = ['gender','car','reality','income_type','edu_type','family_type','house_type','occyp_type']
    for feature in features:
        le = LabelEncoder()
        data1 = le.fit_transform(data1[feature])
        data2 = le.transform(data2[feature])

    return data1, data2


# inf를 없애기 위한 함수 (다른 좋은 방안 있으면 디벨롭 필요)
def add_1(data):
    features = ['child_num', 'DAYS_EMPLOYED', 'family_size', 'begin_month']
    for feature in features:
        data[feature] = data[feature] + 1
    return data


# 파생변수 함수
def add_numeric_process(data):

    # 나이, 일 비교
    data['birth_work_ratio'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

    # 소득과 나이, 일 비교
    data['income_age_ratio'] = data['income_total'] / data['DAYS_BIRTH']
    data['income_em_ratio'] = data['income_total'] / data['DAYS_EMPLOYED']

    # 가족, 아이 (별로 효과 좋지 않음) -> 다듬을 필요 있음
    data['family_child_ratio'] = data['family_size'] / data['child_num']

    # 가족, 아이 대비 소득
    data['income_child_ratio'] = data['income_total'] / data['child_num']

    # 카드 생성과 나이, 일시작
    data['last1'] = data['DAYS_BIRTH'] / (data['begin_month']//12)

    # 재산 가중치
    data['money_property'] = data['car'] + data['reality']
    data['product_property'] = data['work_phone'] + data['phone'] + data['email']
    data['total_property'] = data['car'] + data['reality'] + data['car'] + data['work_phone'] + data['phone'] + data[
        'email']

    return data


'''
# 함수 실행 함수
def function_run(data):
    data = outlier_process(data)
    data = numeric_process(data)
    data = occuyp_process(data)
    data = not_use_column(data)
    data = label_encoding(data)
    return data
'''

train = outlier_process(train)

#train = binary_cate_process(train)
#test = binary_cate_process(test)

train = numeric_process(train)
test = numeric_process(test)

train = occuyp_process(train)
test = occuyp_process(test)

train = not_use_column(train)
test = not_use_column(test)

train, test = label_encoding(train, test)

# inf 제거 함수
train = add_1(train)
test = add_1(test)

# 파생변수 (순서 중요)
train = add_numeric_process(train)
test = add_numeric_process(test)

train_x = train.drop('credit', axis=1)
train_y = train[['credit']]
test_x = test

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y,
                                                  stratify=train_y, test_size=0.25, random_state=42)
cat = CatBoostClassifier()
cat.fit(X_train, y_train)
y_pred = cat.predict_proba(X_val)

print(f"log_loss: {log_loss(y_val['credit'], y_pred)}")
# 값이 작을수록 잘 예측한 것임

'''
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_val)

print(f"log_loss: {log_loss(y_val['credit'], y_pred)}")
'''

###다음에 할 거###
# 1. 이상치 제거 함수 만들기
# 해결 2. occuyp_process 함수 보충하기
# 해결 3. 필요없는 열 제거 함수 만들기 -> data.drop('FLAG_MOBIL', axis=1, inplace=True)
# 보충 4. 파생변수 생성 및 다듬기
# 5. occupy_type 결측값 예측하는 함수 만들기
# 보충 6. inf 해결하기 (분모에 뭘 둬야할지 다시 생각) **
# 해결 7. fit_transfrom 수정하기 ***

###스터디 의견공유###
# 1. 왜도 로그 변환
# 2. 식별자 생성