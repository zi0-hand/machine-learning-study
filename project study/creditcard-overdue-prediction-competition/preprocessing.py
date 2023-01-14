import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

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
os.chdir('C:/Users/sgsgk/Documents/머신러닝 프로젝트 스터디/creditcart-overdue-prediction/open')

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')


# 이상치 처리 (보충 필요)
def outlier_process(train):
    train = train[train['child_num'] < 7]
    train = train[(train['family_size']==1) & (train['child_num']==2)].index
    return train

# 직업 유형 변수 처리
def occuyp_process(data):
    data.fillna('NaN', inplace=True)
    return data


def day_to_year(x):
    return (-1*x)//365

def minus(x):
    return -1*x


# 이진 범주형 변수 처리
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
    data = data[data['DAYS_EMPLOYED'] < 0]
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(day_to_year)

    return data


###다음에 할 거###
# 1. 이상치 제거 함수 만들기
# 2. occuyp_process 함수 보충하기
# 3. 필요없는 열 제거 함수 만들기 -> data.drop('FLAG_MOBIL', axis=1, inplace=True)