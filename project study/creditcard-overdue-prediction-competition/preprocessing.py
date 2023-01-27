from sklearn.preprocessing import LabelEncoder

def day_to_year(x):
    return (x*-1)//365

def minus(x):
    return -1*x


# 이상치 처리 (보충 필요)
def outlier_process(data):
    data = data[data['child_num'] < 7]
    return data

# 수치형 변수 처리
def numeric_process(data):

    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].apply(day_to_year)
    data.loc[data['DAYS_EMPLOYED'] >= 0, 'DAYS_EMPLOYED'] = 0
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(day_to_year)
    data['begin_month'] = data['begin_month'].apply(minus)

    return data

# 직업 유형 변수 처리 (NaN값에 대한 보충 필요)
def occuyp_process(data):

    data['occyp_type'] = data['occyp_type'].fillna('NaN')
    data.loc[(data['DAYS_EMPLOYED']==0) & (data['occyp_type']=='NaN'), 'occyp_type'] = 'no_work'

    return data

# 필요없는 열 제거
def not_use_column(data):
    data.drop(['index','FLAG_MOBIL'], axis=1, inplace=True)
    return data

# 라벨 인코딩
def label_encoding(data1, data2):

    features = ['gender','car','reality','income_type','edu_type','family_type','house_type','occyp_type']

    for feature in features:
        le = LabelEncoder()
        data1[feature] = le.fit_transform(data1[feature])
        data2[feature] = le.transform(data2[feature])

    return data1, data2

def add_1(data):

    features = ['child_num', 'DAYS_EMPLOYED', 'family_size', 'begin_month']

    for feature in features:
        data[feature] = data[feature] + 1

    return data

def add_numeric_process(data):

    data['ratio_birth_work'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['ratio_family_child'] = data['family_size'] / data['child_num']
    data['ratio_income_child'] = data['income_total'] / data['child_num']
    data['ratio_income_age'] = data['income_total'] / data['DAYS_BIRTH']
    data['ratio_income_em'] = data['income_total'] / data['DAYS_EMPLOYED']
    data['money_property'] = data['car'] + data['reality']
    data['product_property'] = data['work_phone'] + data['phone'] + data['email']
    data['total_property'] = data['car'] + data['reality'] + data['car'] + data['work_phone'] + data['phone'] + data['email']

    return data

###다음에 할 거###
# 완료 1. 이상치 제거 함수 만들기
# 해결 2. occuyp_process 함수 보충하기
# 해결 3. 필요없는 열 제거 함수 만들기 -> data.drop('FLAG_MOBIL', axis=1, inplace=True)
# 보충 4. 파생변수 생성 및 다듬기
# 5. occupy_type 결측값 예측하는 함수 만들기
# 보충 6. inf 해결하기 (분모에 뭘 둬야할지 다시 생각) **
# 해결 7. fit_transfrom 수정하기 ***
# 8. 결혼 안한 변수 생성

###스터디 의견공유###
# 미결 1. 왜도 로그 변환 -> 오히려 성능 떨어짐
# 2. 식별자 생성