# %% 1. 피어슨 적률 상관계수
# 사례: 소득과 지출 사이에 선형 상관관계가 있는지 분석

import os
import pandas as pd

file_path = os.path.join('data', 'Student.csv')
Student = pd.read_csv(file_path)
Student.head(10)


# %% 산점도: 소득(Income)과 지출(Expense)의 관계 시각화
import matplotlib.pyplot as plt

plt.plot('Income', 'Expense', 'o', color='black', data=Student)
plt.xlabel('Income')
plt.ylabel('Expense')


# %% 산점도 (다른 방식)
plt.scatter('Income', 'Expense', data=Student)
plt.xlabel('Income')
plt.ylabel('Expense')


# %% 산점도와 회귀선 결합한 시각화
import seaborn as sns
sns.jointplot(x='Income', y='Expense', data=Student, kind="reg")


# %% 여러 변수 간 산점도 및 분포 시각화
sns.pairplot(Student.iloc[:,1:4])

g = sns.PairGrid(Student.iloc[:,1:4])
g.map_upper(sns.scatterplot)  # 상삼각은 산점도
g.map_lower(sns.kdeplot)      # 하삼각은 밀도곡선
g.map_diag(sns.kdeplot, lw=3) # 대각선은 밀도곡선


# %% 피어슨 상관계수 행렬 계산
Student.iloc[:,1:4].corr(method='pearson')


# %% Income과 Expense 간 피어슨 상관계수 및 p-값 계산
from scipy.stats import pearsonr
pearsonr(Student.Income, Student.Expense)


# %% 피어슨 상관계수 행렬 (반올림 처리)
import pingouin as pg
Student.iloc[:,1:4].pairwise_corr(method='pearson').round(3)


# %% 상관계수 행렬 저장
corrMatrix = Student.iloc[:,1:4].corr(method='pearson')
sns.heatmap(corrMatrix, annot=True) # 히트맵


# %% 2. 편상관계수
# 사례: 나이(Age)를 통제했을 때 기능(Satis1)과 디자인(Satis2) 만족도 간 상관관계 분석

import os
import pandas as pd

file_path = os.path.join('data', 'Satis.csv')
Satis = pd.read_csv(file_path)
Satis.head(10)


# %% 단순 상관계수 행렬 (나이 미통제)
Satis.iloc[:,1:4].corr().round(3)


# %% 두 변수 간 단순 피어슨 상관계수 계산
from scipy.stats import pearsonr
pearsonr(Satis.Satis1, Satis.Satis2)


# %% 편상관계수 행렬 계산 (나이 통제)
import pingouin as pg
Satis.iloc[:,1:4].pcorr().round(3)


# %% 특정 두 변수 간 편상관계수 계산
Satis.partial_corr(x='Satis1', y='Satis2', covar='Age').round(3)


# %% 나이 기준으로 집단 생성 (30세 미만, 이상)
Satis.loc[Satis.Age < 30, "AgeGroup"] = "Under30"
Satis.loc[Satis.Age >= 30, "AgeGroup"] = "Over30"


# %% 집단별 산점도 (나이 그룹에 따른 색상 구분)
sns.jointplot(x='Satis1', y='Satis2', data=Satis, hue="AgeGroup")


# %% 집단별 데이터 분리
SatisUnder30 = Satis.loc[Satis.AgeGroup == "Under30", ]
SatisOver30 = Satis.loc[Satis.AgeGroup == "Over30", ]

from scipy.stats import pearsonr

# 집단별 피어슨 상관계수 및 p-값 출력
print(pearsonr(SatisUnder30.Satis1, SatisUnder30.Satis2))
print(pearsonr(SatisOver30.Satis1, SatisOver30.Satis2))


# %% 3. 신뢰도 분석 (크론바흐 알파)
# 사례: 기업 구성원의 의식 설문 문항 신뢰도 평가

import os
import pandas as pd

file_path = os.path.join('data', 'Ability.csv')
Ability = pd.read_csv(file_path)
Ability.head(10)


# %% 문항 간 상관관계 확인 (Q01~Q03)
Ability[["Q01", "Q02", "Q03"]].corr()


# %% 크론바흐 알파 계산
import pingouin as pg
pg.cronbach_alpha(data=Ability[["Q01", "Q02", "Q03"]])


# %% 다른 문항군 상관관계 계산
Ability[["Q04", "Q05", "Q06", "Q07"]].corr()

# %% 크론바흐 알파 계수 계산
pg.cronbach_alpha(data=Ability[["Q04", "Q05", "Q06", "Q07"]])


# %% 역문항 처리: Q07_R = 6 - Q07 (역코딩)
Ability["Q07_R"] = 6 - Ability.Q07
Ability[["Q04", "Q05", "Q06", "Q07_R"]].corr()

# %% 크론바흐 알파 계수 계산
pg.cronbach_alpha(data=Ability[["Q04", "Q05", "Q06", "Q07_R"]])


# %% 추가 문항군 신뢰도 확인
Ability[["Q08", "Q09", "Q10"]].corr()

# %% 크론바흐 알파 계수 계산
pg.cronbach_alpha(data=Ability[["Q08", "Q09", "Q10"]])


# %% 크론바흐 알파 직접 계산 함수 구현
import numpy as np

def CronbachAlpha(df):
    df_corr = df.corr()
    N = df.shape[1]
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    cronbach_alpha = (N * mean_r) / (1 + (N-1) * mean_r)
    return cronbach_alpha

# 그룹 1
CronbachAlpha(Ability[["Q01", "Q02", "Q03"]])

# %% 그룹 2
CronbachAlpha(Ability[["Q04", "Q05", "Q06", "Q07"]])

# %% 그룹 3
CronbachAlpha(Ability[["Q08", "Q09", "Q10"]])