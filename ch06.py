# %% 1. 독립표본 t-검정
# 사례: 새로운 강의 방식이 초등학생 독해력 향상에 도움이 되는가?

import os
import pandas as pd

file_path = os.path.join('data', 'Reading.csv')
Reading = pd.read_csv(file_path)
Reading.head(10)


# 그룹별 점수 분포 
import seaborn as sns

# %% 박스 플롯
sns.boxplot(x='Group', y='Score', data=Reading)

# %% 바이올린 플롯
sns.violinplot(x='Group', y='Score', data=Reading)


# %% 그룹별 기술통계량 확인
Reading.groupby('Group').Score.describe()


# %% 그룹 분리
New = Reading[Reading.Group == 'New']
Old = Reading[Reading.Group == 'Old']


# %% 양측 검정: 두 그룹 간 차이가 없을 것이다.
from scipy.stats import ttest_ind

# 등분산 가정하 독립표본 t-검정
ttest_ind(New.Score, Old.Score, equal_var=True)

# t-통계량: 그룹 간 평균 차이의 존재 여부를 평가하는 지표
# p-값 < 0.05 → 통계적으로 유의미한 차이 존재


# %% 단측 검정: 새로운 학습법이 더 효과적이다.
stat, pval = ttest_ind(New.Score, Old.Score, equal_var=True)
print("P", pval/2)

# p-값이 0.0052로 유의수준 0.05보다 작으므로, 대립가설을 채택할 수 있다.


from statsmodels.stats.weightstats import ttest_ind

# %% 등분산 가정한 단측 검정(정수)
ttest_ind(New.Score, Old.Score, alternative='larger', usevar='pooled')  

# %% 이분산 가정한 단측 검정(실수)
ttest_ind(New.Score, Old.Score, alternative='larger', usevar='unequal')


# %% 2. 대응표본 t-검정
# 사례: 컴퓨터 교육 전후 성적 차이가 있는가?

file_path = os.path.join('data', 'Paired.csv')
Paired = pd.read_csv(file_path)
Paired.head(10)


# %% 전후 테스트 점수 박스 플롯
import seaborn as sns
sns.boxplot(data=Paired.iloc[:, [1, 2]], orient='h')


# %% 전후 차이(Diff) 계산: 교육 효과가 있다면 음수일 가능성 있음
Paired["Diff"] = Paired.Pretest - Paired.Posttest
Paired.iloc[:, 1:4].describe() # 기초 통계량 확인


# %% Diff 변수의 분포 시각화
import matplotlib.pyplot as plt
sns.distplot(Paired.Diff) # Seaborn 0.11.x 이하


# %% Diff 변수의 분포 시각화
sns.histplot(Paired.Diff, stat='density')
sns.kdeplot(Paired.Diff, fill=True)
plt.xlim(-40, 30)


# %% 대응표본 t-검정 (두 관련 집단 간 평균 차이 검정)
from scipy.stats import ttest_rel
ttest_rel(Paired.Pretest, Paired.Posttest)


# %% 단측 검정 (교육 후 점수가 더 높다고 가정)
stat, pval = ttest_rel(Paired.Pretest, Paired.Posttest)
print("one-sided p-value =", pval / 2)


# %% 3. 피셔의 정확검정
# 사례: 성별에 따라 정부 지지 여부가 다른가?

file_path = os.path.join('data', 'Support.csv')
Support = pd.read_csv(file_path)
Support.head(10)


# %% 2차원 교차표
SupportTable = pd.crosstab(index=Support["Gender"], columns=Support["YesNo"])
SupportTable

# %% 행 기준 비율로 변환
pd.crosstab(index=Support["Gender"], columns=Support["YesNo"], normalize="index")


from scipy.stats import fisher_exact

# %% 피셔의 정확검정: 두 범주형 변수의 연관성 확인
fisher_exact(SupportTable, alternative='two-sided')

# %% 카이제곱 검정 (교차표 기반 검정)
from scipy.stats import chi2_contingency
chi2_contingency(SupportTable)


# %% 4. 맥니머 검정
# 사례: 정책 발표 전후 지지율에 변화가 있는가?

file_path = os.path.join('data', 'Prepost.csv')
Prepost = pd.read_csv(file_path)
Prepost.head(10)


# %% 2차원 교차표
PrepostTable = pd.crosstab(index=Prepost["Pre"], columns=Prepost["Post"], margins=True, margins_name="합계")
PrepostTable


# %% 전체 기준 비율 확인
pd.crosstab(index=Prepost["Pre"], columns=Prepost["Post"], margins=True, margins_name="합계", normalize="all")


# %% 맥니머 정확검정 (이항분포 기반)
from statsmodels.stats.contingency_tables import mcnemar
print(mcnemar(PrepostTable, exact=True))


# %% 맥니머 근사검정 (카이제곱 근사)
print(mcnemar(PrepostTable, exact=False))


# %% 5. F-검정 함수 구현 (등분산 검정)
# Reading 데이터에서 두 그룹 간 분산 차이가 존재하는가?

New = Reading[Reading.Group == 'New']
Old = Reading[Reading.Group == 'Old']

import numpy as np
from scipy import stats

def F_test(x, y):
    f = np.var(x, ddof=1) / np.var(y, ddof=1)
    df1 = x.size - 1
    df2 = y.size - 1
    p = 2 * (1 - stats.f.cdf(f, df1, df2))
    return f, p

F_test(New.Score, Old.Score)


# %% Bartlett 검정 (정규성 가정 하 분산 동일성 검정)
stats.bartlett(New.Score, Old.Score)


# %% Levene 검정 (비정규 데이터에 더 강건한 등분산 검정)
stats.levene(New.Score, Old.Score)