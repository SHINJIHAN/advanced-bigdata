# %% 2차원 교차표 작성
import os
import pandas as pd

file_path = os.path.join('data', 'Prefer.csv')
Prefer = pd.read_csv(file_path)
Prefer.head(10)


# %% 관측빈도
PreferTable = pd.crosstab(index=Prefer["Agegroup"], columns=Prefer["Product"])
PreferTable


# %% 행합계, 열합계 포함한 교차표
pd.crosstab(
    index=Prefer["Agegroup"], columns=Prefer["Product"], 
    margins=True, margins_name='합계'
)

# %% 각 행의 비율로 정규화된 교차표
pd.crosstab(
    index=Prefer["Agegroup"], columns=Prefer["Product"], 
    margins=True, margins_name='합계', normalize='index'
)


# %% 피셔의 정확검정 실행
from FisherExact import fisher_exact
fisher_exact(PreferTable, alternative='two-sided')


# %% 카이제곱 검정 실행
from scipy.stats import chi2_contingency
chi2, pval, df, expected = chi2_contingency(PreferTable)

print("Chi2 =", chi2)                # 카이제곱 통계량
print("p-value =", pval)             # p-값
print("df =", df)                    # 자유도
print("Expected value =", expected)  # 기대빈도표



# %% 8.4: 2차원 교차표 작성 – Softdrink 데이터
import pandas as pd
file_path = os.path.join('data', 'Softdrink.csv')
Softdrink = pd.read_csv(file_path)
Softdrink.head(10)


# %% 값 변환 (숫자 → 문자로 범주화)
Softdrink = Softdrink.replace({
    'Agegroup': {1: '20대', 2: '30대', 3: '40대'},
    'Drink': {1: 'coke', 2: 'pepsi', 3: 'fanta', 4: 'others'}
})

# %% 1. 단순 교차표 (Count 합계 기준)
pd.crosstab(
    index=Softdrink["Agegroup"],
    columns=Softdrink["Drink"],
    values=Softdrink["Count"],
    aggfunc='sum',
    margins=True,
    margins_name='전체'
)

# %% 2. 비율 교차표 (행 기준 정규화)
pd.crosstab(
    index=Softdrink["Agegroup"],
    columns=Softdrink["Drink"],
    values=Softdrink["Count"],
    aggfunc='sum',
    margins=True,
    margins_name='전체',
    normalize='index'
).round(3)



# 8.5: 피셔의 정확검정과 카이제곱검정 – Softdrink 데이터

# %% 1. 교차표 만들기
SoftdrinkTable = Softdrink.pivot_table(
    index="Agegroup", 
    columns="Drink", 
    values="Count", 
    aggfunc='sum',
    fill_value=0  # 결측값이 있을 경우 0으로 채움
)

# %% 2. 피셔의 정확검정
fisher_exact(SoftdrinkTable.values, alternative='two-sided')

# %% 3. 카이제곱검정
chi2, pval, df, expected = chi2_contingency(SoftdrinkTable)

print("Chi-squared =", chi2)
print("p-value =", pval)
print("df =", df)
print("Expected values =\n", expected)



# %% 8.6: 완두콩 데이터에 대한 적합도 검정
import numpy as np
from scipy.stats import chisquare

Pea = np.array([315, 108, 101, 32]) # 관측 빈도

# %% 기대비율에 총합을 곱하여 기대빈도 계산
P0 = np.array([9/16, 3/16, 3/16, 1/16]) * np.sum(Pea)
print("기대빈도:", P0)

# %% 카이제곱 적합도 검정
chi2_stat, p_value = chisquare(f_obs=Pea, f_exp=P0)

print("Chi-squared =", chi2_stat)
print("p-value =", p_value)



# 8.7: 오즈비 계산과 독립성 검정

# %% 1. 데이터 생성
Smoking = pd.DataFrame({
    "Y": [1, 1, 2, 2],      # 질병 여부 (예: 1 = 있음, 2 = 없음)
    "X": [1, 2, 1, 2],      # 흡연 여부 (예: 1 = 흡연, 2 = 비흡연)
    "Count": [780, 220, 200, 800]
})

# %% 2. 2x2 분할표 작성
SmokingTable = Smoking.pivot_table(
    index="X", 
    columns="Y", 
    values="Count", 
    aggfunc='sum',
    fill_value=0
)
print(SmokingTable)


# %% 오즈비, p-값
from scipy.stats import fisher_exact
oddsratio, p_value = fisher_exact(SmokingTable.values, alternative="greater")

print("오즈비:", oddsratio)
print("p-값:", p_value)


# %% 8.8: 분할표 작성과 카이제곱검정
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic

# 1. 데이터 정렬 및 모자이크 플롯 시각화
mosaic(Prefer.sort_values("Product"), ["Product", "Agegroup"])
PreferTable

# %% 2. 교차표: 행 백분율 출력 (정규화된 분할표)
pd.crosstab(index=Prefer["Agegroup"], 
            columns=Prefer["Product"], normalize='index'
)

# %% 3. 카이제곱 독립성 검정
chi2_contingency(PreferTable)



# %% 8.9: 명목형 변수 연관성 측도
chi2 = chi2_contingency(PreferTable)[0] 

# %% 행 개수 (관측치 수)
n = len(Prefer)

# 파이 계수 (Phi coefficient)
phi = (chi2 / n) ** 0.5
print("파이 계수:", phi)

# 분할 계수 (Contingency coefficient)
contingency_coef = (chi2 / (chi2 + n)) ** 0.5
print("분할 계수:", contingency_coef)

# 크라머의 V (Cramér's V)
r, c = PreferTable.shape
cramers_v = phi / ((min(r - 1, c - 1)) ** 0.5)
print("크라머의 V:", cramers_v)


# %% 8.10: 교차표 작성 - Economic
import os
import pandas as pd

file_path = os.path.join('data', 'Economic.csv')
Economic = pd.read_csv(file_path)
Economic.head(10)

# %% 1. 분할표 작성 (행, 열 합계 포함)
pd.crosstab(
    index=Economic["Education"],
    columns=Economic["Income"],
    margins=True
)

# %% 2. 행 기준 백분율 교차표 (소수점 3자리)
pd.crosstab(
    index=Economic["Education"],
    columns=Economic["Income"],
    margins=True, normalize='index'
).round(3)



# 8.11: 상관계수 계산 - Economic

# %% 피어슨 상관계수
Economic.iloc[:, 1:3].corr(method="pearson")

# %% 스피어만 순위상관계수
Economic.iloc[:, 1:3].corr(method="spearman")

# %% 켄달의 타우 상관계수
Economic.iloc[:, 1:3].corr(method="kendall")


# %% 8.12: M-H 카이제곱 검정
from scipy.stats import chi2
MHchi2 = (1000 - 1) * 0.206019**2
pval = 1 - chi2.cdf(MHchi2, df=1)
print("M-H chi2:", MHchi2)
print("p-값:", pval)


# %% 8.13: 켄달의 타우 계산
from scipy.stats import kendalltau
tau, p_value = kendalltau(Economic["Education"], Economic["Income"])

print("켄달의 타우", tau)
print("p-값:", p_value)


# %% 8.14: 감마 계수 계산
import itertools

def Concordance(A, B):
    pairs = itertools.combinations(range(len(A)), 2)
    P = 0  # 일치쌍 (concordant pairs)
    T = 0  # 동률 (tie)
    Q = 0  # 비일치쌍 (discordant pairs)

    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        if a * b > 0:        # 같은 부호 → 일치쌍
            P += 1
        elif a * b == 0:     # 하나라도 0이면 동률
            T += 1
        else:                # 다른 부호 → 비일치쌍
            Q += 1
    return P, T, Q

X = Economic["Education"]
Y = Economic["Income"]

P, T, Q = Concordance(X, Y)
gamma = (P - Q) / (P + Q) if (P + Q) != 0 else None
print("Gamma =", gamma)