# %% 9.1: 단순 회귀분석 - Sales
import os
import pandas as pd

file_path = os.path.join('data', 'Sales.csv')
Sales = pd.read_csv(file_path)
Sales


# %% 산점도 및 선형 회귀선 시각화
import seaborn as sns
sns.jointplot(x = 'Adver', y = 'Sales',
             data = Sales, kind = 'reg')

# 단순 선형회귀 모델 적합
import statsmodels.formula.api as smf
SalesFit = smf.ols(formula = 'Sales ~ Adver',
                   data = Sales).fit()
SalesFit.summary()


# 산점도와 회귀직선의 출력

# %% 선형 관계 시각화 (grouping 제외)
sns.lmplot(x = 'Adver', y = 'Sales', data = Sales)

# %% 비선형 경향성을 lowess 방식으로 시각화
sns.regplot(x = 'Adver', y = 'Sales', data = Sales, lowess = True)


# %% 9.3 예측값 및 95% 신뢰구간 계산
predictions = SalesFit.get_prediction()
predictions.summary_frame(alpha = 0.05).round(3)

# %% 새 광고비 값에 대해 예측값 및 95% 신뢰구간 계산
SalesNew = pd.DataFrame({'Adver':[20, 30, 40]})
predictions = SalesFit.get_prediction(SalesNew)
predictions.summary_frame(alpha = 0.05).round(3)



# %% 9.4: 표준화잔차의 탐색
Fitted = SalesFit.predict()
Residual = SalesFit.resid           # 순수한 자기 잔차
RStandard = SalesFit.resid_pearson  # 표준화잔차
pd.DataFrame({'Fitted':Fitted, 
              'Residual':Residual,
              'RStandard':RStandard})

# %% 잔차 분석을 위한 시각화
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (10, 8))
sns.scatterplot(x = Fitted, y = RStandard)
ax.axhline(y = 0)


# %% 9.5: 표준화잔차의 히스토그램
import numpy as np
sns.distplot(RStandard, bins = 10)

# 표준화잔차의 정규확률도표
from scipy.stats import probplot
probplot(RStandard, plot = plt)



# %% 9.6: 더빈-왓슨 통계량의 출력
from statsmodels.stats.stattools import durbin_watson
durbin_watson(RStandard)



# 9.7: 회귀분석 객체에 plot 함수 적용하기 

# %% 진단 플롯
import statsmodels.api as sm
fig = plt.figure(figsize = (12, 8))
fig = sm.graphics.plot_regress_exog(SalesFit, 'Adver', fig = fig)

# %% 잔차 플롯
import seaborn as sns
sns.residplot(x="Adver", y="Sales", data=Sales, 
              lowess=True, color="g"
)


# %% 9.8: 상관분석 - Satisfaction
import os

file_path = os.path.join('data', 'Satisfaction.csv')
Satisfaction = pd.read_csv(file_path)
Satisfaction.head(10)

# %% 피어슨의 상관계수
Satisfaction.iloc[:, 1:6].corr()

# 피어슨 상관계수 및 p값 계산
from scipy.stats import pearsonr

# %% 종속변수 Y와 독립변수 X1 간의 피어슨 상관계수 및 p값 계산
pearsonr(Satisfaction.Y, Satisfaction.X1)

# %% 종속변수 Y와 독립변수 X2 간의 피어슨 상관계수 및 p값 계산
pearsonr(Satisfaction.Y, Satisfaction.X2)

# %% 종속변수 Y와 독립변수 X3 간의 피어슨 상관계수 및 p값 계산
pearsonr(Satisfaction.Y, Satisfaction.X3)

# %% 종속변수 Y와 독립변수 X4 간의 피어슨 상관계수 및 p값 계산
pearsonr(Satisfaction.Y, Satisfaction.X4)


# %% 9.9: 다중 회귀분석 - Satisfaction
import statsmodels.formula.api as smf
SatisfactionFit = smf.ols(formula='Y~X1+X2+X3+X4',
                         data=Satisfaction).fit()
print(SatisfactionFit.summary())


# %% 9.10: 표준화 회귀계수
from scipy import stats
Satisfaction_z = Satisfaction.iloc[:,1:6].apply(stats.zscore)
SatisfactionFit = smf.ols(formula='Y~X1+X2+X3+X4',
                         data=Satisfaction_z).fit()
SatisfactionFit.params.round(5)


# %% 9.11: X4를 포함하지 않은 회귀모형
import statsmodels.formula.api as smf
SatisfactionFit1 = smf.ols(formula='Y~X1+X2+X3',
                          data=Satisfaction).fit()
print(SatisfactionFit1.summary())


# %% 9.12: 다중 회귀분석 - Multico
import os

file_path = os.path.join('data', 'Multico.csv')
Multico = pd.read_csv(file_path)
Multico.head(10)

# %% 다중 선형 회귀모형 적합
MulticoFit = smf.ols(formula = 'Y~X1+X2+X3+X4', data = Multico).fit()
print(MulticoFit.summary())



# %% 9.13: 분산확대인자(VIF)의 출력
import statsmodels.formula.api as smf
MulticoModel = smf.ols('Y~X1+X2+X3+X4', Multico)
from statsmodels.stats.outliers_influence import\
    variance_inflation_factor
print(variance_inflation_factor(MulticoModel.exog,1))

# %% 각 설명변수에 대해 VIF를 계산하여 df 생성
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame({
    'X': MulticoModel.exog_names,
    'VIE': [variance_inflation_factor(MulticoModel.exog, i) 
            for i in range(MulticoModel.exog.shape[1])]
})
vif_data


# %% 9.14: 상태지수의 출력
MulticoFit = smf.ols(formula='Y~X1+X2+X3+X4', data=Multico).fit()
MulticoFit.condition_number


# %% 9.15: 설명변수 X1과 X3만을 사용한 경우
import statsmodels.formula.api as smf
MulticoFit1 = smf.ols(formula='Y~X1+X3', data=Multico).fit()
print(MulticoFit1.summary())

# %% 다중공선성 진단을 위한 VIF 계산
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

MulticoFit1 = smf.ols(formula='Y ~ X1 + X3', data=Multico).fit()

vif_data = pd.DataFrame({
    'X': MulticoFit1.model.exog_names,
    'VIE': [variance_inflation_factor(MulticoFit1.model.exog, i)
            for i in range(MulticoFit1.model.exog.shape[1])]
})
vif_data



# %% 9.16: 가변수를 이용한 회귀분석 - Dummy
import os

file_path = os.path.join('data', 'Dummy.csv')
Dummy = pd.read_csv(file_path)
Dummy.head(10)

# %%  범주형 변수 Region을 더미 변수로 변환
import pandas as pd
Dummy['D1'] = 0
Dummy.loc[Dummy['Region']==1, 'D1']=1
Dummy['D2'] = 0
Dummy.loc[Dummy['Region']==2, 'D2']=1
Dummy.head()

# %% 회귀모형 적합
import statsmodels.formula.api as smf
DummyFit = smf.ols(formula='Y~Age+D1+D2', data=Dummy).fit()
print(DummyFit.summary())


# %% 9.17: 범주형 변수 지정 연산자 C()이용한 회귀분석
import statsmodels.formula.api as smf
DummyFit1 = smf.ols(formula='Y~Age+C(Region)', data=Dummy).fit()
print(DummyFit1.summary())

# %% 회귀모형에 대한 분산분석표 생성
import statsmodels.api as sm
sm.stats.anova_lm(DummyFit1, typ=3)