# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:16:43 2017

@author: milao
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab 
import scipy.stats as stats
from scipy.stats import chisquare
from sklearn.preprocessing import scale
import numpy as np
from statsmodels.stats.proportion import proportions_ztest




df = pd.read_csv("./v1_thesis.csv", encoding="latin1")




#Hrs_to_FirstCulture #LOS #Pregnancy_Week
#Corellations between the variables
A =  df[["LOS","Pregnancy_Week","Weight","GenderNB","Afgar1_nomissing","Afgar5_nomissing","Birth_Type_Code","PoseCode","Mother_age","Anesthesia_Type_code","SGA","Fetus_Count"]]
cor  = A.corr(method = 'pearson')
print(cor)
import seaborn as sns
sns.heatmap(cor, 
          xticklabels=cor.columns.values,
           yticklabels=cor.columns.values)


X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()


counts = np.array([24, 132])
nobs = np.array([38, 132])
stat, pval = proportions_ztest(counts, nobs, 0)
print('{0:0.4f}'.format(pval))
print (pval)




#some descriptives
#Crosstabs
print (pd.crosstab(df['Pregnancy_Week'], df['GenderNB'], rownames=['CultureBinar']))

df[["Year_Admission", "GenderNB", "Pregnancy_Week"]].hist()
plt.show()

#dummie
table = pd.get_dummies(df)
table.groupby('Pregnancy_Week').size()
#weights = table["]
#a = list(table.columns.values)


#table = table[table["Year_Admission"]>2012]

# scaling and scatter plots
X_new = df['Weight']#.values

X_new = X_new.dropna()
#normality test
stats.normaltest(X_new)


#plt.hist(X_new, bins=15)
X_new.plot(kind="hist", normed=True)
X_new.plot.kde(c = "red")
#plt.show()
s
#X_scaled = scale(X_new)

norm=np.random.normal(loc=35, scale=5, size=5021)
aa=stats.chisquare(X_new,f_exp = norm)
mu, sigma = 35, 0.1 # mean and standard deviation

count, bins, ignored = plt.hist(X_new, 15, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

plt.hist(X_new, bins=15)
plt.xlabel('Pregnancy week')
plt.ylabel('Neonates') 
plt.show()

#measurements = np.random.normal(loc = 20, scale = 5, size=5000) 
#stats.probplot(X_new, dist="norm", plot=pylab)
#pylab.show()
days_culture_release
data_x_numeric =  df[["days_culture_release","Pregnancy_Week"]]
data_y = df[["Days_to_FirstCulture"]]
estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, data_y)


