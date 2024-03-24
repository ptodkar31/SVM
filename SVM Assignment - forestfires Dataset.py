# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:13:29 2024

@author: Priyanka
"""

"""
In California, annual forest fires can cause huge loss of wildlife,
human life, and can cost billions of dollars in property damage.
Local officials would like to predict the size of the burnt area in
forest fires annually so that they can be better prepared in future calamities. 
Build a Support Vector Machines algorithm on the dataset and share your insights 
on it in the documentation. 

Business Problem-
What is the business objective?
In 2018, California wildfires caused economic losses of 
nearly $150 billion, or about 0.7 percent of the gross 
domestic product of the entire United States that year, 
and a considerable fraction of those costs affected people 
far from the fires and even outside of the Golden State.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
forest=pd.read_csv("C:/Data Set/Dataset-SVM/forestfires.csv")

forest.dtypes
'''
month             object
day               object
FFMC             float64
DMC              float64
DC               float64
ISI              float64
temp             float64
RH                 int64
wind             float64
rain             float64
area             float64
dayfri             int64
daymon             int64
daysat             int64
daysun             int64
daythu             int64
daytue             int64
daywed             int64
monthapr           int64
monthaug           int64
monthdec           int64
monthfeb           int64
monthjan           int64
monthjul           int64
monthjun           int64
monthmar           int64
monthmay           int64
monthnov           int64
monthoct           int64
monthsep           int64
size_category     object
'''

#EDA
forest.shape
#(517, 31)

#plt.figure(1,figsize=(16,10))
#sns.countplot(forest.month)

plt.figure(figsize=(16, 10))
sns.countplot(x='month', data=forest, order=forest['month'].value_counts().index)
#Aug and sept has highest value

#sns.countplot(forest.day)
sns.countplot(x='day', data=forest, order=forest['day'].value_counts().index)
#Friday sunday and saturday has highest value

sns.distplot(forest.FFMC)
#data isnormal and slight left skewed
sns.boxplot(forest.FFMC)
#There are several outliers

sns.distplot(forest.DMC)
#data isnormal and slight right skewed
sns.boxplot(forest.DMC)
#There are several outliers

sns.distplot(forest.DC)
#data isnormal and slight left skewed
sns.boxplot(forest.DC)
#There are  outliers

sns.distplot(forest.ISI)
#data isnormal 
sns.boxplot(forest.ISI)
#There are  outliers

sns.distplot(forest.temp)
#data isnormal a
sns.boxplot(forest.temp)
#There are  outliers

sns.distplot(forest.RH)
#data isnormal and slight left skewed
sns.boxplot(forest.RH)
#There are  outliers

sns.distplot(forest.wind)
#data isnormal and slight right skewed
sns.boxplot(forest.wind)
#There are  outliers

sns.distplot(forest.rain)
#data isnormal 
sns.boxplot(forest.rain)
#There are  outliers

sns.distplot(forest.area)
#data isnormal 
sns.boxplot(forest.area)
#There are  outliers

#Now let us check the Highest Fire In KM?
forest.sort_values(by="area", ascending=False).head(5)

highest_fire_area = forest.sort_values(by="area", ascending=True)
highest_fire_area.head(5)
'''
month  day  FFMC    DMC  ...  monthnov  monthoct  monthsep  size_category
0     mar  fri  86.2   26.2  ...         0         0         0          small
298   jun  wed  91.2  147.8  ...         0         0         0          small
299   jun  sat  53.4   71.0  ...         0         0         0          small
300   jun  mon  90.4   93.3  ...         0         0         0          small
302   jun  fri  91.1   94.1  ...         0         0         0          small
'''


plt.figure(figsize=(8, 6))
plt.title("Temperature vs area of fire" )
plt.bar(highest_fire_area['temp'], highest_fire_area['area'])
plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()
#once the fire starts,almost 1000+ sq area's temperature goes beyond 25 and 
#around 750km area is facing temp 30+

#Now let us check the highest rain in the forest
highest_rain = forest.sort_values(by='rain', ascending=False)[['month', 'day', 'rain']].head(5)
highest_rain
'''
month  day  rain
499   aug  tue   6.4
509   aug  fri   1.4
243   aug  sun   1.0
500   aug  tue   0.8
501   aug  tue   0.8
'''
#highest rain observed in the month of aug

#Let us check highest and lowest temperature in month and day wise
highest_temp = forest.sort_values(by='temp', ascending=False)[['month', 'day', 'temp']].head(5)
print("Highest Temperature\n",highest_temp)
'''
Highest Temperature:
     month  day  temp
498   aug  tue  33.3
484   aug  sun  33.1
496   aug  mon  32.6
492   aug  fri  32.4
491   aug  thu  32.4
'''
#Highest temp observed in aug

lowest_temp =  forest.sort_values(by='temp', ascending=True)[['month', 'day', 'temp']].head(5)
print("Lowest Temperature\n",lowest_temp)
'''
Lowest Temperature:
     month  day  temp
280   dec  fri   2.2
282   feb  sun   4.2
279   dec  mon   4.6
278   dec  mon   4.6
277   dec  mon   4.6
'''
#lowest temperature in the month of dec

forest.isna().sum()
'''
month            0
day              0
FFMC             0
DMC              0
DC               0
ISI              0
temp             0
RH               0
wind             0
rain             0
area             0
dayfri           0
daymon           0
daysat           0
daysun           0
daythu           0
daytue           0
daywed           0
monthapr         0
monthaug         0
monthdec         0
monthfeb         0
monthjan         0
monthjul         0
monthjun         0
monthmar         0
monthmay         0
monthnov         0
monthoct         0
monthsep         0
size_category    0
'''
#There no missing values in both


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size_category)

forest.dtypes
'''
month              int64
day                int64
FFMC             float64
DMC              float64
DC               float64
ISI              float64
temp             float64
RH                 int64
wind             float64
rain             float64
area             float64
dayfri             int64
daymon             int64
daysat             int64
daysun             int64
daythu             int64
daytue             int64
daywed             int64
monthapr           int64
monthaug           int64
monthdec           int64
monthfeb           int64
monthjan           int64
monthjul           int64
monthjun           int64
monthmar           int64
monthmay           int64
monthnov           int64
monthoct           int64
monthsep           int64
size_category      int64
'''
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['month'])
df_t=winsor.fit_transform(forest[["month"]])
sns.boxplot(df_t.month)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['FFMC'])
df_t=winsor.fit_transform(forest[["FFMC"]])
sns.boxplot(df_t.FFMC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DMC'])
df_t=winsor.fit_transform(forest[["DMC"]])
sns.boxplot(df_t.DMC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DC'])
df_t=winsor.fit_transform(forest[["DC"]])
sns.boxplot(df_t.DC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ISI'])
df_t=winsor.fit_transform(forest[["ISI"]])
sns.boxplot(df_t.ISI)


from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['temp'])
df_t=winsor.fit_transform(forest[["temp"]])
sns.boxplot(df_t.temp)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RH'])
df_t=winsor.fit_transform(forest[["RH"]])
sns.boxplot(df_t.RH)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['wind'])
df_t=winsor.fit_transform(forest[["wind"]])
sns.boxplot(df_t.wind)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rain'])
df_t=winsor.fit_transform(forest[["rain"]])
sns.boxplot(df_t.rain)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['area'])
df_t=winsor.fit_transform(forest[["area"]])
sns.boxplot(df_t.area)


tc = forest.corr()
tc
fig,ax= plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#all the variables are moderately correlated with size_category except area

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_X=train.iloc[:,:30]
train_y=train.iloc[:,30]
test_X=test.iloc[:,:30]
test_y=test.iloc[:,30]

#Kernel linear
model_linear=SVC(kernel="linear")
model_linear.fit(train_X,train_y)
pred_test_linear=model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)
#0.9807692307692307

#RBF
model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)
#0.7051282051282052
