import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
import matplotlib.pyplot as plt
from scipy import correlate
import statsmodels.api as sm
data = pd.read_excel("DataWHR.xls")
x = np.array(data["Log GDP per capita"]).reshape((-1,1))
y = np.array(data["Healthy life expectancy at birth"])

#Calculates regression line based on the sklearn library
lr = LinearRegression()
lr.fit(x, y)
predict_y = lr.predict(x)

r_sq = lr.score(x, y)
b0 = lr.intercept_
b1 = lr.coef_

#Creates an array of data based on b1 and b0
w=np.linspace(6.64,11.65,100)
y11=b1*w+b0

#Scatter plot with regression line
ax1=plt.scatter(x,y)
ax2=plt.plot(w,y11,color="r",linewidth=3)

x=sm.add_constant(x)

#Fits ai model on x(GDP) and y(Life expectancy) to find relevant statistics
model=sm.OLS(y,x).fit()
predictions=model.predict(x)

#Gives summary statistics on the correlation dataset
print_model=model.summary()
print(print_model)
 


