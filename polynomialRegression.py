import numpy as np
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/remku/fishData/main/Fish.csv")

x = data.iloc[:,1]
y = data.iloc[:,6]

from sklearn.preprocessing import PolynomialFeatures

PF = PolynomialFeatures(degree=4)
x_poly = PF.fit_transform(np.array(x).reshape(-1,1))

from sklearn.linear_model import LinearRegression

LM = LinearRegression()
LM.fit(x_poly, y)

y_hat_poly = LM.predict(x_poly)

plt.plot(y, color='r')
plt.plot(y_hat_poly, color='g')
