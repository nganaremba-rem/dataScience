import numpy as np
import pandas as pd

data = pd.read_csv("https://raw.githubcontentuser.com/remku/fishData/main/Fish.csv")

x = data.iloc[:,1]
y = data.iloc[:,6]

from sklearn.preprocessing import PolynomialFeatures

PF = PolynomialFeatures()
y_poly = PF.fit_transform(np.array(y))

from sklearn.linear_model import LinearRegression

LM = LinearRegression()
LM.fit(y_poly)

y_hat_poly = LM.predict(y_poly)

plt.plot(y, color='r')
plt.plot(y_hat_poly, color='g')
