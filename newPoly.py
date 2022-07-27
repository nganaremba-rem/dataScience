import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/remku/fishData/main/FishAllTogetherFinal.csv")
data.head(10)

data = data.fillna(method = "ffill")
toBeConverted = [1, 2, 4, 5 , 6 , 8]
labelEncoding = LabelEncoder()
for i in toBeConverted:
  data.iloc[:,i] = labelEncoding.fit_transform(data.iloc[:,i])

x = data.iloc[:,2]; # Type of fish
y = data.iloc[:,3]; # To predict Price Per KG

PF = PolynomialFeatures(degree=4)
x_poly = PF.fit_transform(np.array(x).reshape(-1,1))

LM = LinearRegression()
LM.fit(x_poly, y)

y_hat = LM.predict(x_poly);

plt.plot(y, color='g')
plt.plot(y_hat, color='r')
plt.show()
