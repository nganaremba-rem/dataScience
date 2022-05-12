import numpy as np
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/remku/fishData/main/Fish.csv")

data.head(10)

x = data.iloc[:,1]
y = data.iloc[:,6]

import matplotlib.pyplot as plt
plt.scatter(x,y, color='r')

from sklearn.linear_model import LinearRegression

LM = LinearRegression()

LM.fit(np.array(x).reshape(-1,1),y)


y_hat = LM.predict(np.array(x).reshape(-1,1))


plt.plot(y,color='r')
plt.plot(y_hat,color='g')
