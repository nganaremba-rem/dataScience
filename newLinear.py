import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

data = pd.read_csv('https://raw.githubusercontent.com/remku/fishData/main/FishAllTogetherFinal.csv')
# If you wanna print the first 10 data use 
# data.head(10)

# Filling missing data and Encoding text data into number
data = data.fillna(method = 'ffill')
label_encoding = LabelEncoder()
toBeEncoded = [1, 2, 4, 5, 6, 8]
for i in toBeEncoded:
    data.iloc[:,i] = label_encoding.fit_transform(data.iloc[:,i])
    
# Taking x and y value (using x to predict y)
x = data.iloc[:, 2] # Type of fish
y = data.iloc[:, 3] # Price Per Kg

# LinearRegression
LM = LinearRegression()
LM.fit(np.array(x).reshape(-1,1), y)

y_hat = LM.predict(np.array(x).reshape(-1,1))

plt.plot(y)
plt.plot(y_hat)
