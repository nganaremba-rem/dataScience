import sklearn

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

text = ["I love NLP",

       "NLP is future",

       "I will learn in 2 months"]

vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(text)

count_array = matrix.toarray()

df = pd.DataFrame(data=count_array, columns = vectorizer.get_feature_names())

print(df)
