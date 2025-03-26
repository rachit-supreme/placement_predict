import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

df1=pd.read_csv('modified_placement_data.csv')

df1

df1.shape

df1.info()

df=pd.read_csv('placement.csv')

df

df=df.iloc[: ,1:]

df

df.info()



plt.scatter(df['cgpa'] , df['iq'],c=df['placement'])

X=df.iloc[: , 0:2]
y=df.iloc[: , -1]

X

y.shape

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.1)

X_test

y_train

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_test

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

y_test

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)



pickle.dump(clf,open('model.pkl','wb'))

file_path = 'model.pkl'

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(data)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except pickle.UnpicklingError:
    print("Error: The file content is not a valid pickle format.")
except EOFError:
    print("Error: The file is incomplete or corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

