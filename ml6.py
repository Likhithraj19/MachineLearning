import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv('D:/MachineLearning/train.csv')
df.head(3)
df.isnull().sum()
df = df.drop(columns='Cabin', axis=1)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum().sum()
df.info()
df= df.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
le=LabelEncoder()
df['Sex']= le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])
df.info()
X = df.drop(columns = ['Survived'],axis=1)
y=df['Survived']
X.head()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
scaled_df = pd.DataFrame(X_train, columns=X.columns)
scaled_df.head()

