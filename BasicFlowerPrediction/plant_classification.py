import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load Dataset
df = pd.read_csv("datasets/Iris.csv")
df=df.drop(columns=["Id"],axis=1)
#Data Distribution
df["Species"].value_counts()

#Data Preprocessing
df.isnull().sum() # none null values

#Data Explorization

##Histogram
df["PetalWidthCm"].hist()

##Scatter Plot

species=['Iris-setosa',
'Iris-versicolor',
'Iris-virginica' ]

colors=["red","green","blue"]

#Sepal Scatter
for i in range(len(species)):
    sub_df=df[df["Species"] == species[i]]
    plt.scatter(sub_df["SepalLengthCm"],sub_df["SepalWidthCm"],c=colors[i],label=species[i])
    plt.legend()
#Petal Scatter
for i in range(len(species)):
    sub_df=df[df["Species"] == species[i]]
    plt.scatter(sub_df["PetalLengthCm"],sub_df["PetalWidthCm"],c=colors[i],label=species[i])
    plt.legend()
    plt.xlabel("PetalLengthCm")
    plt.ylabel('PetalWidthCm')

#Length Scatter
for i in range(len(species)):
    sub_df=df[df["Species"] == species[i]]
    plt.scatter(sub_df["SepalLengthCm"],sub_df["PetalLengthCm"],c=colors[i],label=species[i])
    plt.legend()
    plt.xlabel("SepalLengthCm")
    plt.ylabel('PetalLengthCm')

#Width Scatter
for i in range(len(species)):
    sub_df=df[df["Species"] == species[i]]
    plt.scatter(sub_df["SepalWidthCm"],sub_df["PetalWidthCm"],c=colors[i],label=species[i])
    plt.legend()
    plt.xlabel("SepalWidthCm")
    plt.ylabel('PetalWidthCm')

corr_matrix=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,annot=True,cmap="YlGnBu")


#Encoding
lbl_enc=LabelEncoder()
df["Species_ENC"]=lbl_enc.fit_transform(df["Species"])

#Model Training
X=df.drop(columns=["Species","Species_ENC"])
y=df["Species_ENC"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Classification Algorithms

##Logistic Regression
lgc_reg = LogisticRegression()
lgc_reg.fit(X_train,y_train)
y_pred = lgc_reg.predict(X_test)

acc_score=accuracy_score(y_test,y_pred)

filename = 'iris_model.sav'
pickle.dump(lgc_reg, open(filename, 'wb'))

