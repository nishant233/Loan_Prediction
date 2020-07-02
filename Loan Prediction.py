#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/home/nishant/Downloads/loan.csv')

df.info()
df.isnull().sum()
df.head(20)


#dropping Loan id because it is unique for every entry
df = df.drop(columns = ['Loan_ID'], axis = 1)


#loan Gender wise
print(df['Gender'].value_counts())
sns.countplot(x = 'Gender', data = df, palette = 'Set3')


#loan by marital Status
print(df['Married'].value_counts())
sns.countplot(x = "Married", data = df, palette = "Set2")


#loan by dependents
print(df['Dependents'].value_counts())
sns.countplot(x = 'Dependents', data = df, palette = "Set1")


#loan by Self_employed
print(df['Self_Employed'].value_counts())
sns.countplot(x = 'Self_Employed', data = df, palette = "Set1")


#on the basis of loan amount
ax = df['LoanAmount'].hist(density = True, stacked = True, color = "blue", alpha = 0.6)
df["LoanAmount"].plot(kind = 'density', color = 'blue')
ax.set(xlabel = "Loan Amount")
plt.show()


#replacing null values in Gender with mode() of gender
mode_gen = df.Gender.value_counts().idxmax()
df.Gender.fillna(mode_gen, inplace = True)
df['Gender'].isnull().sum()


#handling null values for married
mode_gen = df.Married.value_counts().idxmax()
df.Married.fillna(mode_gen, inplace = True)


#handling null values for Dependents
df.Dependents.fillna(df.Dependents.value_counts().idxmax(), inplace = True)


#handling null values for Self_Employed
df.Self_Employed.fillna(df.Self_Employed.value_counts().idxmax(), inplace = True)


#handling null values for Loan_Amount_Term
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.value_counts().idxmax(), inplace = True)


#handling null values for Credit_History
df.Credit_History.fillna(df.Credit_History.value_counts().idxmax(), inplace = True)


#handling null values for LoanAmount
df.LoanAmount.fillna(df.LoanAmount.value_counts().mean(), inplace = True)


df.info()


#dummy for gender
dummy_gen = pd.get_dummies(df.Gender)
dummy_gen.head()
df = pd.concat([dummy_gen, df], axis = 1)
df = df.drop(columns = ['Gender'], axis = 1)


#dummy for Married
dummy_gen = pd.get_dummies(df.Married, prefix='Married')
dummy_gen.head()
df = pd.concat([dummy_gen, df], axis = 1)
df= df.drop(columns = ["Married"], axis = 1)


#Dummy for Dependent
dummy_gen = pd.get_dummies(df.Dependents, prefix='Dependents')
dummy_gen.head()
df = pd.concat([dummy_gen, df], axis = 1)
df= df.drop(columns = ["Dependents"], axis = 1)


#Dummy for Education
dummy_gen = pd.get_dummies(df.Education, prefix='Education')
dummy_gen.head()
df = pd.concat([dummy_gen, df], axis = 1)
df= df.drop(columns = ["Education"], axis = 1)


#Dummy for Self_Employed
dummy_gen = pd.get_dummies(df.Self_Employed, prefix='Self_Employed')
dummy_gen.head()
df = pd.concat([dummy_gen, df], axis = 1)
df= df.drop(columns = ["Self_Employed"], axis = 1)


#Dummy for Property_Area
dummy_gen = pd.get_dummies(df.Property_Area, prefix='Property_Area')
dummy_gen.head()
df = pd.concat([dummy_gen, df], axis = 1)
df= df.drop(columns = ["Property_Area"], axis = 1)


#dropping columns
df = df.drop(columns = ['Self_Employed_No','Education_Not Graduate','Married_No','Female'], axis = 1)


#scaling the attributes
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = scaler.fit_transform(df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])


#assigning train and test
train = df.iloc[:,:-1]
test = df['Loan_Status']


#splitting data into training and testing
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(train, test, test_size = 0.2, random_state = 42)


#shape of train and test
print("Train Shape", train_X.shape)
print("Test Shape", test_X.shape)


#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(train_X, train_y)
pred_y = dtc.predict(test_X)
print("Score of Decision Tree:", dtc.score(test_X, test_y))
print("Accuracy of Decision Tree:", accuracy_score(test_y, pred_y))
print("Confusion Matrix of Decision Tree: \n", confusion_matrix(test_y, pred_y))


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 42)
rfc.fit(train_X, train_y)
pred_y = rfc.predict(test_X)
print("Score of Random Forest: ", rfc.score(test_X, test_y))
print("Accuracy of Random Forest: ", accuracy_score(test_y, pred_y))
print("Confusion Matrix of Random Forest: \n", confusion_matrix(test_y, pred_y))


#SGD
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(train_X, train_y)
pred_y = sgd.predict(test_X)
print("Score of SGD: ", sgd.score(test_X, test_y))
print("Accuracy of SGD: ", accuracy_score(test_y, pred_y))
print("Confusion Matrix of SGD: \n", confusion_matrix(test_y, pred_y))


df.to_csv('/home/nishant/Downloads/loan_pre.csv', index = False)

