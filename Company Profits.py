# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading CSV File

df = pd.read_csv("50_Startups (1).csv")
print("First Five Values")
print(df.head())
print(df.describe())

# Checking For Regression

df.plot.scatter('Profit', 'R&D Spend')
plt.show()

sns.pairplot(df, kind="reg")
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

# Feature Selection


x = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']

# Data Splitting


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=63)

# ML Model


model = LinearRegression().fit(xtrain, ytrain)
coef = pd.DataFrame(model.coef_, columns=['coeff'], index=x.columns)

# Checking Accuracy Of The Model


print("Accuracy of Model is:")
print(round(100 * model.score(xtest, ytest), 2))

ypred = model.predict(xtest)
print(xtest.head())

print("Predicted Values")
print(ypred[:5])

print("Original Values")
print(ytest[:5].values)

# Saving ML Model


pd.to_pickle(model, "Company Profits Predictor.pkl")
m = pd.read_pickle("Company Profits Predictor.pkl")

# Final Model


rd = eval(input("Enter Your Company's Research & Development Expenditure:"))
adm = eval(input("Enter Your Company's Administration Expenditure:"))
msp = eval(input("Enter Your Company's Marketing Expenditure:"))

query = pd.DataFrame({'R&D Spend': [rd], 'Administration': [adm], 'Marketing Spend': [msp]})

print(round(m.predict(query)[0], 2))
