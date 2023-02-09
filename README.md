# Linear-Regression
Basic Linear Regression using multiple dependent variables.
#install packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize


#Load the data
LoanPayDF = pd.read_csv("bankPayment.csv")


#check the loaded data
LoanPayDF.head()


#confirm the data types 
LoanPayDF.info()


#Perform Exploratory Data Analysis (EDA) / preprocessing 
#check whether the data has NA
LoanPayDF.isnull().sum()


#Remove Null values (NA)
LoanPayDF.dropna(inplace = True)


#Check again to confirm the NA have been removed
LoanPayDF.isnull().sum()


#Analyze and plot some of the variables
features = ['int_rate', 'LoanTerm', 'home_ownership']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    LoanPayDF.groupby(col).mean()['total_pymnt'].plot.bar()
plt.show()


##plot (1) the relationship between dependent and independent variables 
features = ['loan_amnt', 'total_rec_int']
  
plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sb.scatterplot(data=LoanPayDF, x=col,
                   y='total_pymnt')
                
plt.show()


# Check outliers (by total_payment)
LoanPayDF.shape, LoanPayDF[LoanPayDF['total_pymnt']<5000].shape


#Remove the outliers
LoanPayDF = LoanPayDF[LoanPayDF['total_pymnt']<5000]


##plot (1) the relationship between dependent and independent variables 
features = ['loan_amnt', 'total_rec_int']
  
plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sb.scatterplot(data=LoanPayDF, x=col,
                   y='total_pymnt')
                
plt.show()



#Plot data again to the effects/improvements of removing outliers 
scatter, ax=plt.subplots()
ax=sns.regplot(x='loan_amnt', y='total_pymnt', data=LoanPayDF)
ax.set_title("Bank loan payment")
plt.xlabel('loan amount')
plt.ylabel('total_pymnt')
plt.show()


#normalize the data
LoanPayDF = pd.DataFrame(normalize (LoanPayDF), columns=LoanPayDF.columns)


LoanPayDF.head()

#define x and y variables 
x = LoanPayDF[['total_rec_int','loan_amnt', 'int_rate', 'LoanTerm', 'home_ownership']]
y = LoanPayDF['total_pymnt']


#split the data into train and test dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

#build the model sklearn
LoanPaymod = LinearRegression()
LoanPaymod.fit(x_train, y_train)


# create stastical summary.....
model = sm.OLS(y, x).fit()
print (model.summary ())

