
###########  Importing Libraries ####################
import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
############  Loading Dataset ###############
df=pd.read_csv("https://raw.githubusercontent.com/martin22308/card/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO%20(3).csv")
print(df.head())

print("--------------------------------------------------------------------------------------")

############### Exploratory Some Information About Dataset ###############
#Here is the row count and type of each column
print(df.info())

print("--------------------------------------------------------------------------------------")

# # This code gives us the column and row information of the dataset.
print(df.shape)

print("--------------------------------------------------------------------------------------")

# # This code shows us some values ​​of numerical values ​​such as mean, standard deviation, minimum, maximum.
print(df.describe())

print("--------------------------------------------------------------------------------------")

# # This code gives us the number of null values ​​in the dataset (null or NaN) in each column
print(df.isnull().sum())

print("--------------------------------------------------------------------------------------")
# # This code gives us the names of the columns.
print(df.columns)

print("--------------------------------------------------------------------------------------")

# # This line of code is the separation process to identify the make of the car models in the column named
# # "name". The column named "name" is then saved as a new variable "name_2".
df["name_2"] = df.name.apply(lambda x : ' '.join(x.split(' ')[:1]))
print(df['name_2'])

print("--------------------------------------------------------------------------------------")

# ##################### Data Visualization  ############
df.name_2.value_counts()

sns.countplot(data=df,x="name_2",palette="CMRmap")
plt.xticks(rotation=90)
plt.xlabel("Name",fontsize=10,color="black")
plt.ylabel("Name",fontsize=10,color="black")
plt.title("NAME COUNT",color="black")
plt.show()

# # # From this chart I learned that The most common car model is the Maruti.
# # #  INFORMATION: Maruti Suzuki India Limited, formerly known
# # # as Maruti Udyog Limited, is an Indian automobile manufacturer, based in New Delhi.

print("--------------------------------------------------------------------------------------")


labels = df["name_2"][:30].value_counts().index
sizes = df["name_2"][:30].value_counts()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99',"pink","yellow"]
plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=45)
plt.title('name',color = 'red',fontsize = 15)
plt.show()

# # #  From this chart I learned that The most common car model is the Maruti. It is shown with pie table in this table.

print("--------------------------------------------------------------------------------------")

print(df.year.value_counts())

sns.countplot(data=df,x="year",palette="icefire")
plt.xticks(rotation=90)
plt.xlabel("YEAR",fontsize=10,color="RED")
plt.ylabel("COUNT",fontsize=10,color="RED")
plt.title("YEAR COUNT",color="RED")
plt.show()

labels = df["year"][:40].value_counts().index
sizes = df["year"][:40].value_counts()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99',"pink","yellow"]
plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=45)
plt.title('Year',color = 'red',fontsize = 15)
plt.show()

print("--------------------------------------------------------------------------------------")



# # In this table, it is seen that the most common type of fuel is diesel, followed by petroleum.
print(df.fuel.value_counts())
labels = df["fuel"].value_counts().index
sizes = df["fuel"].value_counts()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99',"pink","yellow"]
plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.f%%',colors=colors,shadow=True, startangle=9)
plt.title('fuel',color = 'blue',fontsize = 15)
plt.show()

print("--------------------------------------------------------------------------------------")

print(df.seller_type.value_counts())

sns.countplot(data=df,x="seller_type",palette="pink")
plt.xlabel("SELLER TYPE",fontsize=10,color="brown")
plt.ylabel("COUNT",fontsize=10,color="brown")
plt.title("SELLER TYPE COUNT",color="brown")
plt.show()
#From this chart I learned that the height year  in sales is 2014
# This is a table showing where sales are made. Individual sales are the highest.

print("--------------------------------------------------------------------------------------")


print(df.transmission.value_counts())

sns.countplot(data=df,x="transmission",palette="Spectral")
plt.xlabel("TRANMISSION",fontsize=10,color="GREEN")
plt.ylabel("COUNT",fontsize=10,color="GREEN")
plt.title("TRANMISSION COUNT",color="GREEN")
plt.show()
# It is the graphic that shows whether the cars are manual or automatic. Most are manual cars

print("--------------------------------------------------------------------------------------")

print(df.owner.value_counts())

sns.countplot(data=df,x="owner",palette="viridis")
plt.xticks(rotation=40)
plt.xlabel("OWNER",fontsize=10,color="purple")
plt.ylabel("COUNT",fontsize=10,color="purple")
plt.title("OWNER COUNT",color="purple")
plt.show()
# From this chart I learned that is the first hand or owner he used his car
# This chart shows us that sellers are selling more first hand.

print("--------------------------------------------------------------------------------------")

labels = df["owner"].value_counts().index
sizes = df["owner"].value_counts()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99',"pink","yellow"]
plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=905)
plt.title('owner',color = 'red',fontsize = 15)
plt.show()
# This pie chart shows us that sellers are selling more first hand.


print("--------------------------------------------------------------------------------------")


sns.histplot(data=df, x="year", hue="transmission")
plt.xticks(rotation=45)
plt.show()
# It is the histogram graph showing the distribution of manual and automatic cars by years.
# From this histogram I learned that in year 2015 it's the biggest users use  manual cars

print("--------------------------------------------------------------------------------------")

sns.violinplot(data=df, x="year", y="fuel",hue="transmission")
plt.show()
# It is the violin graph showing the distribution of manual and automatic cars by years.

print("--------------------------------------------------------------------------------------")


pd.crosstab(df["name_2"], df["transmission"]).plot(kind="bar", figsize=(10, 6), color=["blue","red"], title="name and transmission ")
plt.show()
# This table is a crosstab showing the automatic and manual distribution of car models.
# As usual, the car,maruti , is the highest percentage used manual

print("--------------------------------------------------------------------------------------")


pd.crosstab(df["name_2"], df["owner"]).plot(kind="bar", figsize=(10, 6), color=["purple","orange","lightgreen","blue","red"], title="name and owner ")
plt.show()
#This table is a graph of what car models are called first, second, third, four and above, or test vehicle.
# As usual, the car,maruti , is the highest percentage


fig, ax = plt.subplots(figsize=(20, 15))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df.corr(), annot=True, cmap="copper")
plt.show()


print("============================================================================")


################ Data Encoding ###########################
from sklearn import preprocessing
d_types=df.dtypes
for i in range(df.shape[1]):
     if d_types[i]=='object': Pr_data = preprocessing.LabelEncoder()
     df[df.columns[i]]= Pr_data.fit_transform(df[df.columns[i]])
print('data encoding :\n',df)

print("--------------------------------------------------------------------------------------")
# Data Scaling#############
scaler = preprocessing.MinMaxScaler()
Scaled_data = scaler.fit_transform(df)
Scaled_data = pd.DataFrame(Scaled_data,columns=df.columns)
print("data scaled: \n",Scaled_data)

print("--------------------------------------------------------------------------------------")

# # Data Correlation#######

r=Scaled_data.corr()
pd.plotting.scatter_matrix(r, figsize=(10, 10))
plt.show()

print("--------------------------------------------------------------------------------------")

# Cleaning the data for modelling

print(df.info())
print("--------------------------------------------------------------------------------------")

# #as mileage, engine, maxpower and unique and important parameters for prediction, rows which have these values empty should
print(df.dropna(inplace=True))
print("--------------------------------------------------------------------------------------")

# # To know how many duplicated rows:
print(df.duplicated().sum())
print("--------------------------------------------------------------------------------------")

# To delete duplicated rows:
print(df.shape)
df=df.drop_duplicates()
print(df.shape)
print("----------------------------------------------------------------------------")

################## Modelling polynomial regression   #################

y = df['selling_price']
x = df.drop(['selling_price'] , axis = 1)
print(x)
print(y)

print("--------------------------------------------------------------------------------------")

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.3)
# Transform input features into higher-order features using PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

 # Fit polynomial regression model on training data
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Get intercept and coefficients
intercept = model.intercept_
coef = model.coef_

# # Make predictions on test data
y_pred = model.predict(X_test_poly)

# # Calculate accuracy of model

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2_scor',r2)

print('a0 :',intercept )
print('0 , a1, a2 =',coef)


print("----------------------------------&&&&&&&&&&&&&&-----------------------------------------")



print("please do # before the last code ""polynomial regression"" to run next code \n and remove # from next code TO RUN")

# ############   model linear regression ########################

# X= df.drop(columns='selling_price')
# y=df['selling_price']
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# X= df.drop(columns='selling_price')
# y=df['selling_price']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
# from sklearn.linear_model import LinearRegression

# from sklearn.metrics import r2_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import make_column_transformer
# from sklearn.pipeline import make_pipeline 

# ohe = OneHotEncoder()
# ohe.fit(X[['year','km_driven','name','fuel','seller_type','transmission','owner']])
# ohe.categories_

# column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['year','km_driven','name','fuel','seller_type','transmission','owner']),remainder='passthrough')
# lr=LinearRegression()
# pipe=make_pipeline(column_trans,lr)
# pipe.fit(X_train,y_train)

# y_pred=pipe.predict(X_test)
# print('r2_score :',r2_score(y_test,y_pred))


# #LINEAR REGRESSION PLOT
# sns.regplot(x=y_test, y=y_pred)
# plt.xlabel("Predicted Price")
# plt.ylabel('Actual Price')
# plt.title("Actual vs predicted price")
# plt.show()

# import pickle
# pickle.dump(pipe,open('linearRegressionModel.pkl','wb'))
# m=pipe.predict(pd.DataFrame([['Hyundai Verna 1.6 SX',2016,50000,'Petrol','Individual','Manual','First Owner']], columns=['name','year','km_driven','fuel','seller_type','transmission','owner']))
# print("----------------------------------------------------------------     --------                -   - - - - - - - - - - --")
# print('price is :',m)

