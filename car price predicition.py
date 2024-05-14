#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor



# In[2]:


df = pd.read_csv(r"C:\Users\Parool\Downloads\car data (1).csv")
df.info()


# In[3]:


df


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()



# In[8]:


df.shape


# In[9]:


df.sample(4)


# In[12]:


numeric_columns = df[['Year', 'Selling_Price', 'Present_Price', 
'Kms_Driven','Owner']]
correlation_matrix = numeric_columns.corr()


# In[13]:


correlation_matrix


# In[14]:


plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', fmt=".2f", 
linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[15]:


#top_car
top_car = df['Car_Name'].value_counts().nlargest(10)


# In[16]:


plt.figure(figsize = (8, 6))
sns.countplot(y = df.Car_Name, order=top_car.index, palette='viridis')
plt.title("Top 10 Car companies with their cars", fontsize = 10)
plt.show()


# In[17]:


sns.countplot(y = df.Car_Name, order=top_car.index, 
palette='viridis')


# In[18]:


df['Fuel_Type'].value_counts()


# In[19]:


sns.countplot(x=df['Fuel_Type'],hue=df['Fuel_Type'], 
palette='viridis')


# In[22]:


df['Selling_Price'].value_counts()


# In[24]:


sns.countplot(x=df['Selling_Price'],hue=df['Selling_Price'], 
palette='viridis')


# In[25]:


df['Transmission'].value_counts()


# In[26]:


df['Owner'].value_counts()


# In[27]:


sns.boxplot(x=df['Selling_Price'])


# In[28]:


percentile_75 = np.percentile(df['Selling_Price'],75)
sum(df['Selling_Price']>percentile_75)


# In[29]:


sns.histplot(df['Selling_Price'])


# In[30]:


plt.figure(figsize = (8,6))
sns.countplot(y=df['Year'],palette = 'viridis')
plt.title('How old the car is?')
plt.show()


# In[31]:


sns.countplot(y=df['Year'],palette = 'viridis')


# In[32]:


sns.scatterplot(x=df['Selling_Price'],y=df['Kms_Driven'])


# In[33]:


df.select_dtypes(include=['object']).columns


# In[34]:


label_encoder = LabelEncoder()
# Convert the categorical columns to numerical using LabelEncoder
df['Car_Name'] = label_encoder.fit_transform(df['Car_Name'])
df['Fuel_Type'] = label_encoder.fit_transform(df['Fuel_Type'])
df['Selling_Price'] = label_encoder.fit_transform(df['Selling_Price'])
df['Transmission'] = label_encoder.fit_transform(df['Transmission'])
df.head()


# In[35]:


# Select features (X) and target variable (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=35)


# In[38]:


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[39]:


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
linear_reg_predictions = linear_reg.predict(X_test_scaled)


# In[40]:


# Decision Tree Regressor
decision_tree_reg = DecisionTreeRegressor(random_state=42)
decision_tree_reg.fit(X_train_scaled, y_train)
decision_tree_predictions = decision_tree_reg.predict(X_test_scaled)


# In[41]:


# Random Forest Regressor
random_forest_reg = RandomForestRegressor(n_estimators=100, 
random_state=42)
random_forest_reg.fit(X_train_scaled, y_train)
random_forest_predictions = random_forest_reg.predict(X_test_scaled)


# In[42]:


# XGBoost Regressor
xgboost_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
colsample_bytree = 0.3, learning_rate = 0.1,
 max_depth = 5, alpha = 10, n_estimators
= 100, random_state=42)
xgboost_reg.fit(X_train_scaled, y_train)
xgboost_predictions = xgboost_reg.predict(X_test_scaled)


# In[43]:


# MLP Regressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, 
random_state=42)
mlp_reg.fit(X_train_scaled, y_train)
mlp_predictions = mlp_reg.predict(X_test_scaled)


# In[46]:


# Evaluate the models
models = {'Linear Regression': linear_reg, 'Decision Tree Regressor': 
decision_tree_reg,'Random Forest Regressor': random_forest_reg, 'XGBoost Regressor': xgboost_reg,'MLP Regressor': mlp_reg}
for name, model in models.items():
 predictions = model.predict(X_test_scaled)
 mse = mean_squared_error(y_test, predictions)
 r2 = r2_score(y_test, predictions)
 print(f'{name} - Mean Squared Error: {mse}, R-squared: {r2}')


# In[51]:


models = ['Linear Regression', 'Decision Tree', 'Random Forest', 
'XGBoost', 'MLP']
mse_scores = [6.51, 2.27, 3.48, 7.16, 1.49]
r2_scores = [0.77, 0.92, 0.88, 0.75, 0.95]
# Create a DataFrame for easy plotting
performance_df = pd.DataFrame({'Model': models, 'MSE': mse_scores, 'Rsquared': r2_scores})
# Plotting
plt.figure(figsize=(12, 6))


# In[52]:


# Bar plot for MSE
plt.subplot(1, 2, 1)
sns.barplot(x='MSE', y='Model', data=performance_df, 
palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.xlabel('MSE')


# In[55]:


# Bar plot for R-squared
plt.subplot(1, 2, 2)
sns.barplot(x='R-squared', y='Model', data=performance_df, palette='viridis')
plt.title('R-squared Score')
plt.xlabel('R-squared')
plt.tight_layout()
plt.show()


# In[56]:


sns.barplot(x='MSE', y='Model', data=performance_df, 
palette='viridis')

