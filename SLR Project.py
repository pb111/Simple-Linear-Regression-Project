
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

# The above command sets the backend of matplotlib to the 'inline' backend. 
# It means the output of plotting commands is displayed inline.


# In[2]:


# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# Import the data

url = "C:/project_datasets/SALES.txt"
df = pd.read_csv(url, sep='\t', header=None)


# In[4]:


# Exploratory data analysis

# View the dimensions of df

print(df.shape)


# In[5]:


# View the top 5 rows of df

print(df.head())


# In[6]:


# Rename columns of df dataframe

df.columns = ['Sales', 'Advertising']


# In[7]:


# View the top 5 rows of df with column names renamed

print(df.head())


# In[8]:


# View dataframe summary

print(df.info())


# In[9]:


# View descriptive statistics

print(df.describe())


# In[10]:


# Declare feature variable and target array

X = df['Sales'].values
y = df['Advertising'].values

# Sales and Advertising data values are given by X and y respectively.

# Values attribute of pandas dataframe returns the numpy arrays.


# In[11]:


# Print the dimensions of X and y

print(X.shape)
print(y.shape)


# In[12]:


# Reshape X and y

X = X.reshape(-1,1)
y = y.reshape(-1,1)

# Since we are working with only one feature variable, so we need to do reshaping using NumPy's reshape() method. 
# It specifies first dimension to be -1, which means "unspecified". 
# Its value is inferred from the length of the array and the remaining dimensions. 
    


# In[13]:


# Print the dimensions of X and y after reshaping

print(X.shape)
print(y.shape)

# We can see the difference in dimensions of X and y before and after reshaping.
# It is essential in this case because getting the feature and target variable arrays into the right format for scikit-learn 
# is an important precursor to model building.


# In[14]:


# Visualizing the relationship between X and y by scatterplot

# Plot scatter plot between X and y

plt.scatter(X, y, color = 'blue', label='Scatter Plot')
plt.title('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()


# In[15]:


# Split X and y into training and test data sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


# Print the dimensions of X_train,X_test,y_train,y_test

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[17]:


# Fit the linear model

# Instantiate the linear regression object lm
from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# Train the model using training data sets
lm.fit(X_train,y_train)


# Predict on the test data
y_pred=lm.predict(X_test)


# In[18]:


# Calculate and print Root Mean Square Error(RMSE)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE value: {:.4f}".format(rmse))


# In[19]:


# Calculate and print r2_score

from sklearn.metrics import r2_score
print ("R2 Score value: {:.4f}".format(r2_score(y_test, y_pred)))


# In[20]:


# Compute model slope and intercept

a = lm.coef_
b = lm.intercept_,
print("Estimated model slope, a:" , a)
print("Estimated model intercept, b:" , b) 


# In[21]:


# So, our fitted regression line is 

# y = 1.60509347 * x - 11.16003616 

# That is our linear model.


# In[22]:


# Predicting Advertising values

lm.predict(X)[0:5]

# Predicting Advertising values on first five Sales values.


# In[23]:


# To make an individual prediction using the linear regression model.

print(str(lm.predict(24)))


# In[24]:


# Plot the Regression Line


plt.scatter(X, y, color = 'blue', label='Scatter Plot')
plt.plot(X_test, y_pred, color = 'black', linewidth=3, label = 'Regression Line')
plt.title('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()


# In[25]:


# Plotting residual errors

plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, color = 'red', label = 'Train data')
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, color = 'blue', label = 'Test data')
plt.hlines(xmin = 0, xmax = 50, y = 0, linewidth = 3)
plt.title('Residual errors')
plt.legend(loc = 4)
plt.show()


# In[26]:


# Checking for Overfitting or Underfitting the data

print("Training set score: {:.4f}".format(lm.score(X_train,y_train)))

print("Test set score: {:.4f}".format(lm.score(X_test,y_test)))


# In[27]:


# Save model for future use

from sklearn.externals import joblib
joblib.dump(lm, 'lm_regressor.pkl')

# To load the model

# lm2=joblib.load('lm_regressor.pkl')

