#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np


# In[15]:


#let's import the dataset
import pandas as pd


# In[16]:


df = pd.read_csv('car data.csv')


# In[17]:


df.head()


# In[18]:


df.shape


# In[19]:


#there are many catagorival features categorical_features
#lets check the unique values in 'Sellr_Type' feature
print(df['Seller_Type'].unique())
#same for Transmission
print(df['Transmission'].unique())
#and for Fuel_Type
print(df['Fuel_Type'].unique())
#for Owner too
print(df['Owner'].unique())


# In[20]:


#lest check missing or null values
df.isnull().sum()


# In[21]:


#let's check some other informations
df.describe()


# ### how to deal with year feature

# year feature is very important for our prediction because , more the car old their depreciation is as more
# If we want the how many years old our car is we can subtfact that from 2020
# 

# In[22]:


df.columns


# In[28]:


#lest drop the car_name feature first , it not usefull for our prediction

final_dataset  = df.drop(['Car_Name'], axis = 1)


# In[29]:


#let's make a new feature name current year
final_dataset['current_year'] = 2020


# In[31]:


final_dataset['no_year'] = final_dataset['current_year']-final_dataset['Year']


# In[32]:


final_dataset.head()


# In[35]:


#we don't need current_year and year feature so we will drop it down below here
final_dataset.drop(['Year', 'current_year'], axis  = 1, inplace= True)


# In[37]:


final_dataset.head()


# ### Let's use one hote encoding techniques to get dummy variables 

# In[38]:


final_dataset = pd.get_dummies(final_dataset, drop_first = True) 


# In[39]:


final_dataset.head()


# In[40]:


#let's find out the correlation


# In[42]:


final_dataset.corr()


# In[43]:


import seaborn as sns


# In[46]:


sns.pairplot(final_dataset)


# ### let's understand it through heatmap

# In[48]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:




#get correlations of each features in dataset
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,8))
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[64]:


#split the dataset into independent and dependent features 
X = final_dataset.drop(['Selling_Price'], axis= 1)
# we can use iloc function too
y = final_dataset.iloc[:, 0] 


# In[67]:


X.head()


# In[68]:


y.head()


# ### let's check the feature importance

# In[69]:


### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[70]:


print(model.feature_importances_)


# In[74]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[75]:


#split the data into train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[76]:


X_train.shape


# ### use random_forest regressor 
# #### *** there is not scaring required in random_forest or decision_tree

# In[77]:


from sklearn.ensemble import RandomForestRegressor


# In[78]:


regressor=RandomForestRegressor()


# In[81]:


#n_estimators is basically the numbers of decision tries 
#we can use hyperparameter to choose n_estimators to get best result
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# #### there are two ways to find out hyperparameter tuning
# ##### (1) RandomizedSearchCV
# ##### (2) GreedsearchCV
# ### we are using RandomizedSearchCV here because it is faster than GreedsearchCV

# In[82]:


from sklearn.model_selection import RandomizedSearchCV


# In[84]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[85]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[86]:



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[87]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

#verbose is used to getting displayed the results
#n_jobs means how many cores of our laptop we want to use


# In[88]:


rf_random.fit(X_train,y_train)


# In[89]:


rf_random.best_params_


# In[90]:


rf_random.best_score_


# ### let's do some prediction

# In[93]:


predictions=rf_random.predict(X_test)
predictions


# ### let's plot our result usnig distplot

# In[94]:


sns.distplot(y_test-predictions)


# ##### we are getting a normal ddistribution graphn for the difference which shows that we are geeting well result, because as in above graphs the values are very less
# ###### according to graph most of the differences are around zero which is good

# 
# 
# 
# 

# In[95]:


plt.scatter(y_test,predictions)


# #### the points in scatter plot are linearlly available i.e our result is good

# ##### let's check accuracy

# In[97]:


from sklearn import metrics


# In[98]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# #### let's save our model in pickle formate for deployment

# In[99]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




