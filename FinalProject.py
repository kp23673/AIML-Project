#!/usr/bin/env python
# coding: utf-8

# # MA336_2310628
# 

# # Project Name : OLA - Driver Sustain Ensemble

# # Introduction (Problem Statement) :
# 
# 

# OLA is leading transportation industry. Reducing drivers is seen by industry observers as a tough battle for Ola. Churn among drivers is high and it’s very easy for drivers to stop working for the service on the fly or jump to other transportation services depending on the rates.
# 
# As the companies get bigger, the high churn could become a bigger problem. To find new drivers, Ola is casting a wide net, including people who don’t have cars for jobs. But this acquisition is really costly. Losing drivers frequently impacts the morale of the organization and acquiring new drivers is more expensive than retaining existing ones.
# 
# You are working as a Data Scientist with the Analytics Department of Ola, focused on driver team attrition. You are provided with the monthly information for a segment of drivers for 2019 and 2020 and tasked to predict whether a driver will be leaving the company or not based on their attributes like
# 
# - Demographics (city, age, gender etc.)
# - Tenure information (joining date, Last Date)
# - Historical data regarding the performance of the driver (Quarterly rating, Monthly business acquired, grade, Income)

# In[1]:


# Importing all necessary Libraries to run data efficiently

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc


# # Dataset:

# In[2]:


# Importing Dataset

df = pd.read_csv("ola_driver.csv")


# In[3]:


df.head()


# In[4]:


# Removing Unamed column because there is no use of it

df.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)


# In[5]:


# Data Information

df.info()


# In[6]:


# Generating descriptive statistics

df.describe(include = 'all')


# # PRELIMINARY ANALYSIS:

# ## Filling Null values :
# 

# In[7]:


100*df.isnull().sum() / len(df)


# - In this we have 91.54% null values in last working days
# 
# - So, the values are not present in the columns which means they are not leaving the company so, we can fill it with the 0.
# 
# - In age column also we have missing values that are filled with the preceding values, same for gender also using ffill.

# In[8]:


df.head()


# ## Column Pre- Processing:
# 

# - In LastWorkingDate column we have a date of leaving that particular date is not needed as we need only the value that is 1 or 0.
# 
# - So, if we have that date in a row we fill that with 1 so that the driver is leaving that quarter.
# 
# - Here we can split the reporting MMM-YY to re_Day, re_Month, re_Year.
# 
# - Split the column date of joining to jo_Day, jo_Month, jo_Year.

# In[9]:


df['LastWorkingDate'].fillna(value=0, inplace=True)
df['LastWorkingDate'] = df['LastWorkingDate'].apply(lambda x: 0 if x == 0 else 1)

df['re_Month'] = df['MMM-YY'].apply(lambda x: int(str(x).split('/')[0]) )
df['re_Day'] = df['MMM-YY'].apply(lambda x: int(str(x).split('/')[1]) )
df['re_Year'] = df['MMM-YY'].apply(lambda x: int(str(x).split('/')[2]) )

df['city'] = df['City'].apply(lambda x: str(x)[1:] )

df['jo_Month'] = df['Dateofjoining'].apply(lambda x: int(str(x).split('/')[0]) )
df['jo_Day'] = df['Dateofjoining'].apply(lambda x: int(str(x).split('/')[1]) )
df['jo_Year'] = df['Dateofjoining'].apply(lambda x: int(str(x).split('/')[2]) )

df1 = df[['re_Day', 're_Month',
       're_Year','Driver_ID', 'Age', 'Gender', 'city', 'Education_Level',
       'Income', 'jo_Day', 'jo_Month', 'jo_Year', 'LastWorkingDate', 'Joining Designation',
       'Grade', 'Total Business Value', 'Quarterly Rating']]


# ## Knn Imputer:
# 

# In[10]:


imputer = KNNImputer(n_neighbors=2)
transformed = imputer.fit_transform(df1)
df2 = pd.DataFrame(transformed)

df2.rename(columns= {0:'re_Day', 1:'re_Month',2:'re_Year',3:'Driver_ID',4:'Age',5:'Gender',6:'City',7:'Education_Level',8:'Income',
                     9:'jo_Day',10:'jo_Month',11:'jo_Year',12:'LastWorkingDate',13:'JoiningDesignation',14:'Grade',
                     15:'TotalBusinessValue',16:'QuarterlyRating'}, inplace=True)
df2


# In[11]:


100*df2.isnull().sum() / len(df2)


# - Using KNN Imputer no null value is present.
# - By checking some columns using Imputer there is problem in Age and DriverID. 

# ## Checking KNN Imputer:
# 

# In[12]:


df[df['Age'].isnull()]


# In[13]:


df2[df2['Driver_ID'] == 22]


# - From this we come to know that KNN inputer is not working good (index 97 columns Age).
# 
# - In this case as we have a group of so we can determine the null vales but is not more accurate, so will fill all the null vales using the ffill in fillna.

# ## fillNa:

# In[14]:


df1['Age'].fillna(method= 'ffill', inplace=True)
df1['Gender'].fillna(method= 'ffill', inplace=True)

df1[df1['Driver_ID'] == 22]


# In[15]:


for i in df1.columns:
  print(i, '--->', df1[i].unique())


# # Analysis :
# 

# In[16]:


# Set the size of the figure
plt.figure(figsize=(8, 8))

# Your pie chart code
plt.pie(df1['LastWorkingDate'].value_counts(), labels=['Not leaving', 'Leaving'], explode=[0.1, 0.2], autopct='%1.1f%%')

plt.title('Driver Attrition Status')


# Display the pie chart
plt.show()


# - According to pie chart, we can see 8.5% are leaving the company whereas 91.5% are working.

# ## Univarient analysis : Numerical
# 

# In[17]:


uni_aly = ['Income', 'Total Business Value']
count = 0

plt.figure(figsize=(20,30))
for i in uni_aly:
    count += 1
    plt.subplot(5,3,count)
    sns.boxplot(y= df1[i])
    

plt.show()
    


# # 

# ## Univariate  Analysis : Categorical
# 

# In[18]:


uni_aly = ['Age','Gender', 'city','Education_Level','Joining Designation', 'Grade', 'Quarterly Rating']
count = 0
plt.figure(figsize=(20,30))
for i in uni_aly:
    count += 1
    plt.subplot(5,2,count)
    sns.barplot(x= df1[i], y= df1['LastWorkingDate'])


# - By univariant analysis. Depending on categorical law. Variables gender box that are equal.
# 
# - City are almost equal.
# 
# - Education level is also equal.
# 
# - Joining designation it is major for two and three.
# 
# - When comparing with the grades, it is most dependent on grade one and two. 345 or very less.
# 
# - In quarterly re-rating, so one is more dependent on last working date. And others are very less.

# # 

# ## Bivariate Analysis:
# 

# In[19]:


bi_ana = ['Age','Gender', 'city','Education_Level','Joining Designation', 'Grade', 'Quarterly Rating']
count = 0

plt.figure(figsize = (30,40))
for i in bi_ana:
    count+=1
    plt.subplot(5,2,count)
    sns.barplot(x= df1[i], y = df1['LastWorkingDate'], hue = df1['LastWorkingDate'])


# - From this bivarient analysis, we can see that the count of lastWorkingDate 0 is less so, we can't come to a conclusion with this we have to treat with the imbalance data.

# # 

# ## Correlation analysis:
#  

# In[20]:


plt.figure(figsize=(12, 10))
sns.heatmap(df1.corr(), annot=True, fmt='.1f', cmap="viridis", linewidth=.5)
plt.show()


# - From this correlation map. We can see that our target variable LastWorkingDate.
# 
# 
# - It does not even much more dependent on any other 15 columns.
# 
# 
# - So it is very difficult to find a correlation between them.

# ## Outlier Treatment :
# 

# In[21]:


outliers = ['Income', 'Total Business Value']
count = 0
plt.figure(figsize=(20,30))
for i in outliers:
    count += 1
    plt.subplot(5,3,count)
    sns.boxplot(y= df[i])


# In[22]:


for col in ['Income', 'Total Business Value']:
    
    mean = df[col].mean()
    std = df[col].std()
    q1 = np.percentile(df[col], 25)
    q2 = np.percentile(df[col], 50)
    q3 = np.percentile(df[col], 75)
    IQR = q3-q1
    lower_limt, upper_limit = q1-1.5*IQR , q3+1.5*IQR
    df[col] = df[col].apply(lambda x: lower_limt if x < lower_limt else x)
    df[col] = df[col].apply(lambda x: upper_limit if x > upper_limit else x)
df.shape


# In[23]:


outliers = ['Income', 'Total Business Value']
count = 0
plt.figure(figsize=(20,30))
for i in outliers:
    count += 1
    plt.subplot(5,3,count)
    sns.boxplot(y= df[i])


# - In this outliers, we will make the values that are more than Upper whisker and Lower whisker to inside the range.
# 
# - If we do that what happens is that all the values from total business value will compress and the mean value 0 will be shifted   from 0 to higher value so that will also affect the output.
# 
# - Even if we drop the null values then also we won't be left with more number of values.
# 
# - So in this case its better to not treat.

# ## EDA(Exploratory Data Analysis )/ FE(Feature Engineering):
# 

# In[24]:


df.columns


# In[25]:


df2 = df[['MMM-YY', 'Driver_ID', 'Age', 'Gender', 'City', 'Education_Level',
       'Income', 'Dateofjoining', 'LastWorkingDate', 'Joining Designation',
       'Grade', 'Total Business Value', 'Quarterly Rating']]

df2['City'] = df2['City'].apply(lambda x: int(str(x)[1:]) )
df2['MMM-YY'] = pd.to_datetime(df2['MMM-YY'])
df2['Dateofjoining'] = pd.to_datetime(df2['Dateofjoining'])
df2['TotalexpinDays'] = (df2['MMM-YY'] - df2['Dateofjoining']).dt.days


# GroupBy
# 
# - In MMM-YY we can see that the date are differing by exactly one month so that we can count in its month - month served.
# 
# - Dateofjoining will not change.
# 
# - Quarterly rating to mean reating(total rating/ total months).

# In[26]:


df3 = df2.groupby(by=['Driver_ID', "Gender", 'Dateofjoining', 'Joining Designation']).agg({'MMM-YY': 'count',
                                                                                          'Age' : 'max',
                                                                                          'City' : 'mean',
                                                                                          'Education_Level': 'max',
                                                                                          'Income': 'sum',
                                                                                          'LastWorkingDate': 'max',
                                                                                           'Grade': 'max',
                                                                                           'Total Business Value' : lambda x: list(x),
                                                                                           'Quarterly Rating': 'sum',
                                                                                           'TotalexpinDays':'max'}).reset_index()
df3.rename(columns={'MMM-YY' : 'TotalexpMonths', 'Income': 'tot_income'}, inplace=True)
df3['hasNegBusiValue'] = df3['Total Business Value'].apply(lambda x : 1 if min(x) < 0 else 0)
df3['totBusiValue'] = df3['Total Business Value'].apply(lambda x: sum(x))
df3['avg_income'] = df3['tot_income']/ df3['TotalexpMonths']


# In[27]:


df4 = df3[['Gender', 'Joining Designation',
       'TotalexpMonths', 'Age', 'City', 'Education_Level', 'tot_income','avg_income',
        'Grade', 'Quarterly Rating',
       'TotalexpinDays', 'hasNegBusiValue', 'totBusiValue', 'LastWorkingDate']]
df4


# In[28]:


plt.figure(figsize=(14,10))
sns.heatmap(df4.corr(), annot=True, fmt='.1f', cmap="Greens" , linewidth=.5)


# ## Encoding:

# 
# - Encoding is better to use here as it will work numeric for all categorical columns from range 0 to n.

# In[29]:


df4.head()


# In[30]:


df4.columns


# In[31]:


labelenc = LabelEncoder()

for i in ['Gender', 'Joining Designation', 'TotalexpMonths', 'Age', 'City','Education_Level', 'Grade','Quarterly Rating', 'hasNegBusiValue']:
  df4[i] = labelenc.fit_transform(df4[i])


# In[32]:


df4.head()


# - By label encoding, data looks better for further steps i.e Training and Testing. 

# # Methods:

# ## Train Test Split :
# 

# - Our data is finalised and we split the data for scalling and train our model : data without balancing.
# 

# ### Scalling

# In[33]:


X = df4.drop(columns=['LastWorkingDate'], axis=True)
y = df4['LastWorkingDate']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## Machine Learning Model with imbalance data:
# 

# ### Logistic Regression
# 

# In[35]:


model = LogisticRegression()
model.fit(X_train, y_train)

print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### KNN classifier
# 

# In[36]:


model = KNeighborsClassifier()
model.fit(X_train, y_train)

print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# Initialize empty lists to store training and testing scores
y_train_score = []
y_test_score = []

# Loop through different values of k
for i in range(1, 15):
    # Initialize the KNN classifier with i neighbors
    model = KNeighborsClassifier(n_neighbors=i)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Compute and store the training accuracy
    y_train_score.append(model.score(X_train, y_train))
    
    # Compute and store the testing accuracy
    y_test_score.append(model.score(X_test, y_test))
    
plt.figure(figsize=(10, 8))

# Plot the training and testing accuracy scores
sns.lineplot(range(1, 15), y_train_score, color='red', label='Training Accuracy')
sns.lineplot(range(1, 15), y_test_score, color='yellow', label='Testing Accuracy')

# Add a vertical dashed line at the index where the maximum testing accuracy occurs
best_k = y_test_score.index(max(y_test_score)) + 1  # Add 1 to convert index to k value
plt.axvline(x=best_k, linestyle='--', color='green', label=f'Best k = {best_k}')

# Set plot labels and title
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors Classifier Accuracy')
plt.legend()
plt.show()


# In[38]:


max(y_test_score)


# In[39]:


pip install pydot


# In[40]:


pip install pyparsing pydot


# ## Decision Tree Classifier:
# 

# In[41]:


features = list((df4.drop(columns=['LastWorkingDate'])).columns)

model = DecisionTreeClassifier(criterion = 'gini')
model.fit(X_train, y_train)

print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

dot_data = StringIO()
export_graphviz(model, out_file=dot_data, feature_names=features, filled=True)




# ## RandomForestClassifier <-- Selected Model:
# 

# In[42]:


model = RandomForestClassifier()
model.fit(X_train, y_train)

print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[43]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize empty lists to store training and testing scores
y_train_score = []
y_test_score = []

# Loop through different values of max_depth
for i in range(1, 100):
    # Initialize the Random Forest classifier with max_depth=i
    model = RandomForestClassifier(max_depth=i, random_state=42)  # random_state for reproducibility
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Compute and store the training accuracy
    y_train_score.append(model.score(X_train, y_train))
    
    # Compute and store the testing accuracy
    y_test_score.append(model.score(X_test, y_test))
    

plt.figure(figsize=(10,8))
# Plot the training and testing accuracy scores
sns.lineplot(range(1, 100), y_train_score, color='blue', label='Training Accuracy')
sns.lineplot(range(1, 100), y_test_score, color='orange', label='Testing Accuracy')

# Add a vertical dashed line at the index where the maximum testing accuracy occurs
best_max_depth = y_test_score.index(max(y_test_score)) + 1  # Add 1 to convert index to max_depth value
plt.axvline(x=best_max_depth, linestyle='--', color='red', label=f'Best Max Depth = {best_max_depth}')

# Set plot labels and title
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest Classifier Accuracy')
plt.legend()
plt.show()


# ## Conclusion:

# - From this we come to know that all the 3 algorithms are good but not enough for modelling, will do the following precess: 
# 
# - 1. Balance data
# - 2. Bagging
# - 3. Boosting algorithms
# 
# - We have all the precission value to be around 80% only, but we need above 90 atleast.

# ## Machine Learning Model with balanced data and comparision:
#  

# In[44]:


pip install imbalanced-learn


# In[45]:


from imblearn.over_sampling import SMOTE


# In[46]:


smt = SMOTE()

print('Before SMOTE')
print(y_train.value_counts())

x_sm, y_sm = smt.fit_resample(X_train, y_train)
print('\nAfter SMOTE')
print(y_sm.value_counts())


# - The purpose of this code is to demonstrate how the class distribution changes before and after applying SMOTE, which is a technique commonly used to address class imbalance in classification problems.
# - By balancing the classes, it helps improve the performance of machine learning models, especially when dealing with imbalanced datasets.

# ## Logistic Regression
# 

# In[47]:


model = LogisticRegression()
model.fit(X_train, y_train)
print('for Normal Data')
print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print('\n',classification_report(y_test, y_pred))

model = LogisticRegression()
model.fit(x_sm, y_sm)
print('for Balanced Data')
print("training score",model.score(x_sm, y_sm))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print('\n',classification_report(y_test, y_pred))


# ## KNN classification
# 

# In[48]:


model = KNeighborsClassifier()
model.fit(X_train, y_train)
print('for Normal Data')
print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print('\n',classification_report(y_test, y_pred))

model = KNeighborsClassifier()
model.fit(x_sm, y_sm)
print('for Balanced Data')
print("training score",model.score(x_sm, y_sm))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print('\n',classification_report(y_test, y_pred))


# In[49]:


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize empty lists to store training and testing scores
y_train_score = []
y_test_score = []

# Loop through different values of k
for i in range(1, 15):
    # Initialize the KNN classifier with i neighbors
    model = KNeighborsClassifier(n_neighbors=i)
    
    # Train the model on the balanced training data (after SMOTE)
    model.fit(x_sm, y_sm)
    
    # Compute and store the training accuracy on the balanced data
    y_train_score.append(model.score(x_sm, y_sm))
    
    # Compute and store the testing accuracy on the original testing data
    y_test_score.append(model.score(X_test, y_test))
    
plt.figure(figsize = (10,8))    

# Plot the training and testing accuracy scores
sns.lineplot(range(1, 15), y_train_score, color='purple', label='Training Accuracy (SMOTE)')
sns.lineplot(range(1, 15), y_test_score, color='orange', label='Testing Accuracy (Original)')

# Add a vertical dashed line at the index where the maximum testing accuracy occurs
best_k = y_test_score.index(max(y_test_score)) + 1  # Add 1 to convert index to k value
plt.axvline(x=best_k, linestyle='--', color='brown', label=f'Best k = {best_k}')

# Set plot labels and title
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors Classifier Accuracy with SMOTE')
plt.legend()
plt.show()


# ## DecisionTree Classifier
# 

# In[50]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print('for Normal Data')
print("training score",model.score(X_train, y_train))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print('\n',classification_report(y_test, y_pred))

model = DecisionTreeClassifier()
model.fit(x_sm, y_sm)
print('for Balanced Data')
print("training score",model.score(x_sm, y_sm))
print("test score",model.score(X_test, y_test))

y_pred = model.predict(X_test)
print('\n',classification_report(y_test, y_pred))


# In[51]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

y_train_score = []
y_test_score = []

max_depth_range = range(1, 21)  # Choose a range of max_depth values

for depth in max_depth_range:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_sm, y_sm)
    y_train_score.append(model.score(x_sm, y_sm))
    y_test_score.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 8))
sns.lineplot(max_depth_range, y_train_score, color='blue', label='Training Accuracy')
sns.lineplot(max_depth_range, y_test_score, color='orange', label='Testing Accuracy')
best_max_depth = y_test_score.index(max(y_test_score)) + 1
plt.axvline(x=best_max_depth, linestyle='--', color='red', label=f'Best Max Depth = {best_max_depth}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Classifier Accuracy')
plt.legend()
plt.show()


# ## Hyperparameter Tunning :
# 

# - For hyperparameter tunning we use random forest with its hyperparameters.
# 
# 

# In[52]:


model_rfc = RandomForestClassifier(criterion='gini', n_jobs=-1)
model_rfc.fit(X_train, y_train)


# In[53]:


print('train score',model_rfc.score(X_train, y_train))
print('test score',model_rfc.score(X_test, y_test))


# In[54]:


model_rfc.feature_importances_


# - From this we can see that the train and test score are in a big difference means we have a overfit model.
# 
# - We have to get rid of this overfit model.

# In[55]:


hyp_prams = {
    "n_estimators": [100,200,300,400,500],
    "max_depth" : [10, 20, 30,40,50,60,70,80,90,100]
}

rfc = RandomForestClassifier(criterion='gini', n_jobs=-1)

# model_hyp = GridSearchCV(rfc, hyp_prams)
model_hyp = RandomizedSearchCV(rfc, hyp_prams)
model_hyp.fit(X_train, y_train)

print(model_hyp.best_params_)


# In[56]:


model_rfc = RandomForestClassifier(criterion='gini', n_jobs=-1, **model_hyp.best_params_)
model_rfc.fit(X_train, y_train)

print('train score',model_rfc.score(X_train, y_train))
print('test score',model_rfc.score(X_test, y_test))


# In[57]:


y_pred = model_rfc.predict(X_test)
cr = classification_report(y_test, y_pred)
print('cm', cr)
cm = confusion_matrix(y_test, y_pred)
print('cm', cm)


# ## Bagging :
#  

# In[58]:


model_baga = BaggingClassifier()
model_baga.fit(X_train, y_train)


# In[59]:


print('train score',model_baga.score(X_train, y_train))
print('test score',model_baga.score(X_test, y_test))
y_pred = model_baga.predict(X_test)
cr = classification_report(y_test, y_pred)
print('cm', cr)
cm = confusion_matrix(y_test, y_pred)
print('cm', cm)


# ## Boosting : <-- Best Model:
# 

# ### XGBoost

# In[60]:


pip install xgboost


# In[61]:


from xgboost import XGBClassifier
import xgboost
model_bost = XGBClassifier()
model_bost.fit(X_train, y_train)


# In[62]:


print('train score',model_bost.score(X_train, y_train))
print('test score',model_bost.score(X_test, y_test))
y_pred = model_bost.predict(X_test)
cr = classification_report(y_test, y_pred)
print('cm', cr)
cm = confusion_matrix(y_test, y_pred)
print('cm', cm)


# ## Hyperparmeter tunning
# 

# In[63]:


params = {
        "n_estimators": [150,200, 250, 300],
        "max_depth" : [2, 3, 4, 5, 7],
        "learning_rate": [0.01, 0.02, 0.05, 0.07],
        'subsample': [0.4, 0.5,0.6, 0.8],
        'colsample_bytree': [0.6, 0.8, 1.0],
        }

xgb = XGBClassifier(objective='multi:softmax', num_class=20, silent=True)
random_search = RandomizedSearchCV( xgb, param_distributions = params,scoring='accuracy',n_jobs=-1,cv=3)

random_search.fit(X_train, y_train)


# In[64]:


random_search.best_params_


# In[65]:


xgb = XGBClassifier(**random_search.best_params_ , num_classes=20)
xgb.fit(X_train, y_train)
print('train score',xgb.score(X_train, y_train))
print('test score',xgb.score(X_test, y_test))
y_pred = xgb.predict(X_test)
cr = classification_report(y_test, y_pred)
print('cm', cr)
cm = confusion_matrix(y_test, y_pred)
print('cm', cm)


# - The best Model for training and it precession is good and all points are discovered.
# 
# - Model has fit into the Prefect-model , No-Overfir or No-Underfit.

# In[66]:


plt.figure(figsize = (10,8))
ax = sns.heatmap(cm, annot=True, cmap="crest", fmt='g')
ax.set(xlabel="Actual", ylabel="Preducted")
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()


# - Here we have predicted 299 events that are actually 1. We also predicted as 1. This is changeable.
# 
# - But actual 1 and we predicted 0 which has account of 45.

# In[67]:


prob = (xgb.predict_proba(X_test))[:,1]
fpr, tpr, thr = roc_curve(y_test, prob)
plt.figure(figsize = (10,8))

plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'--',color='red' )
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

print('\nAUC-ROC score : ',roc_auc_score(y_test,prob))


# In[68]:


precision, recall, thr = precision_recall_curve(y_test, prob)

plt.figure(figsize = (10,8))

plt.plot(recall) #blue
plt.plot(precision) # orange
plt.plot(thr) # green

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precession-Recall curve')
plt.show()


# - We have a good P-R curve but its ok to have a curve like this.
# 
# 

# # Results :

# - Before diving into specific recommendations, it's essential to understand what ensemble learning is and how it works. Ensemble learning combines multiple machine learning models to improve prediction accuracy and robustness over individual models. There are various ensemble methods such as bagging, boosting, and stacking. Preprocessing:
# 
# - From the analysis and feature selections, we get the idea how much driver are working and leaving.
# 
# - Encoding used for data efficiency.
# 
# - Played with Imbalanced and Balanced data using  Logistic Regression, KNN classifier, DecisionTree Classifier, RandomForest,     Hyperparameter Tuning, Bagging Boosting to see which algorithm is good for this data.
# 
# - By doing Comparison between balanced and imbalanced data we get know that balanced is good compared to imbalanced. By ROC curve we  get know balanced data is about 90% which is better. 
# 
# - Experiment with different types of algorithms to capture diverse patterns in the data.
# 
# - Perform hyperparameter tuning for both individual models and ensemble methods to optimize their performance.
# 
# - Use techniques like grid search or random search to efficiently search the hyperparameter space.

# # 

# # Conclusion :

# - In conclusion, the analysis of driver team attrition at Ola presents several key findings and recommendations.
# 
# - Firstly, the high churn rate among drivers poses a significant challenge for the company, impacting morale and incurring substantial costs associated with driver acquisition. 
# 
# - Through the examination of monthly data for 2019 and 2020, it is evident that demographic factors such as age, gender, and city, alongside tenure information and historical performance metrics, play crucial roles in predicting driver attrition. 
# 
# - Leveraging ensemble learning techniques such as bagging and boosting, as well as KNN imputation for handling missing values, proves to be effective in developing predictive models for identifying drivers at risk of leaving the company.
# 
# - Additionally, given the imbalanced nature of the dataset, strategies for working with imbalanced data, such as oversampling or incorporating class weights, are essential for achieving accurate predictions. 
# 
# - Moving forward, Ola can use these insights to implement targeted retention strategies, focusing on factors identified as significant predictors of attrition. 
# 
# - By addressing the root causes of driver churn and prioritizing the retention of existing drivers, Ola can mitigate the adverse effects of high turnover rates and ensure the stability and sustainability of its driver workforce.
# 
