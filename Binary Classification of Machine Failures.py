#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[165]:


import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ## Data Importing

# In[166]:


df = pd.read_csv("C:/Machine Learning/Predictive Maintenence/train/train.csv")
df_test = pd.read_csv("C:/Machine Learning/Predictive Maintenence/test/test.csv")


# ## Exploratory Data Analysis(EDA)

# In[167]:


df.head()


# In[168]:


df_test.head()


# Checking for null values

# In[169]:


df.info()


# In[170]:


df_test.info()


# In[171]:


df.describe(include='all').T


# In[172]:


display(df.columns.tolist())


# In[173]:


pd.DataFrame(data= {'Number': df['Machine failure'].value_counts(), 
                    'Percent': df['Machine failure'].value_counts(normalize=True)})


# In[174]:


display(df.nunique())


# Drop the indices as these have no predictive power

# In[175]:


train = df.copy()


# In[176]:


train.pop("Machine failure")


# In[177]:


train.pop('id')


# In[178]:


features = train.columns.tolist()

# Categorical features
cat_features = ['Product ID', 'Type']

# Binary features
bin_features = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Numerical features
num_features = [f for f in features if f not in (cat_features + bin_features)]

print('Number of Categorical_features:', len(cat_features)) 
print('Number of Binary_features:', len(bin_features))  
print('Number of Numerical_features:', len(num_features))
print(num_features)
print("")
print('The total number of features:', len(features))
print(features)


# Checking the percentages of values in columns 

# In[179]:


for f in cat_features:
    print('\t' , f)
    n_f = train[f].value_counts()
    p_f = train[f].value_counts(normalize=True)
    display(pd.DataFrame(data= {'Number': n_f, 'Percent': p_f}))


# In[180]:


dfv = pd.DataFrame(data= {'Value': ['Number 0', 'Percent 0', '', 'Number 1', 'Percent 1']})

for f in bin_features: 
    n_f = train[f].value_counts()
    p_f = train[f].value_counts(normalize=True)
    dfv[f] = [n_f[0], p_f[0], '', n_f[1], p_f[1]]
    
dfv.set_index('Value')


# In[181]:


sns.set_theme(style='whitegrid')
plt.figure(figsize=(20,30))
plotnumber = 1

for column in num_features:
    if plotnumber<=(len(num_features)): 
        ax = plt.subplot(5,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column, fontsize=12)
    plotnumber +=1
plt.show


# #### Some columns in the dataset are normally distributed("Rotational speed [rpm]", "Torque[Nm]")

# In[182]:


#function to visualize the binary machine failures
def plot_binary_machine_failures(dataframe, column):
    # Plot the machine failures
    plt.figure(figsize=(8, 5))

    # Countplot for Machine failure
    ax = plt.subplot(1, 2, 1)
    ax = sns.countplot(x=column, data=dataframe)
    ax.bar_label(ax.containers[0])
    plt.title(column + " Failure", fontsize=20)

    # Pie chart for Outcome
    ax = plt.subplot(1, 2, 2)
    outcome_counts = dataframe[column].value_counts()
    ax = outcome_counts.plot.pie(explode=[0.1, 0.1], autopct='%1.2f%%', shadow=True)
    ax.set_title("Outcome", fontsize=12, color='Red', font='Lucida Calligraphy')

    # Display the plot
    plt.tight_layout()
    plt.show()


# In[183]:


# Visualize the machine failure
plot_binary_machine_failures(df, 'Machine failure')


# In[184]:


plot_binary_machine_failures(df, 'TWF')


# In[185]:


plot_binary_machine_failures(df, 'HDF')


# In[186]:


plot_binary_machine_failures(df, 'PWF')


# In[187]:


plot_binary_machine_failures(train, 'OSF')


# In[188]:


plot_binary_machine_failures(train, 'RNF')


# #### From the above analysis, it can be seen that the dataset is highly imbalanced.

# In[189]:


plt.figure(figsize=(7,5))
threshold = 0.35
sns.set_style("whitegrid", {"axes.facecolor": ".0"})
df_cluster = df.corr()
mask = df_cluster.where((abs(df_cluster) >= threshold)).isna()
plot_kws={"s": 1}
sns.heatmap(df_cluster,
            cmap='RdYlBu',
            annot=True,
            mask=mask,
            linewidths=0.2, 
            linecolor='lightgrey').set_facecolor('white')


# From the graph, we can see there are strongly correlated features namely "Process temperature and Air temperature" and  "Torque and rotational speed".

# ## Feature Engineering

# In[190]:


### normalizing the data

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()

df[num_features] = scaler.fit_transform(df[num_features])
df_test[num_features] = scaler.fit_transform(df_test[num_features])


# #### Encoding

# In[191]:


df_1 = df.drop(['Product ID'] , axis=1)
feature = ['Type']
df_1 = pd.get_dummies(df_1, columns=feature)
df_1.head()


# In[192]:


df_test1 = df_test.drop(['Product ID','id'] , axis=1)
feature = ['Type']
df_test1 = pd.get_dummies(df_test1, columns=feature)
df_test1.head()


# ## Modelling and  Evaluation

# In[193]:


X = df_1.drop(columns= ['Machine failure'],axis = 1)
Y = df_1['Machine failure']


# In[194]:


from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train,y_test = train_test_split(X,Y,test_size = 0.20,random_state = 70)

x_train = x_train1.drop('id', axis=1)
x_test = x_test1.drop('id', axis=1)
x_train.shape, y_test.shape


# ### Model Building

# In[195]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier


# In[196]:


model1 = RandomForestClassifier()
model2 = DecisionTreeClassifier()
model3 = SVC(kernel='linear', probability=True, random_state=42)
model4 = CatBoostClassifier(learning_rate=0.1, iterations=500, random_seed=42)


# In[197]:


##RandomForestClassifier
# training the model 
model1.fit(x_train, y_train)

# Preprocessing of validation data, get predictions
preds1 = model1.predict(x_test)
# Evaluatung the model
accuracy_score(y_test, preds1)


# In[198]:


##DecisionTreeClassifier
model2.fit(x_train, y_train)

# Preprocessing of validation data, get predictions
preds2 = model2.predict(x_test)
# Evaluatung the model
accuracy_score(y_test, preds2)


# In[199]:


##Support Vector Machines
model3.fit(x_train, y_train)

# Preprocessing of validation data, get predictions
preds3 = model3.predict(x_test)
# Evaluatung the model
print(accuracy_score(y_test, preds3))
print("Train AUC :", roc_auc_score(y_test, model3.predict_proba(x_test)[:,1]))


# In[200]:


## CatBoost
model4.fit(x_train, y_train,
          eval_set=(x_test, y_test),
          verbose=False)


# In[202]:


# Making predictions on the test set
preds4 = model4.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, preds4)
print(f'Accuracy: {accuracy}')
print("Train AUC :", roc_auc_score(y_test, model4.predict_proba(x_test)[:,1]))


# ### Visualising the best classifier by comparing accuracy and AUC i.e, CatBoost

# Creating a confusion matrix to visualize the performance of  catboost. A confusion matrix is a performance evaluation tool in machine learning, representing the accuracy of a classification model. It displays the number of true positives, true negatives, false positives, and false negatives.

# In[203]:


# Calculating confusion matrix
cm = confusion_matrix(y_test, preds4)

# Plotting confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# For binary classification, where you have two classes (e.g., positive and negative), a common and effective way to visualize the performance of your model is by using a Receiver Operating Characteristic (ROC) curve and an Area Under the Curve (AUC) plot. These plots provide insights into the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity) at different classification thresholds.

# In[204]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[205]:


# Getting predicted probabilities for the positive class
y_prob = model4.predict_proba(x_test)[:, 1]
y_prob


# In[206]:


# Calculating false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculating AUC score
auc_score = roc_auc_score(y_test, y_prob)


# In[207]:


auc_score


# Plotting the ROC curve to visualize the trade-off between true positive rate and false positive rate at different classification thresholds.

# This code will generate an ROC curve plot with the AUC score displayed in the legend. The diagonal dashed line represents random guessing, and a higher AUC score indicates better model performance.

# In[208]:


# Plotting ROC curve
sns.set_theme(style = 'whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Guessing')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[209]:


## Predicting on the Original test set
predictions = model4.predict_proba(df_test1)[:, 1]

columns = ['id', 'Machine Failure']
values = [df_test['id'], predictions]
submission_df=pd.DataFrame(dict(zip(columns, values)))
submission_df.to_csv( 'submission_base.csv' ,index=False,header=True)
print(submission_df.shape)


# ## Hyperparameter Tuning

# In[210]:


from sklearn.model_selection import RandomizedSearchCV

model = CatBoostClassifier()
# Defining the hyperparameter grid for RandomizedSearchCV
param_grid = {
    'iterations': [500,600],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 6, 8,10],
    'loss_function': ['Logloss']
}

# Initializing RandomizedSearchCV with AUC as the scoring metric
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, scoring='roc_auc', cv=5, verbose=0, random_state=42)

# Performing the hyperparameter search
random_search.fit(x_train, y_train)

# The best hyperparameters and best AUC score
best_params = random_search.best_params_
best_auc = random_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best AUC Score:", best_auc)

# Evaluating the best model on the test set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(x_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print("Test AUC Score:", test_auc)


# In[211]:


# Initializing the CatBoost classifier with best hyperparameters
model_best = CatBoostClassifier(**best_params)

# Training the final model on the training data
model_best.fit(x_train, y_train)

# Making predictions on test data 
y_pred = model_best.predict(x_test)

# Evaluating the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[212]:


## Predicting on the Original test set 
prediction_best = model_best.predict_proba(df_test1)[:, 1]

columns = ['id', 'Machine Failure']
values = [df_test['id'], prediction_best]
submission_df=pd.DataFrame(dict(zip(columns, values)))
submission_df.to_csv( 'submission_final.csv' ,index=False,header=True)
print(submission_df.shape)

