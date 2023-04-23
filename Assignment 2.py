#!/usr/bin/env python
# coding: utf-8

# # Import necessary python library
#   

# In[1]:


import zipfile
import pandas as pd
import numpy as np
from zipfile import ZipFile


# # specifying the zip file name

# In[2]:


file_name = "Stress_dataset.zip"
  


# In[3]:


# Extract the files from the zipped folder
with zipfile.ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()

# extracting all the files
print('Extracting all the files now...')
print('Done!')


# # Extract the data  from the zipped folder

# In[4]:


data=pd.read_excel('SurveyResults.xlsx')


# # Print the first 5 rows of the dataset

# In[5]:


print('--- First 5 rows of the dataset ---')
print(data.head())


# # Explore the dataset by checking its shape, data types, and basic statistics:

# In[6]:


print('\n--- Shape of the dataset ---')
print(data.shape)


# In[7]:


print(data.dtypes)


# In[8]:


# print(data.describe())
print(data.describe(datetime_is_numeric=True))


# In[9]:


data.drop('Lack of supplies', axis=1, inplace=True)


# # check for missing values 

# # drop rows with missing values

# In[10]:


data.dropna(inplace=True)



# In[11]:


print(data)


# In[12]:


# There are no missing values, so here can move on to the next step


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


# Explore the distribution of the 'Stress level' column
print(data['Stress level'].describe())


# # convert 'Stress level' column to float

# In[15]:


# replace 'na' values with NaN
data['Stress level'] = pd.to_numeric(data['Stress level'], errors='coerce')


# In[16]:


data


# # Explore the correlation between different signals and the 'Stress level' column

# In[17]:


print(data.corr(numeric_only=True)['Stress level'])
# print(data.corr()['Stress level'], numeric_only=True)


# In[18]:


# compute correlation matrix
corr_matrix = data.corr(numeric_only=True)


# # Identify signals that might be better candidates for predicting stress

# In[19]:


correlations = data.corr(numeric_only=True)
plt.figure(figsize=(10,10))
plt.title("Correlation Matrix")
plt.imshow(correlations, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.show()


# # Split the dataset into training and testing sets (80% training, 20% testing)

# In[20]:


train = data.sample(frac=0.8, random_state=1)
test = data.drop(train.index)


# # Print the shape of the training and testing sets

# In[21]:


print('\n--- Shape of the training set ---')
print(train.shape)
print('\n--- Shape of the testing set ---')
print(test.shape)



# # Create a subset for testing

# In[22]:


test_size = 0.2
test_data = data.sample(frac=test_size, random_state=1)
train_data = data.drop(test_data.index)


# # Preview the data set

# In[23]:


print(train_data.head())


# In[24]:


print(train_data.columns)


# In[25]:


features = ['COVID related', 'Treating a covid patient', 'Patient in Crisis', 'Patient or patient\'s family', 'Doctors or colleagues', 'Administration, lab, pharmacy, radiology, or other ancilliary services\n', 'Increased Workload', 'Technology related stress', 'Documentation', 'Competency related stress', 'Saftey (physical or physiological threats)', 'Work Environment - Physical or others: work processes or procedures']


# In[26]:


X_train = train_data[features]
y_train = train_data['Stress level']

X_test = test_data[features]
y_test = test_data['Stress level']

print(y_train, y_test)


# In[27]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[28]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].apply(str)
        data[col] = label_encoder.fit_transform(data[col])


# In[29]:


data = data.drop('date', axis=1)


# In[30]:


print(y_train.isna().sum())


# In[31]:


X_train = X_train[~np.isnan(y_train)]
y_train = y_train[~np.isnan(y_train)]


# In[32]:


model.fit(X_train, y_train)


# In[33]:


print(y_test.isna().sum())


# In[34]:


X_test = X_test[~np.isnan(y_test)]
y_test = y_test[~np.isnan(y_test)]


# In[44]:


# Make predictions on new data
y_pred = model.predict(X_train)

#print the predicted values
print(y_pred)


# In[45]:


# Evaluate the model on the training set
train_accuracy = model.score(X_train, y_pred)

#print the accuracy
print('training accuracy: {:.2f}%'.format(train_accuracy * 100))


# In[51]:


from sklearn.linear_model import LogisticRegression

# Define a range of learning rates to try
learning_rates = [0.001, 0.01, 0.1]

# Try each learning rate and record the train_accuracies
train_accuracies = []
for lr in learning_rates:
    model = LogisticRegression(C=1/lr)
    model.fit(X_train, y_train)
    accuracy = model.score(X_train, y_train)
    train_accuracies.append(accuracy)
    
test_accuracies = []
for lr in learning_rates:
    model = LogisticRegression(C=1/lr)
    model.fit(X_test, y_test)
    accuracy = model.score(X_test, y_test)
    test_accuracies.append(accuracy) 


# In[52]:


plt.plot(learning_rates, train_accuracies)
plt.xlabel('Learning Rate')
plt.ylabel('Training Accuracy')
plt.title('Effect of Learning on Training Accuracy')
plt.show()

plt.plot(learning_rates, train_accuracies)
plt.xlabel('Learning Rate')
plt.ylabel('Testing Accuracy')
plt.title('Effect of Learning on Testing Accuracy')
plt.show()


# In[53]:


#choose the learning rate that gave th ebest training accuracy
best_lr = 0.01

# Train a new model using the best learning rate and all the training data
model = LogisticRegression(C=1/best_lr)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluate the model's performance on the test set
test_accuracy = model.score(X_test, y_pred)
print('Test Accuracy:', test_accuracy)

# Use the model to make predictions on new, unseen data
# y_pred = model.predict(X_train)

