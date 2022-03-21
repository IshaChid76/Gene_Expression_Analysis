# In[105]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[106]:


dataset=pd.read_excel('Book1.xlsx')


# In[107]:


'''
To reduce the number or classes, I only included data from 6 classes numbered as 1,2,3,4,5 and 6 respectively.

'''
Y1=dataset[dataset.NAME==1]
Y2=dataset[dataset.NAME==2]
Y3=dataset[dataset.NAME==3]
Y4=dataset[dataset.NAME==4]
Y5=dataset[dataset.NAME==5]
Y6=dataset[dataset.NAME==6]


# In[108]:


#concatenate
dataset=pd.concat([Y1,Y2,Y3,Y4,Y5,Y6], ignore_index=True)


# In[109]:


#randomizing the dataset
from sklearn.utils import shuffle
dataset=shuffle(dataset)


# In[110]:


#Deciding dependent and independent variables 
X=dataset.iloc[:,2:20]
Y=dataset.iloc[:,1].values
#convert object file to int
Y=Y.astype('int')
#replaces nan with 0
X=X.fillna(0)


# In[111]:


#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[112]:


#forming standard data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[113]:


#Different deatures of genes
features=["alpha0", "alpha7","alpha14","alpha21","alpha28","alpha35","alpha42","alpha49","alpha56","alpha63","alpha70","alpha77","alpha84","alpha91","alpha98","alpha105","alpha112","alpha119",]


# In[114]:


#Feature importance:
from yellowbrick.features import Rank1D
visualizer= Rank1D(features=features, algorithm="shapiro")
visualizer.fit(X_test, Y_test)
visualizer.transform(X)


# <center><h3>Different Classifiers</h3></center>

# In[95]:


#SVC
from sklearn.svm import SVC
classifier=SVC(kernel="linear",random_state=0) 
classifier.fit(X_train,Y_train)


# In[71]:


#knn
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)           
classifier.fit(X_train,Y_train)


# In[62]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", random_state=0) 
classifier.fit(X_train,Y_train)


# In[115]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion="entropy",random_state=0)        #we use entropy for information gain#p=1 manhattan distance p=2 euclidean distance
classifier.fit(X_train,Y_train)


# <center><h3>Class Prediction Error</h3></center>

# In[12]:


#Class Prediction error
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
visualizer = ClassPredictionError( classifier, feature=features)
# Fit the training data to the visualizer
visualizer.fit(X_train, Y_train)
# Evaluate the model on the test data
visualizer.score(X_test, Y_test)
# Draw visualization
g = visualizer.poof()


# In[116]:


#prediction
y_pred=classifier.predict(X_test)


# In[117]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print(accuracy_score(Y_test, y_pred, normalize=True))
print(f1_score(Y_test, y_pred, average="macro"))
print(precision_score(Y_test, y_pred, average="macro"))
print(recall_score(Y_test, y_pred, average="macro")) 


# In[118]:


#Yellowbrick confusion matrix
from yellowbrick.classifier import ConfusionMatrix
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class="auto", solver="liblinear")
cm1 = ConfusionMatrix(model, classes=[1,2,3,4,5,6])
cm1.fit(X_train, Y_train)
cm1.score(X_test, Y_test)
#cm1.show()


# In[82]:


#making the Confusion matrix
cm=confusion_matrix(Y_test,y_pred)   
print(cm)


# In[ ]:




