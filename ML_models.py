import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#.............................................................................................
#1-Read in data 

#Insert training dataset
features1 = pd.read_csv("")

#2-spliting target (values we want to predict) of other features
labels1 = np.array(features1["status"]) #"status" is the name of the label colmun
features1 = features1.drop("status",axis =1) #separating features cloumns from the total dataset

#3-convert to numby array
features1 = np.array(features1)

#4-spliting data into training and testing sets
train_features1, test_features1, train_labels1, test_labels1 = train_test_split(features1,
                                                                            labels1,
                                                                            test_size=0.25,
                                                                           random_state=42)
#this method is used for training the model and then for evaluation
#the prameters are chosen to achive the optimal accuracy for the model

#5-Train the model
model1 = svm.SVC(kernel='rbf') #creating a model using Support Victor Classification algorithim
model1.fit(train_features1,train_labels1) # training the model

#6- Predict the model for calculating accuracy
predictions1 = model1.predict(test_features1) 
accuracy1 = accuracy_score(test_labels1, predictions1)# evealuating the model
print("\nAccuracy of Model 1: " + str(accuracy1),"\n")



#7- Using the model for predicting actual data
actual_prediction1=pd.read_csv("") 
result1= model1.predict(actual_prediction1) # prdicting result for the datset
result1=result1.reshape([3,1])
print("entered variables:\n", actual_prediction1)
print("status:\n", result1)

#..................................................................................................
