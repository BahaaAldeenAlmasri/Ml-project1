import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#.............................................................................................
#1-Read in data 

#ادرج عنوان ملف البيانات"training dataset"
features1 = pd.read_csv("")# ادرج عنوان ملف البيانات بين إشارتي الاقتباس

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
predictions1 = model1.predict(test_features1) #تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
accuracy1 = accuracy_score(test_labels1, predictions1)# evealuating the model#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
print("\nAccuracy of Model 1: " + str(accuracy1),"\n")#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ



#ادرج عنوان ملف البيانات التي تريد توقع نتائجها
#7- Using the model for predicting actual data
actual_prediction1=pd.read_csv("") # ادرج عنوان ملف البيانات بين إشارتي الاقتباس
result1= model1.predict(actual_prediction1) # prdicting result for the datset
result1=result1.reshape([3,1])
print("entered variables:\n", actual_prediction1)#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
print("status:\n", result1)#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ

#..................................................................................................

#................................................................................................
#leg Pressure sensors Model:
     
#1-Read in data
       
#ادرج عنوان ملف البيانات"training_leg"
features2 = pd.read_csv("")
      
#2-spliting target (values we want to predict) of other features

labels2 = np.array(features2["Status"])#"status" is the name of the label colmun
        
features2 = features2.drop("Status",axis =1) #separating features cloumns from the total dataset
        
            
#3-convert to numby array
features2 = np.array(features2)
       

#4-spliting data into training and testing sets
train_features2, test_features2, train_labels2, test_labels2 = train_test_split(features2,
                                                                                labels2,
                                                                                test_size=0.1,
                                                                                random_state=40)
#this method is used for training the model and then for evaluation
#the prameters are chosen to achive the optimal accuracy for the model

#5-Train the model
model2 = RandomForestClassifier(n_estimators=500,
                                random_state=0,
                                max_depth=25,
                                max_leaf_nodes=60)
                
model2.fit(train_features2, train_labels2)
        
#6- Predict the model for calculating accuracy
predictions2 = model2.predict(test_features2)#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
accuracy2 = accuracy_score(test_labels2, predictions2)#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
print("\n.........................\n")#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
print("Accuracy of Model2: " + str(accuracy2))#تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ


#ادرج عنوان ملف البيانات التي تريد توقع نتائجها
#7- Using the model for predicting actual data
actual_prediction2=pd.read_csv("") # ادرج عنوان ملف البيانات بين إشارتي الاقتباس
result2= model2.predict(actual_prediction2)
result2=result2.reshape([3,1])
print("entered variables:\n", actual_prediction2) #تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ
print("status:\n", result2) #تركت للتوضيح، احذفها عند ادراج النموذج في البرنامج لتجنب أي خطأ

#.............................................................................................