import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from scipy import signal 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc, precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Dataset Reading
Data = pd.read_csv("eeg.csv") 

# Exploring Data
print(Data.dtypes)
print(Data.columns)
print("Data shape:",Data.shape)
print(Data.head())
print(Data.describe())
print(Data.info())
# Check for any nulls
print(Data.isnull().sum())
print(Data['eye'].describe())
print(Data['eye'].head())

Data['eye'].value_counts().plot.pie(labels = ["1-open","0-closed"],
                                              autopct = "%1.0f%%",
                                              shadow = True,explode=[0,.1])
    
#Data set Parameters
ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'eye'] # channel names   
sfreq = 128  # sampling frequency, in hertz  
info = mne.create_info(ch_names, sfreq)

# Transfer Labels: From Open-Closse to 1-0 levels
Data['eye']=Data["eye"].astype('category')
Data["eye"] = Data["eye"].cat.codes

for i in ch_names[0:14]:
    plt.plot(Data[i])

# Despiking
for i in ch_names[0:14]:
    Data[i]=Data[i].where(Data[i] < Data[i].quantile(0.999), Data[i].mean())
    Data[i]=Data[i].where(Data[i] > Data[i].quantile(0.001), Data[i].mean())
    plt.plot(Data[i])
    

signal.detrend(Data[1:14], axis=- 1, type='linear', bp=0, overwrite_data=False)

for i in ch_names[0:14]:
    plt.plot(Data[i])
    
#Data Normalizing (min-max Scaling)
x = Data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
Data = pd.DataFrame(x_scaled)
Data.columns=ch_names

for i in ch_names[0:14]:
    plt.plot(Data[i])
    
#Numpy Array to mne data
data = Data.to_numpy() # Pandas DataFrame to Numpy Array
data_mne = np.transpose(data)
raw = mne.io.RawArray(data_mne, info)   
raw.plot() #Plot all the signales

# Scatter plot each two channels (featers) together
for i in ch_names[1:14]:
    for k in ch_names[1:14]:
        sns.lmplot(i, k, Data, hue='eye', fit_reg=False)
        fig = plt.gcf()
        #fig.set_size_inches(15, 10)
        plt.show()

# Prepare Train and test Data
splitRatio = 0.2
train, test = train_test_split(Data ,test_size=splitRatio,
                               random_state = 123, shuffle = True)

train_X = train[[x for x in train.columns if x not in ["eye"]]]
train_Y = train["eye"]


feature_cols = train_X.columns

test_X = test[[x for x in train.columns if x not in ["eye"]]]
test_Y = test["eye"]

# Model 1: DT (decission Tree)
## Traiing Model
clf_DT = DecisionTreeClassifier()
clf_DT = clf_DT.fit(train_X,train_Y)
pred_y_DT = clf_DT.predict(test_X)

## Metrics
print("Accuracy:",accuracy_score(test_Y, pred_y_DT))
print("f1 score:", f1_score(test_Y, pred_y_DT))
print("confusion matrix:",confusion_matrix(test_Y, pred_y_DT))
print("precision score:", precision_score(test_Y, pred_y_DT))
print("recall score:", recall_score(test_Y, pred_y_DT))
print("classification report:", classification_report(test_Y, pred_y_DT))

## Plots
### 01 plot Confusion Matrix as heat map
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(test_Y, pred_y_DT),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

### 02 plot ROC curve
test_Y_01 =test_Y.astype('category')
test_Y_01 = test_Y_01.cat.codes

pred_y_01 = pd.Series(pred_y_DT)
pred_y_01 =pred_y_01.astype('category')
pred_y_01 = pred_y_01.cat.codes

fpr,tpr,thresholds = roc_curve(test_Y_01,pred_y_01)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=15)

#Model 2: KNN K-Nearest Neighbours
## Traiing Model
clf_KNN = KNeighborsClassifier(15)
clf_KNN = clf_KNN.fit(train_X,train_Y)
pred_y_KNN = clf_KNN.predict(test_X)

## Metrics
print("Accuracy:",accuracy_score(test_Y, pred_y_KNN))
print("f1 score:", f1_score(test_Y, pred_y_KNN))
print("confusion matrix:",confusion_matrix(test_Y, pred_y_KNN))
print("precision score:", precision_score(test_Y, pred_y_KNN))
print("recall score:", recall_score(test_Y, pred_y_KNN))
print("classification report:", classification_report(test_Y, pred_y_KNN))

## Plots
### 01 plot Confusion Matrix as heat map
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(test_Y, pred_y_KNN),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

### 02 plot ROC curve
test_Y_01 =test_Y.astype('category')
test_Y_01 = test_Y_01.cat.codes

pred_y_01 = pd.Series(pred_y_KNN)
pred_y_01 =pred_y_01.astype('category')
pred_y_01 = pred_y_01.cat.codes

fpr,tpr,thresholds = roc_curve(test_Y_01,pred_y_01)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=15)
