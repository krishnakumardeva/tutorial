# --------------- IMPORTING PACKAGES -------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn
from sklearn.metrics import classification_report

# --------------- READING DATASET -------------------------

data = pd.read_csv('data.csv')

print("--------------------------------------------------")
print("                     Data Selection               ")
print("--------------------------------------------------")
print()
print(data.head(10))

# --------------- PREPROCESSING -------------------------

print("--------------------------------------------------")
print("                 Data Preprocessing               ")
print("--------------------------------------------------")
print()
print()


data.dropna(inplace =True)
# data

print("--------------------------------------------------")
print("   Missing Handling Values             ")
print("--------------------------------------------------")
print()

### checking null values
data.isna().sum()

(data.isnull().sum()/data.shape[0])*100

print("------------------------------------------------")
print(" Before label Encoding ")
print("------------------------------------------------")
print()

print(data['Anxiety_disorder'].head(20))

from sklearn import preprocessing
  
label_encoder = preprocessing.LabelEncoder()

  
# Encode labels in column
data['Jaundice']= label_encoder.fit_transform(data['Jaundice'])
data['Sex']= label_encoder.fit_transform(data['Sex'])
data['Global developmental delay/intellectual disability'] = label_encoder.fit_transform(data['Global developmental delay/intellectual disability'])
#data['Social/Behavioural Issues']= label_encoder.fit_transform(data['Social/Behavioural Issues'])
data['Anxiety_disorder']= label_encoder.fit_transform(data['Anxiety_disorder'])
data['Jaundice'].unique()
data['Sex'].unique()
data['Global developmental delay/intellectual disability'].unique()
data['Anxiety_disorder'].unique()

print("-------------------------------------------")
print(" After label Encoding ")
print("------------------------------------------")
print()

print(data['Anxiety_disorder'].head(20))


# ====== DATA SPLITTING 


X = data.drop(['Ethnicity','Family_mem_with_ASD','Social/Behavioural Issues','Who_completed_the_test','CASE_NO_PATIENT','Speech Delay/Language Disorder','Learning disorder','Genetic_Disorders','Depression','ASD_traits'], axis = 1)

Y = data['ASD_traits']

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.25)


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ====== STANDARD SCALAR

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ====== DECISION TREE

from sklearn import tree
from sklearn.metrics import confusion_matrix
model2 = tree.DecisionTreeClassifier(random_state=42, max_depth=10)

model2.fit(X_train, Y_train)

print("-------------------------------------------")
print(" Decision Tree Classifier")
print("------------------------------------------")
print()

print("Dt_acc:" ,model2.score(X_test, Y_test)*100)

pred = model2.predict(X_test)
print(classification_report(y_true=Y_test,y_pred=pred))
tn, fp, fn, tp = confusion_matrix(Y_test,pred).ravel()

#Specificity
specificity = tn/(tn+fp)
print("specificity:" , specificity)

# sensitivity
sensitivity = tp/(tp+fn)
print("sensitivity:" ,sensitivity)

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics



confusion_matrix = metrics.confusion_matrix(y_true=Y_test,y_pred=pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['No','Yes'])

cm_display.plot()
plt.show()
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model2, X_test, Y_test)
plt.show()

# ====== RANDOM FOREST

print("-------------------------------------------")
print(" Random Forest Classifier")
print("------------------------------------------")
print()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model5 = RandomForestClassifier(max_depth=10, random_state=42).fit(X_train, Y_train)

print("RF_Accuracy:" ,model5.score(X_test, Y_test)*100)
pred = model5.predict(X_test)
print(classification_report(y_true=Y_test,y_pred=pred))
tn, fp, fn, tp = confusion_matrix(Y_test,pred).ravel()

#Specificity
specificity = tn/(tn+fp)
print("RF_specificity:" ,specificity)
# sensitivity
sensitivity = tp/(tp+fn)
print("RF_sensitivity:" ,sensitivity)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_true=Y_test,y_pred=pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['No','Yes'])

cm_display.plot()
plt.show()


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model5, X_test, Y_test)
plt.show()



# ================= PREDICTION


for i in range(0,10):
    
    if pred[i] == 0:
        print("----------------------")
        print()
        print([i],"Affected by AUTISUM")
    else:
        print("------------------------")
        print()
        print([i],"Affected by NOT AUTISUM")













