import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler

url = 'https://raw.githubusercontent.com/SanB-1/Stroke-Prediction/main/stroke.csv'
df = pd.read_csv(url)
df = df.dropna()
df = df.drop(['id'], axis=1)
df.info()

le = LabelEncoder()
labelG = le.fit_transform(df['gender'])
labelMa = le.fit_transform(df['ever_married'])
labelWo = le.fit_transform(df['work_type'])
labelRe = le.fit_transform(df['Residence_type'])
labelSm = le.fit_transform(df['smoking_status'])
df["gender"] = labelG
df["ever_married"] = labelMa
df["work_type"] = labelWo
df["Residence_type"] = labelRe
df["smoking_status"] = labelSm

x = df.drop(['stroke'], axis=1)
y = df['stroke']
oversample = RandomOverSampler(sampling_strategy='minority')
x_over, y_over = oversample.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.2, random_state=10)
train_df = x_train.join(y_train)
test_df = x_test.join(y_test)

classifier = RandomForestClassifier(n_estimators=10, random_state=0, oob_score=True)
classifier.fit(x_train, y_train)
classifier.score(x_test, y_test)

import pickle
pickle.dump(classifier, open('classifier.pkl', 'wb'))