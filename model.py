import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

heart_data = pd.read_csv('heart.data')
heart_data.drop_duplicates(inplace=True)

min_thresold , max_thresold = heart_data.chol.quantile([0.001,0.999])
min_thresold , max_thresold

heart_data = heart_data[(heart_data.chol<max_thresold)& (heart_data.chol>min_thresold)]
X = heart_data.drop(columns ='target',axis=1) 
y= heart_data['target']

X_train,X_test,Y_train,Y_test = train_test_split(X ,y, test_size =0.2 , random_state=20)

#training the logistic regression model with training data
LR_model = LogisticRegression(C= 1.0, penalty= 'l2', solver= 'lbfgs')

bag_model = BaggingClassifier(
    base_estimator = LR_model,
    n_estimators = 200,
    max_samples = 0.7,
    oob_score = True,
    random_state = 0

)

bag_model.fit(X_train,Y_train)

pickle.dump(bag_model,open('heart.pkl','wb'))