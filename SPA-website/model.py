#importing the required libraries
import pandas as pd
import random
from sklearn.cluster import KMeans
import numpy as np
import pickle

#importing the dataset
X = pd.read_csv(r'''student_new.csv''')

#creating an array for all the attributes that we need for the model
corr_columns = ['paid' , 'internet' , 'address',  'studytime', 'traveltime', 'Walc','health','average']
df = pd.DataFrame(X, columns=corr_columns)

#creating the k-means model and fitting it according to our attributes
kmeans = KMeans(n_clusters=5 , random_state = 990)
y = kmeans.fit(df[corr_columns].values)

#saving model to disk
pickle.dump(y, open('model.pkl','wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
