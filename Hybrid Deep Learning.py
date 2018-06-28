# Hybrid Deep Learning Model :
# Self Organizing maps + Artificial Neural Network

# Problem : Identifying Potential Frauds from Credit Card Applications



# Part 1 - Data Handling and Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)



# Part 2 - Building and training the SOM

from minisom import MiniSom

# Initializing the model
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Randomizing the weights
som.random_weights_init(X)
# Training the model
som.train_random(data = X, num_iteration = 100)



# Part 3 - Visualizing the results

from pylab import bone, pcolor, colorbar, plot, show

# Initialize the plot
bone()
# Put winning nodes on the map -> Take transpose for the right order of MID matrix
pcolor(som.distance_map().T)
# Adding a legend
colorbar()
# Frauds are identified by the outlying winning nodes
# Adding markers to the map to see if the customers who cheated got approval or not
markers = ['o', 's']
colors = ['r', 'g']
# Approved -> Red Circle
# Not Approved -> Green Circle
for i, x in enumerate(X):
    # get the winning node for the customer
    w = som.winner(x)
    # Plotting if the winning node was approved or not
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()



# Part 4 - Finding the frauds
mappings = som.win_map(X)
''' mappings is a dictionary which contains the (x,y) coordinates as it's keys and 
    each key corresponds to a numpy array containing all the customers corresponding
    to that winning node '''
# Get Coordinates of the outlying winning nodes using the map
frauds = np.concatenate((mappings[(7,2)], mappings[(5,1)]), axis = 0)
# frauds consists lists of potential customers who may have cheated
frauds = sc.inverse_transform(frauds)



# Part 5 - Going from Unsupervised to Supervised Deep Learning

''' Adding a new feature of is_fraud as 0 : No Fraud or 1 : Fraud
    If CustomerID in frauds[0] set is_fraud=1 else is_fraud=0 '''

# Create a numpy array for the target vector (Dependent Variable)
is_fraud = []
for customer in sc.inverse_transform(X)[:,0]:
    if customer in frauds[:,0]:
        is_fraud.append(1)
    else:
        is_fraud.append(0)
is_fraud = np.array(is_fraud)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting data into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[:,1:],is_fraud,random_state=42,test_size=0.3)

# Import keras and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()
# Adding the input layer
classifier.add(Dense(units=7,input_dim=14,activation='relu',kernel_initializer='uniform'))
# Adding hidden layer
classifier.add(Dense(units=7,activation='relu',kernel_initializer='uniform'))
# Adding second hidden layer
classifier.add(Dense(units=7,activation='relu',kernel_initializer='uniform'))
# Adding the output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting and training the model
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Making the predictions
y_pred = classifier.predict(X_test)

# Create a pandas Dataframe with the is_fraud column
dataset_final = dataset
dataset_final['Is Fraud'] = is_fraud
# Sort the dataset_final in ascending order of fraud probabilities
dataset_final.sort_values(by='Is Fraud',axis=0,inplace=True)

# Because of the sigmoid function y_pred contains probabilities
# Convert probabilities to binary outcomes
for count,value in enumerate(y_pred):
    if(value>0.5):
        y_pred[count] = 1
    else:
        y_pred[count] = 0

# Checking the accuracy of the model
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
accuracy = np.sum(np.diag(cm)/np.sum(cm))

