import numpy as np
import random
import math

def func1(x):
    return math.log(x)
    
x_list = [x for x in range(1,50)]
#x_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
random.seed(1)
random.shuffle(x_list)
train_num  = 40
x_list = [[x] for x in x_list]
x_train = x_list[0:train_num]
x_test = x_list[train_num:]

y_list = [[func1(x[0])] for x in x_list]
y_train = y_list[0:train_num]
y_test = y_list[train_num:]



from sklearn.neural_network import MLPRegressor


# relu
#mlp1 = MLPRegressor(hidden_layer_sizes=(1,1), activation='tanh', verbose = True, learning_rate_init=0.01, max_iter=2000)
mlp1 = MLPRegressor(hidden_layer_sizes=(500, 10), activation='tanh', 
                    verbose = True, solver='adam', learning_rate_init = 0.0001, 
                    max_iter=50000, tol=0.000001, random_state=2)
                    
mlp1.fit(x_train, y_train)
y_predict = mlp1.predict(x_test)

print ("x_train: ", x_train)
print ("y_train: ", y_train)
print ("y_test: ", y_test)
print ("y_predict: ", y_predict)