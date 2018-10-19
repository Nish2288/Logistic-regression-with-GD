import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

def LogisticRegression():
    train_x=pd.read_csv("train_cancer_data.csv")
    train_x=np.array(train_x)
    train_y=pd.read_csv("train_cancer_data_y.csv")
    train_y=np.array(train_y)
    model(train_x.T,train_y.T,num_iter=90000,alpha=0.000000065)
    
    
def initialize(dim):
    w=np.zeros(shape=(dim,1))
    b=0
    
    return w,b
     
def sigmoid(z):
    
    s=1/(1+np.exp(-z))
    return s    
    
def propogate(X,Y,w,b):
    
    #Forward propogation
    
    m=X.shape[1]  #Number of rows
    

    A=sigmoid(np.dot(w.T, X) + b)
    
    cost=(-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    cost=np.squeeze(cost)
    
    #Back-propogation
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)
   
    return cost,dw,db
     
def optimize(X,Y,w,b,num_iter,alpha):
    
    costs=[]
    for i in range(num_iter):
        cost,dw,db=propogate(X,Y,w,b)
        w=w-alpha*dw
        b=b-alpha*db
        
        if i %100==0:
            costs.append(cost)
            
        if i%100==0:
            print("Iteration:%i  Cost: %f " %(i,cost))
    
    return costs,w,b
    
    
def predict(X,w,b):
    #w=w.reshape(X.shape[0],1)
    m=X.shape[1]
    y_pred=np.zeros(shape=(1,m))
    
    A=sigmoid(np.dot(w.T,X)+b)    
    
    for i in range(A.shape[1]):
        y_pred[0,i]=1 if A[0,i]>0.5 else 0
    
  
    print("Predict")
    return y_pred
     
     
     
def model(train_x,train_y,num_iter,alpha):
     
    w,b=initialize(train_x.shape[0])
          
    costs,w,b= optimize(train_x,train_y,w,b,num_iter,alpha)
     
    print("**************************")
    X_test=pd.read_csv("test_cancer_data.csv")
    X_test=np.array(X_test)
    X_test=X_test.T
    Y_test=pd.read_csv("test_cancer_data_y.csv")
    Y_test=np.array(Y_test)
    Y_test=Y_test.T     
    y_pred_train=predict(train_x,w,b)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - train_y)) * 100))
    
    y_pred_test=predict(X_test,w,b)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - Y_test)) * 100))
    
    #costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    #plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
    #Train accuracy:90.91
    #Test:89.26
     
     
LogisticRegression()