''' Logistic regression with gradient descent.  '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LogisticRegression():
    ''' Importing dataset. '''
    
    train_x=pd.read_csv('train_cancer_data.csv')
    train_x=np.array(train_x)
    train_y=pd.read_csv('train_cancer_data_y.csv')
    train_y=np.array(train_y)
    
    d=model(train_x.T,train_y.T,n_iter=173700,alpha=0.000000065) #173700
    
def initialize(m):
    ''' Initialize w & b '''
    w=np.zeros(shape=(m,1))   
    b=0
    
    return w,b
    
def sigmoid(z):
    s=1/(1+np.exp(-z))    
    return s
    
def propogate(X,Y,w,b):
    ''' Forward propagation '''
    m=X.shape[1]
    
    z=np.dot(w.T,X)+b
    A=sigmoid(z)
    cost=(-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost=np.squeeze(cost)
    
    '''Back propagation '''
    dZ=A-Y
    dw=(1/m)*np.dot(X,dZ.T)
    db=(1/m)*np.sum(dZ)
    return cost,dw,db
    
    
def optimize (X,Y,n_iter,alpha):
    w,b=initialize(X.shape[0])
    
    costs=[]
    for i in range(n_iter):
        cost,dw,db=propogate(X,Y,w,b)
        w=w-alpha*dw
        b=b-alpha*db
        if i%100==0:
            costs.append(cost)
            print("Iteration: %i , Cost: %f"%(i,cost))
            
        
    return w,b,costs  
  
def predict(w,b,X):
    y_pred=sigmoid(np.dot(w.T,X)+b)
    y_pred=np.round(y_pred)
    return y_pred
      
def model(X,Y,n_iter,alpha):
    w,b,costs=optimize(X,Y,n_iter,alpha) 
    
    test_X=pd.read_csv('test_cancer_data.csv')
    test_X=np.array(test_X)
    test_y=pd.read_csv('test_cancer_data_y.csv')
    test_y=np.array(test_y)
    
    ''' Training accuracy '''
    y_pred_train=predict(w,b,X)
    train_accuracy=(100-np.mean(np.abs(y_pred_train-Y))*100)
    print("Training accuracy:%f" %train_accuracy)

    ''' Test accuracy '''
    y_pred_test=predict(w,b,test_X.T)
    test_accuracy=(100-np.mean(np.abs(y_pred_test-test_y.T))*100)
    print("Test accuracy:%f" %test_accuracy)
    
   
    
    TrainPredList=y_pred_train.tolist()
    YList=Y.tolist()
    
    truePositive=0
    trueNegative=0
    falsePositive=0
    falseNegative=0
    
    for i in range(len(TrainPredList[0])):
        if TrainPredList[0][i]==1 and YList[0][i]==1 :
            truePositive +=1
        elif TrainPredList[0][i]==0 and YList[0][i]==0:
            trueNegative +=1
        elif TrainPredList[0][i]==1 and YList[0][i]==0 :
            falsePositive +=1
        elif TrainPredList[0][i]==0 and YList[0][i]==1:
            falseNegative+=1
    
    tpr = truePositive / (truePositive + falseNegative) * 100
    fpr = falsePositive / (falsePositive + trueNegative) * 100
    
    precision = truePositive / (truePositive + falsePositive) * 100
    print("On training set:\nTrue Positive:  ", truePositive)
    print("True Negative:  ", trueNegative)
    print("False Negative:  ", falseNegative)
    print("False Positive:  ", falsePositive)
    print("True Positive Rate / Recall: %.2f" % tpr+str('%'))
    print("Precision: %.2f" %precision+str('%'))
    print("False Positive Rate / Fallout: %.2f" %fpr+str('%'))
    
    ''' Test set '''
    TestPredList=y_pred_test.tolist()
    YList=test_y.T.tolist()
    
    truePositive=0
    trueNegative=0
    falsePositive=0
    falseNegative=0
    
    
    
    for i in range(len(TestPredList[0])):
        
        if TestPredList[0][i]==1 and YList[0][i]==1 :
            truePositive +=1
        elif TestPredList[0][i]==0 and YList[0][i]==0:
            trueNegative +=1
        elif TestPredList[0][i]==1 and YList[0][i]==0 :
            falsePositive +=1
        elif TestPredList[0][i]==0 and YList[0][i]==1:
            falseNegative+=1
    
    tpr = truePositive / (truePositive + falseNegative) * 100
    fpr = falsePositive / (falsePositive + trueNegative) * 100
    
    precision = truePositive / (truePositive + falsePositive) * 100
    print("\nOn Test set:\nTrue Positive:  ", truePositive)
    print("True Negative:  ", trueNegative)
    print("False Negative:  ", falseNegative)
    print("False Positive:  ", falsePositive)
    print("True Positive Rate / Recall: %.2f" % tpr+str('%'))
    print("Precision: %.2f" %precision+str('%'))
    print("False Positive Rate / Fallout: %.2f" %fpr+str('%'))
    
    ''' Plot '''
    cost=np.squeeze(costs)
    plt.plot(cost)
    plt.ylabel('Cost')
    plt.xlabel('Iteration per 100')
    plt.title('Learning rate ' +str(+alpha))
    plt.show()  
    
    return True
    
    
    
LogisticRegression()
