import time as time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def regression_data(path):
    columns = ["CIC0", "SM1_Dz", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]
    df = pd.read_csv(path , delimiter=";", names=columns)
    x = df.copy(deep=True)
    x.drop(["LC50"], inplace=True, axis=1)
    y = df["LC50"].copy(deep=True)
    x = np.array(x)
    y = np.array(y)
    # test train split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_train, y_train, x_test, y_test


class Ielm_mnist():
    
    def __init__(self,x_train,y_train,x_test,y_test):
         
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.input_neuron=self.x_train.shape[1] #input_neuron=784

    # activation function:
    def sigmoid(self,x):
       return 1/(1 + np.exp(-x))

    # train module:
    def train(self,x_train,y_train) :
        start_train_process= time.process_time()
        print("-------------------------------")
        print("Training Ielm")
        print("-------------------------------")
        E=self.y_train
        L=0 #hidden_neuron
        while L<=1000:  
           input_weight = np.random.normal(size=[self.input_neuron,L]) 
           bias= np.random.normal(size=[L])
           g = np.dot(x_train, input_weight)
           g = g + bias
           H = self.sigmoid(g)
           inverse=np.linalg.pinv(H)
           beta = np.dot(inverse, y_train)
           d= np.dot(H, beta)
           E = E - d
           L=L+1
           ee=y_train-d
           mse =np.mean(pow(ee,2))
           rmse=np.sqrt(mse)   
        print("-------------------------------")
        print(" Training RMSE:", rmse)
        print("-------------------------------")
        end_train_process= time.process_time()
        training_time=end_train_process-start_train_process
        print("Training Time:",training_time)
        print("-------------------------------")
        return L,beta

    # test module:
    def test(self, x_test, y_test, L, beta):
        start_test_process= time.process_time()
        L = L-1
        print("-------------------------------")
        print("Testing Ielm")
        print("-------------------------------")
        input_weight=np.random.normal(size=[self.input_neuron,L])
        bias= np.random.normal(size=[L])
        g = np.dot(x_test, input_weight)
        g = g + bias
        H = self.sigmoid(g)
        inverse=np.linalg.pinv(H)
        beta = np.dot(inverse, y_test)
        d = np.dot(H, beta)
        ee=y_test-d
        mse =np.mean(pow(ee,2))
        rmse=np.sqrt(mse) 
        print("-------------------------------")
        print(" Testing RMSE:", rmse)
        print("-------------------------------")
        end_test_process= time.process_time()
        testing_time=end_test_process-start_test_process
        print("Testing Time:",testing_time)
        print("-------------------------------")
        print("-------------------------------")
 
    
def main():
    
    # calling qsar_fish_toxicity dataset
    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv'
    x_train, y_train, x_test, y_test = regression_data(path)
   
    # calling the ielm class
    ielm = Ielm_mnist(x_train, y_train, x_test, y_test)
    L, beta = ielm.train(x_train, y_train)
    ielm.test(x_test, y_test, L, beta)


if __name__ == "__main__":
    main()




