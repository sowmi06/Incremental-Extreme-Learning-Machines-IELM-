import time as time
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


def mnist_data():
    # loading the image in train and test samples
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    total_train_sample, height, width = x_train.shape
    dimension = height * width

    # preprocessing the dataset
    scale = 225.0
    x_train = x_train.reshape(total_train_sample, dimension)
    x_test = x_test.reshape(x_test.shape[0], dimension)
    x_train = x_train.astype('float32')
    x_train = x_train / scale
    x_test = x_test.astype('float32')
    x_test = x_test / scale
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


class Ielm_mnist():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_neuron = self.x_train.shape[1]  # input_neuron=784
        self.output_neuron = self.y_train.shape[1]  # output_neuron=10

    # activation function:
    def sigmoid(self, x):
       return 1/(1 + np.exp(-x))

    # train module:
    def train(self,x_train,y_train) :
        start_train_process= time.process_time()
        print("-------------------------------")
        print(" Training Ielm")
        print("-------------------------------")
        E = self.y_train
        L = 0  # hidden_neuron
        while L <= 1000:
           input_weight = np.random.normal(size=[self.input_neuron,L]) 
           bias = np.random.normal(size=[L])
           g = np.dot(x_train, input_weight)
           g = g + bias
           H = self.sigmoid(g)
           inverse = np.linalg.pinv(H)
           beta = np.dot(inverse,y_train) #learning parameter
           L = L+50
           d = np.dot(H, beta)
           E = E - d
        total_train = x_train.shape[0]
        correct_value = 0
        for i in range(total_train):
            actual = np.argmax(y_train[i])
            predict = np.argmax(d[i])
            if predict == actual:
                 correct_value =correct_value + 1
        testing_accuracy = correct_value/total_train
        print("-------------------------------")
        print("Training accuracy:", testing_accuracy)
        print("-------------------------------")
        end_train_process = time.process_time()
        training_time =end_train_process-start_train_process
        print("Training Time:",training_time)
        print("-------------------------------")
        return L,beta
       
    # test module:
    def test(self, x_test, y_test, L, beta ):
        L = L-50
        start_test_process = time.process_time()
        print("-------------------------------")
        print(" Testing Ielm")
        print("-------------------------------")
        input_weight=np.random.normal(size=[self.input_neuron,L])
        bias = np.random.normal(size=[L])
        g = np.dot(x_test, input_weight)
        g = g + bias
        H = self.sigmoid(g)
        d = np.dot(H, beta)
        total_test = x_test.shape[0]
        correct_value = 0
        for i in range(total_test):
            actual = np.argmax(y_test[i])
            predict = np.argmax(d[i])
            if predict == actual:
                 correct_value = correct_value + 1
            testing_accuracy = correct_value/total_test       
        print("-------------------------------")
        print("Testing accuracy:", testing_accuracy)
        print("-------------------------------")
        end_test_process = time.process_time()
        testing_time = end_test_process-start_test_process
        print("Testing Time:", testing_time)
        print("-------------------------------")
 
    
def main():
    # calling mnist dataset
    x_train, y_train, x_test, y_test = mnist_data()
   
    # calling the ielm class
    ielm = Ielm_mnist(x_train, y_train, x_test, y_test)
    L, beta = ielm.train(x_train, y_train)
    ielm.test(x_test, y_test, L, beta)


if __name__ == "__main__":
    main()




