# zip - wraps the data in lists

import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate = 0.1):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        #self.weights = self.weights + 2   #adding constant to all elements int the numpy array

    def predict(self, inputs):
        summation = np.dot(inputs,self.weights[1:]) + self.weights[0]   #wx+b
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        print("Threshold \t Prediction \t\t Weights")
        print("-----------------------------------------------------------------")
        for _ in range(self.threshold):
            dummy = []
            dum_l = []
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                dummy.append(prediction)
                dum_l.append(label)
                print(_,end=" \t\t ")
                print(prediction, end= " \t\t\t ")
                print(self.weights)        

                if len(dummy) == len(dum_l):
                    flag = 0
                    if(dummy == dum_l):
                        flag+=1

            if flag > 0:
                print("Threshold:" , _)
                print("Weights:" , self.weights)
                break                                    
        print("-----------------------------------------------------------------")
