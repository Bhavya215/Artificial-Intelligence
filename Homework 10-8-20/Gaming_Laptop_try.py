import numpy as np

from Perceptron_Train import Perceptron

training_inputs = []
training_inputs.append(np.array([0,0,0]))  
training_inputs.append(np.array([0,0,1]))
training_inputs.append(np.array([0,1,0]))
training_inputs.append(np.array([0,1,1]))
training_inputs.append(np.array([1,0,0]))   
#training_inputs.append(np.array([1,0,1]))  -> 1
#training_inputs.append(np.array([1,1,0]))  -> 1
#training_inputs.append(np.array([1,1,1]))  -> 1

labels = np.array([0,0,0,0,0])

perceptron = Perceptron(3)
perceptron.train(training_inputs,labels)


print("INPUT       |      OUTPUT          |\t\t\t\t    |")
print("---------------------------------------------------------------------")

inputs = np.array([1,1,1])
print("[1,1,1]     |        ", perceptron.predict(inputs),end="            |  ")
if(perceptron.predict(inputs) == 1):
    print("I will buy the gaming laptop  |")
else:
    print("I won't buy the gaming laptop |")

inputs = np.array([1,0,1])
print("[1,0,1]     |        ", perceptron.predict(inputs),end="            |  ")
if(perceptron.predict(inputs) == 1):
    print("I will buy the gaming laptop  |")
else:
    print("I won't buy the gaming laptop |")


inputs = np.array([1,1,0])
print("[1,1,0]     |        ", perceptron.predict(inputs),end="            |  ")
if(perceptron.predict(inputs) == 1):
    print("I will buy the gaming laptop  |")
else:
    print("I won't buy the gaming laptop |")


inputs = np.array([0,1,0])
print("[0,1,0]     |        ", perceptron.predict(inputs),end="            |  ")
if(perceptron.predict(inputs) == 1):
    print("I will buy the gaming laptop  |")
else:
    print("I won't buy the gaming laptop |")
print("---------------------------------------------------------------------")
