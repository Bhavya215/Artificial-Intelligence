import numpy as np

from Perceptron_Train import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1,1]))
training_inputs.append(np.array([1,0,0]))
training_inputs.append(np.array([1,0,1]))
training_inputs.append(np.array([0,0,1]))
training_inputs.append(np.array([1,1,0]))
training_inputs.append(np.array([0,0,0]))

labels = np.array([1,1,1,1,1,0])

perceptron = Perceptron(3)
perceptron.train(training_inputs,labels)


print("INPUT       |      OUTPUT")

inputs = np.array([0,1,0])
print("[0,1,1]     |        ", perceptron.predict(inputs))

inputs = np.array([0,1,1])
print("[0,1,0]     |        ", perceptron.predict(inputs))
