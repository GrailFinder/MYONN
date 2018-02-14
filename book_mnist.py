
import numpy
from nn import NeuralNetwork
from mnist_play import get_mnist, get_local_mnist
from tqdm import tqdm
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1
# train the neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# epochs is the number of times the training data set is used for training
epochs = 5

training_data_list, test_data_list = get_local_mnist()

for e in tqdm(range(epochs)):
    # go through all records in the training data set
    for record in tqdm(training_data_list):
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = numpy.asfarray(all_values[1:]) / 255.0
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes)
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 1
        n.train(inputs, targets)


scorecard = []

# go through all the records in the test data set
for record in tqdm(test_data_list):
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = numpy.asfarray(all_values[1:]) / 255.0
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs[0])
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)



# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

