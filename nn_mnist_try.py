from nn import NeuralNetwork
from mnist_play import get_mnist
from tqdm import tqdm
import numpy as np

# nn for mnist data
out = 10
neun = NeuralNetwork(inp=784, hid=200, out=out, alpha=.1)

#  get data
train_set, valid_set, test_set = get_mnist()

# train

epochs = 5

for e in tqdm(range(epochs)):
    inp, answ = train_set
    for i in tqdm(range(len(inp))):
        # scale the input data
        inputs = inp[i] / 255.0

        targets = np.zeros(out)
        targets[int(answ[i])] = 1

        neun.train(inputs, targets)


# test the nn

score_history = []
inp, answ = test_set
for i in tqdm(range(len(inp))):
    inputs = inp[i] / 255.0
    outputs = neun.query(inputs)[0]

    label = np.argmax(outputs)
    # print(outputs)
    # print('-'*50)
    # print(label)
    # print('-'*50)
    # print(answ[i])
    if int(label) == int(answ[i]):
        score_history.append(1)
    else:
        score_history.append(0)

# calculate performance
print(f"nn performance rate: {sum(score_history) / len(score_history)}")