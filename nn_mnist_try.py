from nn import NeuralNetwork
from mnist_play import get_mnist, get_local_mnist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# nn for mnist data
out = 10
neun = NeuralNetwork(inp=784, hid=200, out=out, alpha=.1)

#  get data and train on it
train_set, valid_set, test_set = get_mnist()

epochs = 2
train_err = []

for e in tqdm(range(epochs)):
    inp, answ = train_set
    for i in tqdm(range(len(inp))):
        # scale the input data
        inputs = inp[i] / 255.0

        targets = np.zeros(out)
        targets[int(answ[i])] = 1

        train_err.append(neun.train(inputs, targets))


# test the nn
score_history = []
inp, answ = test_set
for i in tqdm(range(len(inp))):
    inputs = inp[i] / 255.0
    outputs = neun.query(inputs)

    label = np.argmax(outputs[0])
    if int(label) == int(answ[i]):
        score_history.append(1)
    else:
        score_history.append(0)
    

# calculate performance
print(f"nn performance rate: {sum(score_history) / len(score_history)}")

# train_err hax nx10 shape
plt.scatter(range(len(train_err)), [e[0] for e in train_err],
 s=1, label=0)
plt.scatter(range(len(train_err)), [e[1] for e in train_err],
 s=1, label=1)
plt.scatter(range(len(train_err)), [e[2] for e in train_err],
 s=1, label=2)
plt.scatter(range(len(train_err)), [e[3] for e in train_err],
 s=1, label=3)
plt.legend()
plt.show()