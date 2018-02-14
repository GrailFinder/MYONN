from nn import NeuralNetwork
from mnist_play import get_mnist, get_local_mnist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# nn for mnist data
out = 10
neun = NeuralNetwork(inp=784, hid=200, out=out, alpha=.1)

#  get data
train_set, valid_set, test_set = get_mnist()
#train_set, test_set = get_local_mnist()


# train
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
    

sns.boxplot(data=train_err)
plt.show()
# calculate performance
print(f"nn performance rate: {sum(score_history) / len(score_history)}")