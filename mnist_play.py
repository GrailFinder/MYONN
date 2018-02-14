import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip, urllib.request, json, os

def get_mnist():
    """returns train_set, valid_set, test_set"""
    if not os.path.isfile('mnist.pkl.gz'):
        urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    
    return train_set, valid_set, test_set

def get_local_mnist():
    with open('mnist_train.csv') as lf:
        train_set = lf.readlines()

    with open('mnist_test.csv') as lf:
        test_set = lf.readlines()

    return train_set, test_set



def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)
    plt.show()

if __name__ == '__main__':
    train_set, *other_args = get_mnist()
    ran_num = [np.random.randint(0, len(train_set))]
    show_digit(train_set[0][ran_num], 'This is a {}'.format(train_set[1][ran_num]))
