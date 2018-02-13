import numpy as np
import matplotlib.pyplot as plt


with open('mnist_train.csv') as lf:
    nums = []
    for i in range(100):
        nums.append(lf.readline())

#print(nums[0])

vals = nums[0].split(',')

plt.imshow(np.asfarray(vals[1:]).reshape((28, 28)),
    cmap='Greys')
plt.show()

#scaled_input = np.asfarray()