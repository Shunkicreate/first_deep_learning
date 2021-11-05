import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TowLayerNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)
train_size = x_train.shape[0]


# hyper paramater
iters_num = 10000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Continuation of 1 epoch pea
iter_per_epoch = max(train_size/batch_size, 1)

network = TowLayerNet(input_size=784, hidden_size=50, output_size=10)
print(vars(network))
epoch=0


for i in range(iters_num):
    # get minibatch
    print("i: {0}".format(i))
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch,t_batch) #highspeed version

    # update paramater
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] = learning_rate*grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # calculte recognition accuracy for each 1 epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch {0}".format(epoch))
        epoch+=1
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

print(vars(network))
# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
