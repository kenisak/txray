import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP

basepath = 'minst'

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte' % kind)
    print("labelpath {}". format(labels_path))
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte' % kind)
    print("labelpath {}".format(images_path))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
            len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

#X_train, y_train = load_mnist(basepath, kind='train')

"""
# Ladda datat från disk
X_test, y_test = load_mnist(basepath, kind='t10k')
print('Rows: %d, columns: %d'
      % (X_test.shape[0], X_test.shape[1]))

# Spara undan resultatet på disk.
# save multidimensional arrays to disk is NumPy's savez function
np.savez_compressed('mnist_scaled.npz',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)
"""
# Ladda datat från disk som multidimensionell array
mnist = np.load('mnist_scaled.npz')
X_train, y_train, X_test, y_test = [mnist[f] for
                                    f in mnist.files]

print('Rows: %d, columns: %d'
      % (X_train.shape[0], X_train.shape[1]))
"""
# Visa siffrorna 0-9 i originalformatet 28x28 pixlar
fig, ax = plt.subplots(nrows=2, ncols=5,
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Vi kollar hur olika siffran 7 kan se ut bland de handskrivna siffrorna:
fig, ax = plt.subplots(nrows=9,
                       ncols=9,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()
for i in range(81):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""

nn = NeuralNetMLP(n_hidden=100,
    l2 = 0.1,
    epochs = 200,
    eta = 0.0005,
    minibatch_size = 100,
    shuffle = True,
    seed = 1)

print(nn.fit(X_train=X_train[:55000],
      y_train=y_train[:55000],
      X_valid=X_train[55000:],
      y_valid=y_train[55000:]))

# In our NeuralNetMLP implementation, we also defined an eval_
# attribute that collects the cost, training, and validation
# accuracy for each epoch so that we can visualize the results
# using Matplotlib:
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

# Next, let's take a look at the training and validation accuracy:
plt.plot(range(nn.epochs), nn.eval_['train_acc'],
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Finally, let's evaluate the generalization performance of the model
# by calculating the prediction accuracy on the test set:
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
      .astype(np.float) / X_test.shape[0])
print('Training accuracy: %.2f%%' % (acc * 100))

# Lastly, let's take a look at some of the images that our MLP struggles with:
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab= y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,
                 cmap='Greys',
                 interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'
                    % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()