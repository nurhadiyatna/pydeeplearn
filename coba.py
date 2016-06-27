import gzip
import cPickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

mnist_dataset = 'mnist.pkl.gz'

f = gzip.open(mnist_dataset, 'rb')
dataset = cPickle.load(f)
f.close()
# print "Dataset length:",len(dataset)
# print "Dataset:\n",dataset

train_set, valid_set, test_set = dataset
# print "Train set length:",len(train_set)
# print "Train Set:\n",train_set
# print "Valid set length:",len(valid_set)
# print "Valid Set:\n",valid_set
# print "Test set length:",len(test_set)
# print "Test set:",test_set

train_set_x, train_set_y = train_set
# print "Train set x shape:",train_set_x.shape
# print "Train set x:\n",train_set_x
# print "Train set y shape:",train_set_y.shape
# print "Train set y:\n",train_set_y

# print "Train set sample (index 0), label:",train_set_y[0]
# plt.imshow(train_set_x[0].reshape((28, 28)), cmap = cm.Greys_r)
# plt.show()

valid_set_x, valid_set_y = valid_set
print "Valid set x shape:",valid_set_x.shape
print "Valid set x:\n",valid_set_x
print "Valid set y shape:",valid_set_y.shape
print "Valid set y:\n",valid_set_y

print "Valid set sample (index 0), label:",valid_set_y[0]
plt.imshow(valid_set_x[0].reshape((28, 28)), cmap = cm.Greys_r)
plt.show()

test_set_x, test_set_y = test_set
print "Test set x shape:",test_set_x.shape
print "Test set x:\n",test_set_x
print "Test set y shape:",test_set_y.shape
print "Test set y:\n",test_set_y
#print (test_set_x[0])
print "Test set sample (index 0), label:",test_set_y[0]
plt.imshow(test_set_x[0].reshape((28, 28)), cmap = cm.Greys_r)
plt.show()