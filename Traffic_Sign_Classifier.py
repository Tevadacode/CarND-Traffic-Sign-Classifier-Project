# read training, validation, test data
import pickle

train_file = "./traffic-sign/train.p"
validation_file = "./traffic-sign/valid.p"
test_file = "./traffic-sign/test.p"

with open(train_file, "rb") as f:
    train = pickle.load(f)
with open(validation_file, "rb") as f:
    valid = pickle.load(f)
with open(test_file, "rb") as f:
    test = pickle.load(f)

X_train, y_train = train["features"], train["labels"]
X_valid, y_valid = valid["features"], valid["labels"]
X_test, y_test = test["features"], test["labels"]

n_train = len(X_train)
n_test = len(X_test)
n_valid = len(X_valid)

n_classes = len(set(y_train))

# display number of training, validation, test examples, and unique classes
print("Number of training examples = {}".format(n_train))
print("Number of validation examples = {}".format(n_valid))
print("Number of test examples = {}".format(n_test))
print("Number of classes = {}".format(n_classes))

# Data exploration visualization code goes here
import random
import matplotlib.pyplot as plt

import csv
signnames = {}
with open("./signnames.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        k, v = row
        signnames[k] = v


index = random.randint(0, n_train)
image = X_train[index]
plt.figure(figsize=(1,1))
plt.title(signnames[str(y_train[index])])
plt.imshow(image)
plt.show()
'''
import numpy as np
# Normalize data
X_train = np.subtract(np.divide(X_train, 127), 1)
X_valid = np.subtract(np.divide(X_valid, 127), 1)
X_test = np.subtract(np.divide(X_test, 127), 1)

from helper import *
from sklearn.utils import shuffle

EPOCHS = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("start training ...")
    print()
    num_examples = len(X_train)
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})

        validation_loss, validation_acc = evaluate(X_valid, y_valid)   
        print("EPOCH {} ...".format(i + 1))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Acc = {:.3f}".format(validation_acc))
        print()

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    save_path = saver.save(sess, saved_file)
    print ("Model saved at {}".format(save_path))

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, saved_file)

    test_loss, test_acc = evaluate(X_test, y_test)
    print("Train Loss = {:.3f}".format(test_loss))
    print("Test Acc = {:.3f}".format(test_acc))
'''
