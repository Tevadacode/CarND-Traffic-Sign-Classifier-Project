
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

import numpy as np
import glob
import cv2
image_paths = glob.glob("./test_images/*.jpg")
images = []
original_img = []
for image in image_paths:
    file_name = image.split('\\')[-1]
    relative_path = './test_images/' + file_name
    img = cv2.imread(relative_path)
    img = cv2.resize(img, (32,32))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img.append(img)
    images.append(img)

images = np.subtract(np.divide(images, 127), 1)
images = np.reshape(images, (len(images), 32,32, 3)).astype(np.float32)

from helper import *

with tf.Session() as sess:
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
        saver.restore(sess, saved_file)
    
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

import numpy as np
import glob
import cv2
image_paths = glob.glob("./test_images/*.jpg")
images = []
original_img = []
for image in image_paths:
    file_name = image.split('\\')[-1]
    relative_path = './test_images/' + file_name
    img = cv2.imread(relative_path)
    img = cv2.resize(img, (32,32))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img.append(img)
    images.append(img)

images = np.subtract(np.divide(images, 127), 1)
images = np.reshape(images, (len(images), 32,32, 3)).astype(np.float32)

from helper import *

with tf.Session() as sess:
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
        saver.restore(sess, saved_file)
    index = predict(images)
    my_tensor1 = sess.graph.get_tensor_by_name('conv1:0')
    my_tensor2 = sess.graph.get_tensor_by_name('conv2:0')
    test = my_tensor1.eval(session=sess, feed_dict={x:images, keep_prob:1.0})
    print (test.shape)
