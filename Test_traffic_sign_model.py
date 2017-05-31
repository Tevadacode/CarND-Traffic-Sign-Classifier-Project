def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=tf.get_default_session(),feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            
    plt.show()

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
    '''
    my_tensor1 = sess.graph.get_tensor_by_name('conv1:0')
    my_tensor2 = sess.graph.get_tensor_by_name('conv2:0')
    outputFeatureMap(images, my_tensor1,activation_min=1)
    outputFeatureMap(images, my_tensor2)
    '''
    
nrows = len(original_img)
plt_number = 1

for indImage in zip(index, original_img):    
    plt.subplot(3, 3, plt_number)
    plt.imshow(indImage[1])
    plt.xlabel(signnames[str(indImage[0])])
    plt_number = plt_number + 1
plt.show()
