import tensorflow as tf

input_channels = 3
mu = 0
sigma = 0.095
weights = [
    tf.Variable(tf.truncated_normal(shape=(5, 5, input_channels, 6), mean = mu, stddev = sigma), name='conv1_W'),
    tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name='conv2_W'),
    tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name='fc1_W'),
    tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name='fc2_W'),
    tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma), name='fc3_W')
    ]
bias = [
    tf.Variable(tf.zeros(6), name='conv1_b'),
    tf.Variable(tf.zeros(16), name='conv2_b'),
    tf.Variable(tf.zeros(120), name='fc1_b'),
    tf.Variable(tf.zeros(84), name='fc2_b'),
    tf.Variable(tf.zeros(43), name='fc3_b')
]
from tensorflow.contrib.layers import flatten


keep_prob = tf.placeholder(tf.float32)

def LeNet(x):    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1   = tf.nn.conv2d(x, weights[0], strides=[1, 1, 1, 1], padding='VALID', name='conv1') + bias[0]

    # Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2   = tf.nn.conv2d(conv1, weights[1], strides=[1, 1, 1, 1], padding='VALID', name='conv2') + bias[1]
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    conv2    = tf.nn.dropout(conv2, keep_prob)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1   = tf.matmul(fc0, weights[2]) + bias[2]
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    fc1    = tf.nn.dropout(fc1, keep_prob)
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2    = tf.matmul(fc1, weights[3]) + bias[3]
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2, keep_prob)
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.matmul(fc2, weights[4]) + bias[4]
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, input_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 512

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss/ num_examples, total_accuracy / num_examples

saved_file = './test.ckpt'

def predict(sign):
    sess = tf.get_default_session()
    print(sign.shape)
    prop = sess.run(logits, feed_dict={x:sign, keep_prob:1.0})
    result = tf.nn.softmax(prop)   
    index = sess.run(tf.argmax(result, 1))
    return index
