import tensorflow as tf
import numpy as np


def identity_block(X, f, filters):

    F1, F2, F3 = filters
    X_shortcut = X   
    
    c=tf.convert_to_tensor(X.shape[3])   
    W1=tf.Variable(tf.random_normal([1,1,c,F1],stddev=0.01))
    X=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="VALID")
    print(X.shape)
    X=tf.layers.batch_normalization(X,axis=3)
    X=tf.nn.relu(X)

    c=tf.convert_to_tensor(X.shape[3])
    W2=tf.Variable(tf.random_normal([f,f,c,F2],stddev=0.01))
    X=tf.nn.conv2d(X,W2,strides=[1,1,1,1],padding="SAME")
    print(X.shape)
    X=tf.layers.batch_normalization(X,axis=3)
    X=tf.nn.relu(X)

    c=tf.convert_to_tensor(X.shape[3])
    W3=tf.Variable(tf.random_normal([1,1,c,F3],stddev=0.01))
    X=tf.nn.conv2d(X,W3,strides=[1,1,1,1],padding="VALID")
    print(X.shape)
    X=tf.layers.batch_normalization(X,axis=3)

    X=tf.add(X,X_shortcut)
    X=tf.nn.relu(X)
    
    return X
"""
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6],
                       stage = 1, block = 'a')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run([A],feed_dict = {A_prev : X})
    sess.close()

    
    #test.run(tf.global_variables_initializer())
    #out = test.run([A], feed_dict={A_prev: X})
    print("out = " + str(out[0][1][1][0]))
    #print(out)
    #out = np.array(out)
    #print(out.shape)
"""
def convolutional_block(X, f, filters, s = 2):
    F1, F2, F3 = filters

    X_shortcut = X

    c1=tf.convert_to_tensor(X.shape[3])   
    W1=tf.Variable(tf.random_normal([1,1,c1,F1],stddev=0.01))
    X=tf.nn.conv2d(X,W1,strides=[1,s,s,1],padding="VALID")
    X=tf.layers.batch_normalization(X,axis=3)
    X=tf.nn.relu(X)
    print(X.shape)
    
    c2=tf.convert_to_tensor(X.shape[3])
    W2=tf.Variable(tf.random_normal([f,f,c2,F2],stddev=0.01))
    X=tf.nn.conv2d(X,W2,strides=[1,1,1,1],padding="SAME")
    X=tf.layers.batch_normalization(X,axis=3)
    X=tf.nn.relu(X)
    print(X.shape)
    
    c3=tf.convert_to_tensor(X.shape[3])
    W3=tf.Variable(tf.random_normal([1,1,c3,F3],stddev=0.01))
    X=tf.nn.conv2d(X,W3,strides=[1,1,1,1],padding="VALID")
    X=tf.layers.batch_normalization(X,axis=3)
    print(X.shape)
    
    W4=tf.Variable(tf.random_normal([1,1,c1,F3],stddev=0.01))
    X_shortcut=tf.nn.conv2d(X_shortcut,W4,strides=[1,s,s,1],padding="VALID")
    X_shortcut=tf.layers.batch_normalization(X_shortcut,axis=3)
    print(X_shortcut.shape)#(3, 2, 2, 6)
    
    X=tf.add(X,X_shortcut)
    X=tf.nn.relu(X)
    
    return X
"""
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6],
                       stage = 1, block = 'a')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run([A],feed_dict = {A_prev : X})
    sess.close()
    print("out = " + str(out[0][1][1][0]))
"""


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    X_input = tf.keras.Input(input_shape)
    #X_input=tf.placeholder(tf.float32,shape=input_shape)
    
    print(X_input)
    print(X_input.shape)

    X=tf.pad(X_input,[[0,0],[3,3],[3,3],[0,0]],"CONSTANT")
    
    W1=tf.Variable(tf.random_normal([7,7,3,64],stddev=0.01))
    X=tf.nn.conv2d(X,W1,strides=[1,2,2,1],padding="VALID")
    X=tf.layers.batch_normalization(X,axis=3)
    X=tf.nn.relu(X)
    X=tf.layers.max_pooling2d(X,pool_size=(3,3),strides=(2,2))
    #X=tf.layers.MaxPooling2D(input=X,pool_size=(3,3),strides=(2,2))
   
    X=convolutional_block(X,f=3,filters=[64,64,256],s=1)
    X=identity_block(X,3,[64,64,256])
    X=identity_block(X,3,[64,64,256])

    X=convolutional_block(X,f=3,filters=[128,128,512],s=2)
    X=identity_block(X,3,[128,128,512])
    X=identity_block(X,3,[128,128,512])
    X=identity_block(X,3,[128,128,512])

    X=convolutional_block(X,f=3,filters=[256,256,1024],s=2)    
    X=identity_block(X,3,[256,256,1024])
    X=identity_block(X,3,[256,256,1024])
    X=identity_block(X,3,[256,256,1024])
    X=identity_block(X,3,[256,256,1024])
    X=identity_block(X,3,[256,256,1024])

    X=convolutional_block(X,f=3,filters=[512,512,2048],s=2)   
    X=identity_block(X,3,[512,512,2048])
    X=identity_block(X,3,[512,512,2048])

    X=tf.layers.average_pooling2d(X,pool_size=(2,2),strides=(2,2))
    X=tf.contrib.layers.flatten(X)

    X=tf.nn.softmax(X)
    

model = ResNet50(input_shape = (64, 64, 3), classes = 6)









    
