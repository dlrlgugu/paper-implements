import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
#print(model)

content_image = scipy.misc.imread("dog.jpg")
imshow(content_image)
#plt.show(content_image)

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C,[n_H*n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)

    return J_content
"""
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))
"""
style_image = scipy.misc.imread("style.jpg")
imshow(style_image)
#plt.show(style_image)

def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA
"""
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = " + str(GA.eval()))
"""


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S,[n_C,n_H*n_W]))
    a_G = tf.transpose(tf.reshape(a_S,[n_C,n_H*n_W]))

    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*n_C**2*(n_H*n_W)**2)

    return J_style_layer
"""

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    print("J_style_layer = " + str(J_style_layer.eval()))
"""

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0

    for layer_name , coeff in STYLE_LAYERS:
        out = model[layer_name]

        a_S = sess.run(out)

        a_G = out

        J_style_layer = compute_layer_style_cost(a_S,a_G)

        J_style += coeff*J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha*J_content + beta*J_style
    return J
"""
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))
"""



# Reset the graph
tf.reset_default_graph()

# Start interactive session

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

config.gpu_options.allocator_type ='BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.50

sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

im = Image.open("dog.jpg")
img=im.resize((400,300))
img.save("dog.jpg","JPEG")
content_image = scipy.misc.imread("dog.jpg")
content_image = reshape_and_normalize_image(content_image)

im = Image.open("style.jpg")
img=im.resize((400,300))
img.save("style.jpg","JPEG")
style_image = scipy.misc.imread("style.jpg")
style_image = reshape_and_normalize_image(style_image)


generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model("imagenet-vgg-verydeep-19.mat")


# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


J = total_cost(J_content,J_style,alpha=10,beta=40)


optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 1):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        _ = sess.run(train_step)
        generated_image = sess.run(model['input'])
        """    
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image(str(i) + ".png", generated_image)
        """
        Jt, Jc, Js = sess.run([J, J_content, J_style])
        save_image('generated_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)




    
