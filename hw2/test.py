# Disable the CPU instructions support warning by not enabling AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow code snippet
import tensorflow as tf
a = tf.constant([5.0])
sess = tf.InteractiveSession()
a_val = sess.run(a)
print(a_val)
