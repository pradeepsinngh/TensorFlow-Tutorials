import tensorflow as tf
import numpy as np

# create tensor
hello = tf.constant("hello world")
print(hello)

# To access a Tensor value, call numpy
print(hello.numpy())
