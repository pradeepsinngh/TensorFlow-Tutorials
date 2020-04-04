from __future__ import print_function
import tensorflow as tf
import numpy

# define tensors constant
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# tensor operations
add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

# Access tensors value.
print("add =", add.numpy())
print("sub =", sub.numpy())
print("mul =", mul.numpy())
print("div =", div.numpy())

# Some more operations.
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

# Access tensors value.
print("mean =", mean.numpy())
print("sum =", sum.numpy())

# Matrix multiplication
matrix1 = tf.constant([[1,2,3],[3,4,5]])
matrix2 = tf.constant([[6,7,8],[9,1,3]])

matrixProd = tf.multiply(matrix1, matrix2)

# display tensor
matrixProd

# convert tensor to numpy
matrixProd.numpy()
