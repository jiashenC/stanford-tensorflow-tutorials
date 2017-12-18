"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.random_uniform([], -1, 1, dtype=tf.int32)
    y = tf.random_uniform([], -1, 1, dtype=tf.int32)
    add = lambda: tf.add(x, y)
    mul = lambda: tf.multiply(x, y)
    default = lambda: tf.constant(2)
    out = tf.case({tf.less(x, y): add, tf.greater(x, y): mul}, default=default)
    print sess.run(out)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.constant([[0, -2, 1], [0, 1, 2]])
    y = tf.zeros_like(x)
    out = tf.equal(x, y)
    print sess.run(out)

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.constant([
        29.05088806, 27.61298943, 31.19073486, 29.35532951, 30.97266006, 26.67541885, 38.08450317, 20.74983215,
        34.94445419, 34.45999146, 29.06485367, 36.01657104, 27.88236427, 20.56035233, 30.20379066, 29.51215172,
        33.71149445, 28.59134293, 36.05556488, 28.66994858
    ])
    thirty = tf.fill(x.get_shape(), 30.0)
    cond = tf.greater(x, thirty)
    where = tf.where(cond)
    out = tf.gather(x, where)
    print sess.run(out)

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.range(6)
    x_minus = tf.subtract(tf.range(6), tf.ones_like(x))
    matrix = tf.diag(tf.subtract(x, x_minus))
    print sess.run(matrix)

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.random_uniform([10, 10])
    deter = tf.matrix_determinant(x)
    print sess.run(deter)


###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
    unique = tf.unique(x)
    ele, index = sess.run(unique)
    print ele, index

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

print '#######################################################################'
with tf.Session() as sess:
    x = tf.random_normal([300])
    y = tf.random_normal([300])
    mse = lambda: tf.losses.mean_squared_error(y, x)
    absval = lambda: tf.reduce_sum(tf.abs(tf.subtract(x, y)))
    out = tf.cond(tf.reduce_mean(x - y) < 0, mse, absval)
    print sess.run(out)
