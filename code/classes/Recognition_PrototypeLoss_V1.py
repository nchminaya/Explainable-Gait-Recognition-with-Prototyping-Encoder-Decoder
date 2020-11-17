# csulb-datascience
#
# Author: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#
# Class version: 1.0
# Date: July 2020
#
# Include a reference to this site if you will use this code.

import tensorflow as tf
from typing import Optional

#Return relative labels for the units
def getLabels(units, batchSize):
    i = tf.constant(0)
    vector = tf.zeros(batchSize, dtype=tf.dtypes.int32)
    condition = lambda i, v: tf.less(i, batchSize)

    def body(i, v):
        equals = tf.math.equal(units[i], units)
        mask = tf.math.reduce_min(tf.cast(equals, dtype=tf.dtypes.int32), axis=[1,2])
        duplicate = tf.cast(tf.math.greater(tf.multiply(mask,v),0), dtype=tf.dtypes.int32)
        row = tf.math.multiply(tf.bitwise.bitwise_xor(mask, duplicate), i+1)
        v = tf.add(v, row)
        return [tf.add(i, 1), v]

    # do the loop:
    r = tf.while_loop(condition, body, [i,vector])
    return(r[1])


# Computes the prototype loss.
#    Args:
#      y_true: 3-D float `Tensor`. List of prototype (mean) units. Each
#       prototype is a 2D matrix l2 normalized.
#      y_pred: 3-D float `Tensor`. List of decoded units. Each unit 
#       is a 2D matrix l2 normalized.

def prototype_loss(y_true, y_pred):
    #  |g(f(x)) - Ci|^2 
    prototype, decoded = y_true, y_pred
    difference = tf.math.subtract(decoded, prototype)
    norm_squared = tf.cast(tf.math.square(tf.norm(difference, axis=(1,2))), dtype=tf.dtypes.float32)
    
    # Obtain the mask of positives
    batch_size = tf.shape(prototype)[0]    
    labels = getLabels(prototype, batch_size)
    labels = tf.reshape(labels, [batch_size, 1])    
    adjacency = tf.math.equal(labels, tf.transpose(labels))    
    adjacency_mat = tf.cast(adjacency, dtype=tf.dtypes.float32)    
    mask_positives = adjacency_mat - tf.linalg.diag(tf.ones([batch_size]))
    
    #obtain the mask of anchors
    sum_positives = tf.math.reduce_sum(mask_positives, axis=0)
    mask_anchors =  tf.cast(tf.math.greater(sum_positives, 0.0), dtype=tf.dtypes.float32)

    # |g(f(a)) - Ci|^2    
    norm_anchors = tf.math.multiply(norm_squared, mask_anchors)
    sum_anchors = tf.math.reduce_sum(norm_anchors)
    num_anchors = tf.math.reduce_sum(mask_anchors)    
    prototype_loss = tf.math.truediv(sum_anchors, num_anchors)
    prototype_loss = tf.math.truediv(prototype_loss, 14.0)
    return prototype_loss 


class PrototypeLoss(tf.keras.losses.Loss):
    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name) #, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        return prototype_loss(y_true, y_pred)
