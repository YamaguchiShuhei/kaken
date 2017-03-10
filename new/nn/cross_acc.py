import tensorflow as tf
import numpy as np

def accuracy_calc(y, y_, l, mask):
    correct_prediction = tf.equal(tf.argmax(y,2), tf.argmax(y_,2))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float") * mask) / tf.cast(tf.reduce_sum(l), "float")
    return accuracy

def cross_entropy_calc(y, y_, l, mask):
    cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[2]) * mask) / tf.cast(tf.reduce_sum(l), dtype=tf.float32) 
    return cross_entropy

def cross_acc(params, y, y_, l):
    mask = tf.sequence_mask(l, dtype=tf.float32)

    loss = cross_entropy_calc(y, y_, l, mask)
    accuracy = accuracy_calc(y, y_, l, mask)
    return loss, accuracy
