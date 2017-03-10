import tensorflow as tf
import numpy as np

def lattice_accuracy_calc(out, gold, l, mask):
    lattice_correct_prediction = tf.equal(tf.sign(out - 0.5), tf.sign(gold - 0.5))
    lattice_accuracy = tf.reduce_sum(tf.cast(lattice_correct_prediction, "float") * mask) / tf.cast(tf.reduce_sum(l), "float")
    return lattice_accuracy

def lattice_cross_entropy_calc(out, gold, l, mask):
    cross_entropy = -tf.reduce_sum(gold * tf.log(out+1e-8) * mask + (1 - gold) * tf.log(1 - out+1e-8) * mask) / tf.cast(tf.reduce_sum(l), dtype=tf.float32)
    return cross_entropy

def lattice_cross_acc(params, out, gold, l):
    mask = tf.sequence_mask(l, dtype=tf.float32)

    loss = lattice_cross_entropy_calc(out, gold, l, mask)
    accuracy = lattice_accuracy_calc(out, gold, l, mask)
    return loss, accuracy
