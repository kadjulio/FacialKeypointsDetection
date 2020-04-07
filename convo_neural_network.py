from readCsvFiles import load
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

X_train, y_train = load()
print("X_train.shape == {}; X_train.min == {:.3f}; X_train.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))