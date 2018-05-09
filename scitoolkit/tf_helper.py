from scitoolkit.py23 import *  # py2/3 compatibility

import tensorflow as tf
from keras import backend as k_bkend

# https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
GPU = True
num_cores = 4
if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0  # TODO
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': num_CPU, 'GPU': num_GPU})
session = tf.Session(config=config)
k_bkend.set_session(session)
