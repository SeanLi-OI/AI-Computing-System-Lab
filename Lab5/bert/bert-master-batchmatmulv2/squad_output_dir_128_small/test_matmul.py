import tensorflow as tf
import time
import numpy as np
import os

#a = np.ones((1,12,128,64))
#b = np.ones((1,12,64,128))

A = tf.Variable(tf.ones([1,12,128,64], dtype=tf.float32))
B = tf.Variable(tf.ones([1,12,64,128], dtype=tf.float32))

A = tf.Variable(tf.ones([128,64], dtype=tf.float32))
B = tf.Variable(tf.ones([64,128], dtype=tf.float32))
C = tf.matmul(A,B)
#B = tf.Variable(tf.ones([1,12,128,64], dtype=tf.float32))
#C = tf.add(A,B)

os.putenv('MLU_VISIBLE_DEVICES','')

def test():
    config = tf.ConfigProto(allow_soft_placement=True,
                      inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    config.mlu_options.data_parallelism = 1
    config.mlu_options.model_parallelism = 1
    config.mlu_options.core_num = 1
    config.mlu_options.core_version = "MLU270"
    config.mlu_options.precision = "int16"
    session = tf.Session(config=config)
    init_op = tf.initialize_all_variables()
    session.run(init_op)
    a = session.run(A)
    b = session.run(B)
    for i in range(100):
        start_time = time.time()
        out_ = session.run(C,feed_dict={A:a,B:b})
        end_time = time.time()
        print("matmul time is: ",end_time - start_time)
    print(out_.shape)
test() 
