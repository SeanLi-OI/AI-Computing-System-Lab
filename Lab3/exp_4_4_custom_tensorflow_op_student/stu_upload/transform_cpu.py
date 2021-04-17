import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import argparse
import numpy as np
import cv2 as cv
import time
from power_diff_numpy import *

os.putenv('MLU_VISIBLE_DEVICES','')
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('ori_pb')
    parser.add_argument('ori_power_diff_pb')
    parser.add_argument('numpy_pb')
    args = parser.parse_args()
    return args

def run_ori_pb(ori_pb, image):
    config = tf.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    model_name = os.path.basename(ori_pb).split(".")[0]
    image_name = os.path.basename(image).split(".")[0]

    g = tf.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(ori_pb,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(image)
        X = cv.resize(img, (256, 256))
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            sess.run(tf.global_variables_initializer())

            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')

            start_time = time.time()
            ret =sess.run(output_tensor, feed_dict={input_tensor:[X]})
            end_time = time.time()
            print("C++ inference(CPU) origin pb time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_cpu.jpg',img_numpy)


def run_ori_power_diff_pb(ori_power_diff_pb, image):
    config = tf.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    model_name = os.path.basename(ori_power_diff_pb).split(".")[0]
    image_name = os.path.basename(image).split(".")[0]

    g = tf.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(ori_power_diff_pb,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(image)
        X = cv.resize(img, (256, 256))
        with tf.Session(config=config) as sess:
            # TODO：完成PowerDifference Pb模型的推理
            _________________

            start_time = time.time()
            ret =sess.run(...)
            end_time = time.time()
            print("C++ inference(CPU) time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_cpu.jpg',img_numpy)

def run_numpy_pb(numpy_pb, image):
    config = tf.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    model_name = os.path.basename(numpy_pb).split(".")[0]
    image_name = os.path.basename(image).split(".")[0]

    g = tf.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(numpy_pb,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(image)
        X = cv.resize(img, (256, 256))
        with tf.Session(config=config) as sess:
            # TODO：完成Numpy版本 Pb模型的推理
            _________________

            start_time = time.time()
            _________________
            ret = sess.run(...)
            end_time = time.time()
            print("Numpy inference(CPU) time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_cpu.jpg',img_numpy)


if __name__ == '__main__':
    args = parse_arg()
    run_ori_pb(args.ori_pb, args.image)
    run_ori_power_diff_pb(args.ori_power_diff_pb, args.image)
    run_numpy_pb(args.numpy_pb, args.image)
