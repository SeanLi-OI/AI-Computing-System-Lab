# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import sys
import numpy as np
import time
import cv2 as cv
import os

model = sys.argv[1]
#os.putenv('MLU_VISIBLE_DEVICES','')
#os.putenv('MLU_OP_PRECISION','float32')
def inference_pb(model):
    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    config = tf.ConfigProto(allow_soft_placement=True,
                      inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    config.mlu_options.data_parallelism = 1
    config.mlu_options.model_parallelism = 1
    config.mlu_options.core_num = 4
    config.mlu_options.core_version = "MLU270"
    config.mlu_options.precision = "int16"
    config.mlu_options.optype_black_list = "OneHot"

    session = tf.Session(config=config)
    i = 0

    input_1 = tf.get_default_graph().get_tensor_by_name("input_ids:0")
    input_2 = tf.get_default_graph().get_tensor_by_name("input_mask:0")
    input_3 = tf.get_default_graph().get_tensor_by_name("segment_ids:0")
    output_1 = tf.get_default_graph().get_tensor_by_name("unstack:0")

    a = np.array([101,2029,5088,2136,3421,1996,10511,2012,3565,4605,2753,1029,102,3565,4605,2753,2001,2019,2137,2374,2208,2000,5646,1996,3410,1997,1996,2120,2374,2223,1006,5088,1007,2005,1996,2325,2161,1012,1996,2137,2374,3034,1006,10511,1007,3410,7573,14169,3249,1996,2120,2374,3034,1006,22309,1007,3410,3792,12915,2484,1516,2184,2000,7796,2037,2353,3565,4605,2516,1012,1996,2208,2001,2209,2006,2337,1021,1010,2355,1010,2012,11902,1005,1055,3346,1999,1996,2624,3799,3016,2181,2012,4203,10254,1010,2662,1012,2004,2023,2001,1996,12951,3565,4605,1010,1996,2223,13155,1996,1000,3585,5315,1000,2007,2536,2751,1011,11773,11107,1010,2004,2092,2004,8184,28324,2075,1996,102])
    b = np.ones(128)
    c = np.ones(128)

    while i<20:
        print("count: ",i)
        i += 1
        start_time = time.time()
        out_ = session.run(output_1,feed_dict={input_1:[a],input_2:[b],input_3:[c]})
        end_time = time.time()
        print("inference time is: ",end_time-start_time)

inference_pb(model)
