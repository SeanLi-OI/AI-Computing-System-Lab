import tensorflow as tf
import sys
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
model = sys.argv[1]

def change_op_name(model):
    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        for tensor in graph_def.node:
            if tensor.op == "BatchMatMulV2" and "Softmax" in tensor.input[0]:
                print("change op name")
                tensor.op = "BatchMatMul"
        with tf.gfile.FastGFile("frozen_model_matmul_softmax.pb","w") as f:
            f.write(graph_def.SerializeToString())
            f.close()


change_op_name(model)
