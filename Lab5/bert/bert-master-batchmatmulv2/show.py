import tensorflow as tf
import sys
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
model = sys.argv[1]

def show_graph(model):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    tfconfig.device_count["CPU"]=1

    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        for node in graph_def.node:
            print(node.op)
    with tf.Session(config=tfconfig) as session:
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        #for tensor in tf.get_default_graph().as_graph_def().node:
        #    print tensor
        summaryWriter = tf.summary.FileWriter('log_/',session.graph)

def change_op_name(model):
    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        for tensor in graph_def.node:
            #print(tensor.name)
            if tensor.name == "moments/SquaredDifference":
                print("change op name")
                tensor.op = "SquaredDifference_new"
        with tf.gfile.FastGFile("style_SquaredDifference_new.pb","w") as f:
            f.write(graph_def.SerializeToString())
            f.close()

def change_op_name_new(model):
    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        part1_graph_def = graph_pb2.GraphDef()
        part2_graph_def = graph_pb2.GraphDef()
        #tmp_graph_def = graph_pb2.GraphDef()
        #for node in graph_def.node:
            #print(tensor.name)
        input_ = node_def_pb2.NodeDef()
        input_.name = "moments/SquaredDifference_new"
        input_.op = "Placeholder"
        dtype = attr_value_pb2.AttrValue(type="DT_FLOAT")
        input_.attr["dtype"].CopyFrom(dtype)
        #input_.attr["Shape"].CopyFrom(node.attr["shape"])
        #tmp_graph_def.node.extend([input_])
        part2_graph_def.node.extend([input_])
        for node in graph_def.node:
            if node.name == "moments/SquaredDifference":
                with tf.gfile.FastGFile("new_pb_model/style_SquaredDifference_part1.pb","w") as f:
                    f.write(part1_graph_def.SerializeToString())
                    f.close()
                part1_graph_def = part2_graph_def
                #break
                #tmp_graph_def.node.extend([input_])
            if node.name == "moments/variance":
                node.input[0] = "moments/SquaredDifference_new"#moments/variance
                    #print(node)
                    #tmp_graph_def.node.extend([input_])
            if node.name != "moments/SquaredDifference":
                part1_graph_def.node.extend([node])
        #print(part2_graph_def)
        #with tf.gfile.FastGFile("new_pb_model/style_SquaredDifference_part1.pb","w") as f:
        #    f.write(part1_graph_def.SerializeToString())
        #    f.close()
        with tf.gfile.FastGFile("new_pb_model/style_SquaredDifference_part2.pb","w") as f:
            f.write(part1_graph_def.SerializeToString())
            f.close()

def change_op_name_new2(model):
    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        tmp_graph_def = graph_pb2.GraphDef()
        #for node in graph_def.node:
            #print(tensor.name)
        input_ = node_def_pb2.NodeDef()
        input_.name = "moments/SquaredDifference_new"
        input_.op = "Placeholder"
        dtype = attr_value_pb2.AttrValue(type="DT_FLOAT")
        input_.attr["dtype"].CopyFrom(dtype)
        #input_.attr["Shape"].CopyFrom(node.attr["shape"])
        tmp_graph_def.node.extend([input_])
        for node in graph_def.node:
            if node.name == "moments/SquaredDifference":
                #tmp_graph_def.node.extend([input_])
                continue
            if node.name == "moments/variance":
                node.input[0] = "moments/SquaredDifference_new"#moments/variance
                    #print(node)
                    #tmp_graph_def.node.extend([input_])
            if node.name != "moments/SquaredDifference":
                tmp_graph_def.node.extend([node])
        with tf.gfile.FastGFile("new_pb_model/style_SquaredDifference_delete_one_node.pb","w") as f:
            f.write(tmp_graph_def.SerializeToString())
            f.close()

show_graph(model)
#change_op_name(model)
#change_op_name_new(model)
#change_op_name_new2(model)
