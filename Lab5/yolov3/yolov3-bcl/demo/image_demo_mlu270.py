import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "../yolov3_int8.pb"
image_path      = "./docs/images/road.jpeg"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
#print(image_data)
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

#########
config = tf.ConfigProto(allow_soft_placement=True,
            inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)
config.mlu_options.data_parallelism = 1
config.mlu_options.model_parallelism = 1
config.mlu_options.core_num = 4
config.mlu_options.core_version = "MLU270"
config.mlu_options.precision = "int8"
config.mlu_options.save_offline_model = False
config.mlu_options.offline_model_name = "yolov3_int8.cambricon"
#########
with tf.Session(config = config, graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes)
#image = Image.fromarray(image)
#image.show()
cv2.imwrite ('result_mlu270.jpg',image)
