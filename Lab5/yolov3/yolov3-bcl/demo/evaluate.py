#! /usr/bin/env python
# coding=utf-8

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
import time
import argparse
from core.config import cfg
from core.yolov3 import YOLOV3

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.number           = cfg.TEST.NUMBER
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.model_file       = cfg.TEST.MODEL_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL
        self.batch_size       = cfg.TEST.BATCH_SIZE

        self.core_version     = cfg.RUNTIME.CORE_VERSION
        self.precision        = cfg.RUNTIME.PRECISION
        self.data_parallelism = cfg.RUNTIME.DATA_PARALLELISM
        self.model_parallelism = cfg.RUNTIME.MODEL_PARALLELISM
        self.core_num          = cfg.RUNTIME.CORE_NUM

        if os.path.exists(self.model_file):
            print ("model is exit")
        else :
            print ("please check out model_file")
        graph = load_graph(self.model_file)
        self.input_data = graph.get_tensor_by_name("import/input/input_data:0" )
        self.pred_sbbox = graph.get_tensor_by_name("import/pred_sbbox/concat_2:0" )
        self.pred_mbbox = graph.get_tensor_by_name("import/pred_mbbox/concat_2:0" )
        self.pred_lbbox = graph.get_tensor_by_name("import/pred_lbbox/concat_2:0" )
        self.bbox_raw = graph.get_tensor_by_name("import/Yolov3DetectionOutput:0" )
        config = tf.ConfigProto(allow_soft_placement=True,
                    inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=1)
        config.mlu_options.data_parallelism = self.data_parallelism
        config.mlu_options.model_parallelism = self.model_parallelism
        config.mlu_options.core_num = self.core_num
        config.mlu_options.core_version = self.core_version
        config.mlu_options.precision = self.precision
        config.mlu_options.save_offline_model = True
        config.mlu_options.offline_model_name = "yolov3_int8.cambricon"
        self.sess  = tf.Session(config = config, graph = graph)

    def predict_bak(self, images):

        org_h = [0 for i in range(self.batch_size)]
        org_w = [0 for i in range(self.batch_size)]
        for i in range(self.batch_size):
            org_h[i], org_w[i], _ = images[i].shape

        image_data = utils.images_preporcess(images, [self.input_size, self.input_size])

        start = time.time()
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
            }
        )
        np.savetxt("pred_sbbox.txt",pred_sbbox.flatten())
        np.savetxt("pred_mbbox.txt",pred_mbbox.flatten())
        np.savetxt("pred_lbbox.txt",pred_lbbox.flatten())
        end = time.time()

        batch_bboxes = []
        for idx in range(self.batch_size):
            pred_bbox = np.concatenate([np.reshape(pred_sbbox[idx], (-1, 5 + self.num_classes)),
                                        np.reshape(pred_mbbox[idx], (-1, 5 + self.num_classes)),
                                        np.reshape(pred_lbbox[idx], (-1, 5 + self.num_classes))], axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, (org_h[idx], org_w[idx]), self.input_size, self.score_threshold)
            batch_bboxes.append(utils.nms(bboxes, self.iou_threshold))
        print("bbox num : ",len(batch_bboxes))
        exit(0)
        return batch_bboxes, (end - start)

    def predict(self, images):

        org_h = [0 for i in range(self.batch_size)]
        org_w = [0 for i in range(self.batch_size)]
        for i in range(self.batch_size):
            org_h[i], org_w[i], _ = images[i].shape

        image_data, dh, dw, scale = utils.images_preporcess(images, [self.input_size, self.input_size])
        start = time.time()
        bbox_raw = self.sess.run(
            self.bbox_raw,
            feed_dict={
                self.input_data: image_data,
            }
        )
        end = time.time()
        print("inference time include postprocess is: ", (end-start) * 1000)
        batch_bboxes = []
        num_batches = 1
        num_boxes = 1024 * 2
        predicts_mlu = bbox_raw.flatten()
        for batchIdx in range(num_batches):
            result_boxes = int(predicts_mlu[batchIdx * (64 + num_boxes * 7)])
            current_bboxes = []
            for i in range(result_boxes):
                batchId = predicts_mlu[i * 7 + 0 + 64 + batchIdx * (64 + num_boxes * 7)]
                classId  = predicts_mlu[i * 7 + 1 + 64 + batchIdx * (64 + num_boxes * 7)]
                score    = predicts_mlu[i * 7 + 2 + 64 + batchIdx * (64 + num_boxes * 7)]
                x1       = 1.0*(predicts_mlu[i * 7 + 3 + 64 + batchIdx * (64 + num_boxes * 7)] * self.input_size - dw)/scale
                y1       = 1.0*(predicts_mlu[i * 7 + 4 + 64 + batchIdx * (64 + num_boxes * 7)] * self.input_size - dh)/scale
                x2       = 1.0*(predicts_mlu[i * 7 + 5 + 64 + batchIdx * (64 + num_boxes * 7)] * self.input_size - dw)/scale
                y2       = 1.0*(predicts_mlu[i * 7 + 6 + 64 + batchIdx * (64 + num_boxes * 7)] * self.input_size - dh)/scale
                bbox = [x1, y1, x2, y2, score, classId]
                current_bboxes.append(np.array(bbox))
            batch_bboxes.append(current_bboxes)
        return batch_bboxes, (end - start)

    def evaluate(self):
        predicted_dir_path = self.write_image_path + '/mAP/predicted'
        ground_truth_dir_path = self.write_image_path + '/mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.makedirs(predicted_dir_path)
        os.makedirs(ground_truth_dir_path)

        batch_idx = 0
        alltime_sess = 0
        start = []
        end = []
        start_end2end = 0.0
        start_post = 0.0
        end_post = 0.0
        alltime_end2end = 0.0
        alltime_prepare = 0.0
        alltime_post = 0.0
        alltime_sess_run = 0.0

        batch_count = 0
        batch_image = []
        batch_image_name = []
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                if batch_idx == 0:
                    start_end2end = time.time()
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                batch_image_name.append(image_name)
                image = cv2.imread(image_path)
                batch_image.append(image)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                if batch_idx < self.batch_size - 1:
                    batch_idx += 1
                    continue

                print("=> Predicting %d th batch images." % (batch_count + 1))
                start.append(time.time())
                bboxes_pr, sess_run_time = self.predict(batch_image)
                end.append(time.time())
                if batch_count > 0:
                    alltime_sess_run += sess_run_time
                    duration_time = (end[batch_count] - start[batch_count])
                    alltime_sess += duration_time
                    alltime_prepare = alltime_prepare + (start[batch_count] - start_end2end)
                if self.write_image:
                    for idx in range(self.batch_size):
                        image = utils.draw_bbox(batch_image[idx], bboxes_pr[idx], show_label=self.show_label)
                        print("######### SAVE IMAGE ,",self.write_image_path+"/"+batch_image_name[idx])
                        cv2.imwrite(self.write_image_path+"/"+batch_image_name[idx], image)

                for idx in range(self.batch_size):
                    predict_result_path = os.path.join(predicted_dir_path,
                            str(batch_count * self.batch_size + idx) + '.txt')
                    with open(predict_result_path, 'w') as f:
                        for bbox in bboxes_pr[idx]:
                            coor = np.array(bbox[:4], dtype=np.int32)
                            score = bbox[4]
                            class_ind = int(bbox[5])
                            class_name = self.classes[class_ind + 1]
                            score = '%.4f' % score
                            xmin, ymin, xmax, ymax = list(map(str, coor))
                            bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                            f.write(bbox_mess)
                if batch_count > 0:
                    temp = time.time()
                    alltime_end2end = alltime_end2end + (temp - start_end2end)
                    alltime_post = alltime_post + temp - end[batch_count]
                batch_count += 1
                if self.number < (batch_count + 1) * self.batch_size:
                    print("we have evaluated %d batch images"%(batch_count))
                    break
                batch_idx = 0
                batch_image = []
                batch_image_name = []
        if(self.number > 1):
            print('latency: %f (ms)' % (alltime_sess_run * 1000 / (batch_count - 1)))
            print('throughput: %f' % (((batch_count - 1) * self.batch_size) / alltime_sess_run))

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    print("model_file",model_file)
    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--result_path", help="result path to write")
    parser.add_argument("--records", help="records to be processed")
    parser.add_argument("--number", type=int, help="number of records to be processed")
    parser.add_argument("--core_version", type=str, help="MLU100/MLU270", default="MLU100")
    parser.add_argument("--precision", type=str, help="float/int8", default="float")
    parser.add_argument("--data_parallelism", type=int, help="data_parallelism")
    parser.add_argument("--model_parallelism", type=int, help="model_parallelism")
    parser.add_argument("--core_num", type=int, help="core_num")
    parser.add_argument("--input_size", type=int, help="choose 416 or 544", default=416)
    parser.add_argument("--batch_size", type=int, help="batch size")
    args = parser.parse_args()
    if args.graph:
        cfg.TEST.MODEL_FILE = args.graph
    if args.result_path:
        cfg.TEST.WRITE_IMAGE_PATH = args.result_path
    if args.records:
        cfg.TEST.ANNOT_PATH = args.records
    if args.number:
        cfg.TEST.NUMBER = args.number
    if args.core_version:
        cfg.RUNTIME.CORE_VERSION = args.core_version
    if args.precision:
        cfg.RUNTIME.PRECISION = args.precision
    if args.data_parallelism:
        cfg.RUNTIME.DATA_PARALLELISM = args.data_parallelism
    if args.model_parallelism:
        cfg.RUNTIME.MODEL_PARALLELISM = args.model_parallelism
    if args.core_num:
        cfg.RUNTIME.CORE_NUM = args.core_num
    if args.input_size:
        cfg.TEST.INPUT_SIZE = args.input_size
    if args.batch_size:
        cfg.TEST.BATCH_SIZE = args.batch_size

    YoloTest().evaluate()
