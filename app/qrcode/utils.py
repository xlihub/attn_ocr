# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import ast
import argparse


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=str,
        default='/home/attnroot/attn_ocr/app/qrcode/ppyolo_mbv3_large_qr',
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py.")
    )
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch_size for inference.")
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="device id of camera to predict.")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='fluid',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='CPU',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Deprecated, please use `--device`.")
    parser.add_argument(
        "--run_benchmark",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict a image_file repeatedly for benchmark")
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn with CPU.")
    parser.add_argument(
        "--enable_mkldnn_bfloat16",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn bfloat16 inference with CPU.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,
        help="If the model is produced by TRT offline quantitative "
        "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save visualization image results.')
    parser.add_argument(
        '--save_mot_txts',
        action='store_true',
        help='Save tracking results (txt).')
    parser.add_argument(
        '--save_mot_txt_per_img',
        action='store_true',
        help='Save tracking results (txt) for each image.')
    parser.add_argument(
        '--scaled',
        type=bool,
        default=False,
        help="Whether coords after detector outputs are scaled, False in JDE YOLOv3 "
        "True in general detector.")
    parser.add_argument(
        "--tracker_config", type=str, default=None, help=("tracker donfig"))
    parser.add_argument(
        "--reid_model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."))
    parser.add_argument(
        "--reid_batch_size",
        type=int,
        default=50,
        help="max batch_size for reid model inference.")
    parser.add_argument(
        '--use_dark',
        type=ast.literal_eval,
        default=True,
        help='whether to use darkpose to get better keypoint position predict ')
    parser.add_argument(
        "--action_file",
        type=str,
        default=None,
        help="Path of input file for action recognition.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Temporal size of skeleton feature for action recognition.")
    parser.add_argument(
        "--random_pad",
        type=ast.literal_eval,
        default=False,
        help="Whether do random padding for action recognition.")
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether save detection result to file using coco format")

    return parser


class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self, with_tracker=False):
        super(Timer, self).__init__()
        self.with_tracker = with_tracker
        self.preprocess_time_s = Times()
        self.inference_time_s = Times()
        self.postprocess_time_s = Times()
        self.tracking_time_s = Times()
        self.img_num = 0

    def info(self, average=False):
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            total_time = total_time + track_time
        total_time = round(total_time, 4)
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))
        preprocess_time = round(pre_time / max(1, self.img_num),
                                4) if average else pre_time
        postprocess_time = round(post_time / max(1, self.img_num),
                                 4) if average else post_time
        inference_time = round(infer_time / max(1, self.img_num),
                               4) if average else infer_time
        tracking_time = round(track_time / max(1, self.img_num),
                              4) if average else track_time

        average_latency = total_time / max(1, self.img_num)
        qps = 0
        if total_time > 0:
            qps = 1 / average_latency
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, qps))
        if self.with_tracker:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}, tracking_time(ms): {:.2f}".
                format(preprocess_time * 1000, inference_time * 1000,
                       postprocess_time * 1000, tracking_time * 1000))
        else:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}".
                format(preprocess_time * 1000, inference_time * 1000,
                       postprocess_time * 1000))

    def report(self, average=False):
        dic = {}
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        dic['preprocess_time_s'] = round(pre_time / max(1, self.img_num),
                                         4) if average else pre_time
        dic['inference_time_s'] = round(infer_time / max(1, self.img_num),
                                        4) if average else infer_time
        dic['postprocess_time_s'] = round(post_time / max(1, self.img_num),
                                          4) if average else post_time
        dic['img_num'] = self.img_num
        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            dic['tracking_time_s'] = round(track_time / max(1, self.img_num),
                                           4) if average else track_time
            total_time = total_time + track_time
        dic['total_time_s'] = round(total_time, 4)
        return dic


def get_current_memory_mb():
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    import pynvml
    import psutil
    import GPUtil
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    gpu_percent = 0
    gpus = GPUtil.getGPUs()
    if gpu_id is not None and len(gpus) > 0:
        gpu_percent = gpus[gpu_id].load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return round(cpu_mem, 4), round(gpu_mem, 4), round(gpu_percent, 4)
