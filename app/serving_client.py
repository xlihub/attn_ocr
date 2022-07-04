import grpc
import logging

try:
    from paddle_serving_server_gpu.pipeline import PipelineClient
except ImportError:
    from paddle_serving_server.pipeline import PipelineClient
import numpy as np
import requests
import json
import cv2
import base64
import os
import app.utils as utils

logger = logging.getLogger(__name__)


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


class PaddleClient:
    def __init__(self):
        self.client = PipelineClient()
        self.client.connect(['127.0.0.1:18091'])

    def predict(self, img_base64):
        image = utils.read_img_from_base64(img_base64, 'base64')
        # img_dir = "/home/xli/ppocr/imgs/1/"
        # if os.path.exists(img_dir):
        #     print(os.path)
        # else:
        #     print(os.path.sys.prefix)
        # for img_file in os.listdir(img_dir):
        #     with open(os.path.join(img_dir, img_file), 'rb') as file:
        #         image_data = file.read()
        #     image = cv2_to_base64(image_data)
        #
        # results = {}
        # for i in range(1):
        ret = self.client.predict(feed_dict={"image": image}, fetch=["res"])
        # print(ret.value)
        return ret

class PaddleCHClient:
    def __init__(self):
        self.client = PipelineClient()
        self.client.connect(['127.0.0.1:18092'])

    def predict(self, img_base64):
        image = utils.read_img_from_base64(img_base64, 'base64')
        # img_dir = "/home/xli/ppocr/imgs/1/"
        # if os.path.exists(img_dir):
        #     print(os.path)
        # else:
        #     print(os.path.sys.prefix)
        # for img_file in os.listdir(img_dir):
        #     with open(os.path.join(img_dir, img_file), 'rb') as file:
        #         image_data = file.read()
        #     image = cv2_to_base64(image_data)
        #
        # results = {}
        # for i in range(1):
        ret = self.client.predict(feed_dict={"image": image}, fetch=["res"])
        # print(ret.value)
        return ret
