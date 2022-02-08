import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
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


class BaseClient:
    def __init__(self, portal, name, timeout=100):
        self.name = name
        self.timeout = timeout
        self.options = [('grpc.max_send_message_length', 50 * 1024 * 1024),
                        ('grpc.max_receive_message_length', 50 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(portal, options=self.options)

    def _from_requests(self, data):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = 'model'
        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data))
        return request

    def predict(self, data):
        stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        request = self._from_requests(data)
        response = stub.Predict(request, self.timeout)
        results = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results[key] = nd_array
        return results['pred']


class MaskRcnnClient(BaseClient):
    def _from_requests(self, data_list):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = 'model'
        request.inputs['input0'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data_list[0]))
        request.inputs['input1'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data_list[1]))
        request.inputs['input2'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data_list[2]))
        return request

    def predict(self, data):
        stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        request = self._from_requests(data)
        response = stub.Predict(request, self.timeout)
        results = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results[key] = nd_array
        return results['output0'], results['output3']


class AsyncClient(BaseClient):
    def predict(self, data, callback=None, idx=None):
        stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        request = self._from_requests(data)
        response = stub.Predict.future(request, self.timeout)
        response.add_done_callback(self._create_rpc_callback(callback, idx))

    def _create_rpc_callback(self, callback, idx):
        results = {}

        def _callback(result_future):
            try:
                response = result_future.result()
                for key in response.outputs:
                    tensor_proto = response.outputs[key]
                    nd_array = tf.contrib.util.make_ndarray(tensor_proto)
                    results[key] = nd_array
                callback(results['pred'], idx)
            except Exception as e:
                logger.error('AsyncClient callback err: {}'.format(e))

        return _callback


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
