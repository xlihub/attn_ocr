from fastapi import FastAPI

from app import utils
from app.utils import debug
from app.handler import OcrEngine, TxOutputParser, \
    SegmentationOutputParser, MultiOutputParser, PaddleOutputParser, PaddleMutiOutputParser, PaddleCHOutputParser
from app.items import InputItem, SegInputItem, PaddleItem
from app.serving_client import PaddleClient, PaddleCHClient
from app.extractor.information_extraction import DataHandle
from app.extractor.direction_filter_generator import get_direction_filter
from app.invoice_template.template import get_examples, tencent_name_transform
from app.extractor.invoice_config import *
import ast
import numpy as np
import shutil
import os

app = FastAPI()


##
# invoice_direction_filter = get_direction_filter(get_examples())


@app.get("/health")
def read_root():
    return {"project": "alive"}


@app.get("/update_temp_file")
def update_temp():
    import app.invoice_template.update_template as u
    u.update_temp()
    return {"status": "OK"}


@app.post("/qrcode")
def decode_qrcode(item: PaddleItem):
    if item.ImageBase64 is not None:
        result = utils.qrcode_dewarp(utils.read_img_from_base64(item.ImageBase64, 'pil'))
    if item.ImageList is not None:
        datetime, strtime, timestamp = utils.current_datetime()
        timestamp = '_' + str(timestamp)
        raw_path = '/home/attnroot/attn_ocr/app/qrcode/rawImages' + timestamp
        imagelist = []
        for base64 in item.ImageList:
            img = utils.read_img_from_base64(base64, 'pil')
            imagelist.append(img)
        result = utils.qrcode_dewarp(imagelist, raw_path)
        if os.path.exists(raw_path):
            shutil.rmtree(raw_path)
    return {"result": result}


@debug
@app.post('/predict')
def predict(item: PaddleItem):
    client = PaddleClient()
    predicts = client.predict(item.ImageBase64)
    # predicts = eval(predicts.value[0])
    # print(predicts.value[0])
    # from google.protobuf.json_format import MessageToJson
    # jsonObj = MessageToJson(predicts)
    # from google.protobuf.json_format import MessageToDict
    # dict_obj = MessageToDict(predicts)
    if len(predicts.value) == 1:
        im_type = predicts.value[0]
        if im_type == 'unknown':
            output_parser = PaddleOutputParser(item, {'result': [], 'im_type': im_type, 'extra': {}})
            response = output_parser.parse_output()
        else:
            res = ast.literal_eval(im_type)
            results = []
            text_dicts = []
            boxes_dicts = []
            score_dicts = []
            mask_dicts = []
            extra_results =[]
            for index, preb_dict in enumerate(res):
                if preb_dict['im_type'] == 'extra':
                    main_pred = results[-1]
                    text_list = ast.literal_eval(preb_dict['text'])
                    extra_text = ''
                    for text in text_list:
                        extra_text += text
                    ext_key = preb_dict['ext_key']
                    main_pred['extra'][ext_key] = extra_text
                    results[-1] = main_pred
                else:
                    preb_list = []
                    preb_ls = ast.literal_eval(preb_dict['preb'])
                    for preb_str in preb_ls:
                        preb = np.frombuffer(preb_str['bytes'], dtype=preb_str['dtype']).reshape(preb_str['shape'])
                        preb_list.append(preb)
                        # preds_idx = preb.argmax(axis=1)
                        # print(preds_idx)
                    # data = np.fromstring(predicts.value[1], np.float32)
                    text_dict = ast.literal_eval(preb_dict['text'])
                    boxes_dict = ast.literal_eval(preb_dict['boxes'])
                    im_type = preb_dict['im_type']
                    score_dict = ast.literal_eval(preb_dict['score'])
                    mask_dict = ast.literal_eval(preb_dict['mask_list'])
                    # dic = pb2dict(predicts)
                    response = {'text': str(text_dict)}
                    print(text_dict)
                    # print(data)
                    direction_filter = get_direction_filter(get_examples())
                    state, predict_result = DataHandle(text_dict, boxes_dict, preb_list, score_dict, im_type, direction_filter,
                                                       True).extract()
                    print(state, predict_result)
                    if state == 'Failed':
                        predicts_dict = {'result': [], 'im_type': im_type, 'extra': {}}
                    else:
                        extra = get_extra()[im_type]
                        if extra is None:
                            extra = {}
                        if len(extra.keys()):
                            extra_dic = {}
                            predicts_dict = {'result': [predict_result], 'im_type': im_type, 'extra': extra_dic}
                        else:
                            predicts_dict = {'result': [predict_result], 'im_type': im_type, 'extra': {}}
                    results.append(predicts_dict)
                    text_dicts.append(text_dict)
                    boxes_dicts.append(boxes_dict)
                    score_dicts.append(score_dict)
                    mask_dicts.append(mask_dict)
            output_parser = PaddleMutiOutputParser(item, results, text_dicts, boxes_dicts, score_dicts, mask_dicts)
            # output_parser = TxOutputParser(item, *predicts)
            response = output_parser.parse_output()
    else:
        preb_dict = ast.literal_eval(predicts.value[1])
        preb_list = []
        for preb_str in preb_dict:
            preb = np.frombuffer(preb_str['bytes'], dtype=preb_str['dtype']).reshape(preb_str['shape'])
            preb_list.append(preb)
            # preds_idx = preb.argmax(axis=1)
            # print(preds_idx)
        # data = np.fromstring(predicts.value[1], np.float32)
        text_dict = ast.literal_eval(predicts.value[0])
        boxes_dict = ast.literal_eval(predicts.value[2])
        im_type = predicts.value[3]
        score_dict = ast.literal_eval(predicts.value[4])
        raw_score = ast.literal_eval(predicts.value[5])
        mask_dict = ast.literal_eval(predicts.value[6])
        score_list = []
        for preb_str in raw_score:
            by = preb_str['bytes']
            dtype = preb_str['dtype']
            preb = np.frombuffer(by, dtype=dtype)
            preb = preb.reshape(preb_str['shape'])
            score_list.append(preb)

        # dic = pb2dict(predicts)
        # response = {'text': str(text_dict)}
        print(text_dict)
        # print(data)
        direction_filter = get_direction_filter(get_examples())
        state, predict_result = DataHandle(text_dict, boxes_dict, preb_list, score_list, im_type, direction_filter,
                                           True).extract()
        print(state, predict_result)
        if state == 'Failed':
            predicts = {'result': [], 'im_type': im_type, 'extra': {}}
        else:
            extra = get_extra()[im_type]
            if extra is None:
                extra = {}
            if len(extra.keys()):
                extra_dic = {}
                predicts = {'result': [predict_result], 'im_type': im_type, 'extra': extra_dic}
            else:
                predicts = {'result': [predict_result], 'im_type': im_type, 'extra': {}}
            # predicts = [predicts_dict]
        output_parser = PaddleOutputParser(item, predicts, text_dict, boxes_dict, score_dict, mask_dict)
        # output_parser = TxOutputParser(item, *predicts)
        response = output_parser.parse_output()
    return response


@debug
@app.post('/pred_raw')
def predict(item: PaddleItem):
    client = PaddleClient()
    predicts = client.predict(item.ImageBase64)
    # predicts = eval(predicts.value[0])
    # print(predicts.value[0])
    # from google.protobuf.json_format import MessageToJson
    # jsonObj = MessageToJson(predicts)
    # from google.protobuf.json_format import MessageToDict
    # dict_obj = MessageToDict(predicts)
    if len(predicts.value) == 1:
        im_type = predicts.value[0]
        if im_type == 'unknown':
            output_parser = PaddleOutputParser(item, {'result': [], 'im_type': im_type, 'extra': {}})
            response = output_parser.parse_output()
        else:
            res = ast.literal_eval(im_type)
            results = []
            for preb_dict in res:
                preb_list = []
                preb_ls = ast.literal_eval(preb_dict['preb'])
                for preb_str in preb_ls:
                    preb = np.frombuffer(preb_str['bytes'], dtype=preb_str['dtype']).reshape(preb_str['shape'])
                    preb_list.append(preb)
                    # preds_idx = preb.argmax(axis=1)
                    # print(preds_idx)
                # data = np.fromstring(predicts.value[1], np.float32)
                text_dict = ast.literal_eval(preb_dict['text'])
                boxes_dict = ast.literal_eval(preb_dict['boxes'])
                im_type = preb_dict['im_type']
                # dic = pb2dict(predicts)
                response = {'text': str(text_dict)}
                print(text_dict)
            output_parser = SegmentationOutputParser(item, text_dict)
            # output_parser = TxOutputParser(item, *predicts)
            response = output_parser.parse_output()
    else:
        preb_dict = ast.literal_eval(predicts.value[1])
        preb_list = []
        for preb_str in preb_dict:
            preb = np.frombuffer(preb_str['bytes'], dtype=preb_str['dtype']).reshape(preb_str['shape'])
            preb_list.append(preb)
            # preds_idx = preb.argmax(axis=1)
            # print(preds_idx)
        # data = np.fromstring(predicts.value[1], np.float32)
        text_dict = ast.literal_eval(predicts.value[0])
        boxes_dict = ast.literal_eval(predicts.value[2])
        im_type = predicts.value[3]
        # dic = pb2dict(predicts)
        response = {'text': str(text_dict)}
        print(text_dict)
        output_parser = SegmentationOutputParser(item, text_dict)
        # output_parser = TxOutputParser(item, *predicts)
        response = output_parser.parse_output()
    return response


@debug
@app.post('/pred')
def predict(item: PaddleItem):
    client = PaddleCHClient()
    predicts = client.predict(item.ImageBase64)
    # predicts = eval(predicts.value[0])
    # print(predicts.value[0])
    # from google.protobuf.json_format import MessageToJson
    # jsonObj = MessageToJson(predicts)
    # from google.protobuf.json_format import MessageToDict
    # dict_obj = MessageToDict(predicts)
    if len(predicts.value) == 1:
        im_type = predicts.value[0]
        if im_type == 'unknown':
            output_parser = PaddleOutputParser(item, [], im_type)
            response = output_parser.parse_output()
        else:
            res = ast.literal_eval(im_type)
            output_parser = PaddleCHOutputParser(item, res)
            # output_parser = TxOutputParser(item, *predicts)
            response = output_parser.parse_output()
    return response


# @debug
# @app.post('/predict')
# def predict(item: InputItem):
#     engine = OcrEngine(east_client, ocr_client, angle_client, item.InvoiceType)
#     predicts = engine.predict(item)
#     output_parser = TxOutputParser(item, *predicts)
#     return output_parser.parse_output(item.InvoiceType)


def pb2dict(obj):
    """
    Takes a ProtoBuf Message obj and convertes it to a dict.
    """
    adict = {}
    if not obj.IsInitialized():
        return None

    for field in obj.DESCRIPTOR.fields:
        if not getattr(obj, field.name):
            continue
        from google.protobuf.descriptor import FieldDescriptor
        if not field.label == FieldDescriptor.LABEL_REPEATED:
            if not field.type == FieldDescriptor.TYPE_MESSAGE:
                adict[field.name] = getattr(obj, field.name)
            else:
                value = pb2dict(getattr(obj, field.name))
                if value:
                    adict[field.name] = value
        else:
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                adict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                adict[field.name] = [v for v in getattr(obj, field.name)]
    return adict
