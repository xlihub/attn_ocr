from fastapi import FastAPI

from app import utils
from app.utils import debug
from app.handler import OcrEngine, TxOutputParser, \
    SegmentationOutputParser, MultiOutputParser, PaddleOutputParser, PaddleMutiOutputParser, PaddleCHOutputParser
from app.items import InputItem, SegInputItem, PaddleItem
from app.serving_client import PaddleClient, PaddleCHClient
from app.extractor.information_extraction import DataHandle, StringMatcher
from app.extractor.direction_filter_generator import get_direction_filter
from app.invoice_template.template import get_examples, tencent_name_transform
from app.extractor.invoice_config import *
import ast
import numpy as np
import shutil
import os
import requests
import httpx
import copy

app = FastAPI()

Template_Type = ['invoice_A4', 'invoice_A5']


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
            extra_results = []
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
                    score_list = []
                    score_ls = ast.literal_eval(preb_dict['raw_score'])
                    for score_str in score_ls:
                        score = np.frombuffer(score_str['bytes'], dtype=score_str['dtype']).reshape(score_str['shape'])
                        score_list.append(score)
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
                    ocr_handle = DataHandle(
                        text_dict, boxes_dict, preb_list, score_list, im_type,
                        direction_filter,
                        True)
                    state, predict_result, new_text_list, new_boxes_list, new_score_list, text_boxes_list = ocr_handle.extract()
                    print(state, predict_result)
                    if state == 'Failed':
                        predicts_dict = {'result': [], 'im_type': im_type, 'extra': {}}
                    else:
                        # 获取自定义模板数据
                        template = get_template_info(im_type, predict_result)
                        # print('template')
                        # print(template)
                        if template:
                            S_UNINO = predict_result['S_UNINO']
                            f_result = find_result_from_template(predict_result, new_text_list, new_boxes_list,
                                                                 new_score_list,
                                                                 mask_dict, template, ocr_handle, text_boxes_list)
                            ocr_handle.check_symbol = []
                            for res in f_result:
                                label = res['label']
                                if not label.startswith("__"):
                                    if res['final_text'] is not '':
                                        # print(res['final_text'])
                                        if ocr_handle.output_handle is not None:
                                            handle_text = ocr_handle.output_handle_(res['final_text'],
                                                                                    ocr_handle.output_handle.get(
                                                                                        res['label'], []),
                                                                                    res['label'],
                                                                                    res['final_score'])
                                            # print(handle_text)
                                            predict_result[label] = handle_text
                                        else:
                                            predict_result[label] = res['final_text']
                            # 校验方法，当所有金额栏位第一位字符都低于阈值时，将第一位截掉
                            if len(ocr_handle.check_symbol):
                                check_data = list(filter(lambda c: c['check'], ocr_handle.check_symbol))
                                if len(check_data) == len(ocr_handle.check_symbol):
                                    for item in ocr_handle.check_symbol:
                                        text = item['text']
                                        predict_result[item['field']] = text[1:]
                                        # print(text[1:])
                            # print(f_result)
                            predict_result['S_UNINO'] = S_UNINO
                        extra = get_extra()[im_type]
                        if extra is None:
                            extra = {}
                        if len(extra.keys()):
                            extra_dic = {}
                            predicts_dict = {'result': [predict_result], 'im_type': im_type, 'extra': extra_dic}
                        else:
                            predicts_dict = {'result': [predict_result], 'im_type': im_type, 'extra': {}}
                    results.append(predicts_dict)
                    text_dicts.append(new_text_list)
                    boxes_dicts.append(new_boxes_list)
                    score_dicts.append(new_score_list)
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
        ocr_handle = DataHandle(text_dict, boxes_dict, preb_list, score_list, im_type, direction_filter,
                                True)
        state, predict_result, new_text_list, new_boxes_list, new_score_list, text_boxes_list = ocr_handle.extract()
        print(state, predict_result)
        if state == 'Failed':
            predicts = {'result': [], 'im_type': im_type, 'extra': {}}
        else:
            # 获取自定义模板数据
            template = get_template_info(im_type, predict_result)
            # print('template')
            # print(template)
            if template:
                S_UNINO = predict_result['S_UNINO']
                f_result = find_result_from_template(predict_result, new_text_list, new_boxes_list, new_score_list,
                                                     mask_dict, template, ocr_handle, text_boxes_list)
                ocr_handle.check_symbol = []
                for res in f_result:
                    label = res['label']
                    if not label.startswith("__"):
                        if res['final_text'] is not '':
                            # print(res['final_text'])
                            if ocr_handle.output_handle is not None:
                                handle_text = ocr_handle.output_handle_(res['final_text'],
                                                                        ocr_handle.output_handle.get(res['label'], []),
                                                                        res['label'],
                                                                        res['final_score'])
                                # print(handle_text)
                                predict_result[label] = handle_text
                            else:
                                predict_result[label] = res['final_text']
                # 校验方法，当所有金额栏位第一位字符都低于阈值时，将第一位截掉
                if len(ocr_handle.check_symbol):
                    check_data = list(filter(lambda c: c['check'], ocr_handle.check_symbol))
                    if len(check_data) == len(ocr_handle.check_symbol):
                        for item in ocr_handle.check_symbol:
                            text = item['text']
                            predict_result[item['field']] = text[1:]
                            # print(text[1:])
                # print(f_result)
                predict_result['S_UNINO'] = S_UNINO
            extra = get_extra()[im_type]
            if extra is None:
                extra = {}
            if len(extra.keys()):
                extra_dic = {}
                predicts = {'result': [predict_result], 'im_type': im_type, 'extra': extra_dic}
            else:
                predicts = {'result': [predict_result], 'im_type': im_type, 'extra': {}}
            # predicts = [predicts_dict]
            # output_parser = PaddleOutputParser(item, predicts, text_dict, boxes_dict, score_dict, mask_dict)
        output_parser = PaddleOutputParser(item, predicts, new_text_list, new_boxes_list, new_score_list, mask_dict)
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
def find_result_from_template(result, text_list, boxes_list, score_list, mask_dict, template, ocr_handle,
                              text_boxes_list):
    # 根据mask中x,y的值，将template中的所有point进行变换n_x = t_x - x,n_y = t_y - y
    template_list = prepare_template(mask_dict, template)
    # 处理result中的字段，每个字段形成key text box score的结构
    result_list = prepare_result(result, text_list, boxes_list, score_list, template_list, ocr_handle, text_boxes_list)
    # 遍历所有template中的字段，将point的box与result中的相同字段的box比较distance
    # final_result = check_result_form_template(result_list, template_list)
    # distance误差小，保持原值，distance误差大，在boxes_list里找出离box最近的box,采用它的值
    return result_list


def prepare_template(mask, template):
    box = mask['box']
    mask_x = box[0]
    mask_y = box[1]
    tem_list = template['TemplateData']['shapes']
    anchor_list = []
    for label in tem_list:
        if label['label'].startswith("__"):
            anchor_list.append(label)
        label_points = label['points']
        label_box = label_points[0].copy()
        label_box.extend(label_points[2])
        new_x = label_box[0] - mask_x
        new_y = label_box[1] - mask_y
        new_xx = label_box[2] - mask_x
        new_yy = label_box[3] - mask_y
        label_new_box = [new_x, new_y, new_xx, new_yy]
        label['label_box'] = label_box
        label['label_new_box'] = label_new_box
    for label in tem_list:
        # 获取栏位到瞄点的相对距离
        if not label['label'].startswith("__"):
            for anchor in anchor_list:
                key = anchor['label']
                label[key] = get_distance(label['label_box'], anchor['label_box'])
    return tem_list


def prepare_result(result, text_list, boxes_list, score_list, template_list, ocr_handle, text_boxes_list):
    result_list = []
    data_field = ocr_handle.data
    anchors = {anchor: ocr_handle.current_score[anchor] for anchor in ocr_handle.current_score.keys() if
               anchor.startswith("__")}
    for key in result.keys():
        result_dic = {}
        text = result[key]
        result_dic['label'] = key
        result_dic['text'] = text
        result_dic['field'] = data_field[key]
        if not key.startswith("__"):
            siamese_threshold = 0.6
            siamese_text_list = {i: textbox for i, textbox in enumerate(text_boxes_list) if
                                 data_field[key].siamese_ratio(textbox) > siamese_threshold}
            result_dic['siamese_box'] = siamese_text_list
        for index, original_text in enumerate(text_list):
            m2 = StringMatcher(text, original_text)
            r = m2.ratio()
            if r > 0.9:
                result_dic['original_text'] = original_text
                result_dic['box'] = boxes_list[index]
                result_dic['score'] = score_list[index]
                break
        result_dic['final_text'] = check_result_form_template(result_dic, template_list, text_list, boxes_list,
                                                              score_list, anchors)
        result_list.append(result_dic)
    return result_list


def check_result_form_template(result_dic, template_list, text_list, boxes_list, score_list, anchors):
    label = result_dic['label']
    if label.startswith("$"):
        label = label[1:]
    if label.startswith("__"):
        label = label[2:]
    # print(label)
    final_text = ''
    for template_dic in template_list:
        m2 = StringMatcher(label, template_dic['label'])
        r = m2.ratio()
        if r > 0.9:
            template_box = template_dic['label_new_box']
            text_boxes = result_dic['siamese_box']
            new_distance = {i: get_distance(template_box, list(text_boxes[i].box)) for i in text_boxes.keys()}
            nearest = min(new_distance, key=new_distance.get)
            dict_sorted = sorted(new_distance.items(), key=lambda i: i[1], reverse=False) if len(
                new_distance) > 1 else list(new_distance.items())
            best_box = text_boxes[nearest]
            if label in ['AMTN_NET', 'TAX', 'AMTN']:
                temp_diff = 0
                check = True
                diff_list = []
                temp_diff_list = []
                for i, dis in dict_sorted:
                    temp_box = text_boxes[i]
                    diff = 0
                    for item in template_dic.keys():
                        if item.startswith("__"):
                            # print(label + item)
                            if item in anchors.keys():
                                anchor = anchors[item]
                                anchor_box = anchor.box
                                result_dic[item] = get_distance(list(temp_box.box), list(anchor_box))
                                # print('____________')
                                # print(temp_box.text)
                                # print('template_distance:' + str(template_dic[item]))
                                # print('result_distance:' + str(result_dic[item]))
                                # print('_distance_:' + str(result_dic[item] - template_dic[item]))
                                # print('%_distance_%:' + str((result_dic[item] - template_dic[item]) / template_dic[item]))
                                # print('____________')
                                diff = abs(result_dic[item] - template_dic[item]) / template_dic[item]
                                if diff > 0.05:
                                    check = False
                                    diff_list.append(diff)
                    if check:
                        break
                    else:
                        if temp_diff is 0:
                            temp_diff = diff
                            temp_diff_list = diff_list
                            continue
                        else:
                            if diff < temp_diff:
                                if abs(diff - temp_diff) < 0.1:
                                    if len(diff_list) < len(temp_diff_list):
                                        best_box = temp_box
                                else:
                                    best_box = temp_box
                                break
                            else:
                                continue
            # print(best_box.text)
            final_text = best_box.text
            result_dic['final_box'] = boxes_list[nearest]
            result_dic['final_score'] = score_list[nearest]
            return final_text
    return final_text


def get_distance(box1, box2):
    center1_x, center1_y = (box1[0] + box1[2]) / 2, (
            box1[1] + box1[3]) / 2
    center2_x, center2_y = (box2[0] + box2[2]) / 2, (
            box2[1] + box2[3]) / 2
    return (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2


def get_template_info(im_type, result):
    if im_type in Template_Type:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        url = 'http://0.0.0.0:3008/api/templateInfo'
        query = {
            'type': im_type,
            'no': result['S_UNINO'],
            'name': ''
        }
        client = httpx.Client(verify=False)
        try:
            response = client.get(url, params=query, headers=headers)
            response.raise_for_status()
            template = {}
            if response.status_code is 200:
                res = response.json()
                if res['success']:
                    template = res['template']
            return template
        except httpx.HTTPError as exc:
            print(f"HTTP Exception for {exc.request.url} - {exc}")
            return False
        finally:
            client.close()
    else:
        return False


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
