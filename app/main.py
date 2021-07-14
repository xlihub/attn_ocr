from fastapi import FastAPI
from app.utils import debug
from app.handler import OcrEngine, east_client, ocr_client, angle_client, maskrcnn_client, TxOutputParser, \
    SegmentationOutputParser, MultiOutputParser, PaddleOutputParser
from app.items import InputItem, SegInputItem, PaddleItem
from app.serving_client import PaddleClient
from app.extractor.information_extraction import DataHandle
from app.extractor.direction_filter_generator import get_direction_filter
from app.invoice_template.template import examples, tencent_name_transform
import ast
import numpy as np

app = FastAPI()

##
invoice_direction_filter = get_direction_filter(examples)


@app.get("/health")
def read_root():
    return {"project": "alive"}


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
    # print(data)
    state, predict_result = DataHandle(text_dict, boxes_dict, preb_list, im_type, invoice_direction_filter,
                                       True).extract()
    print(state, predict_result)
    predicts = [predict_result]
    output_parser = PaddleOutputParser(item, predicts, im_type)
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


@debug
@app.post('/segmentation')
def segmentation(item: SegInputItem):
    engine = OcrEngine(east_client, ocr_client, angle_client, maskrcnn=maskrcnn_client)
    predicts = engine.segmentation_predict(item)
    output_parser = SegmentationOutputParser(item, predicts[2])

    return output_parser.parse_output()


@debug
@app.post('/multi_invoices_predict')
def multi_invoices_predict(item: SegInputItem):
    engine = OcrEngine(east_client, ocr_client, angle_client, maskrcnn=maskrcnn_client)

    predicts = engine.multi_invoices_predict(item)
    output_parser = MultiOutputParser(item, *predicts)

    # print("nicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenice")

    return output_parser.parse_output()


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
