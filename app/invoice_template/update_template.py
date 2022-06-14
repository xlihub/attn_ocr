# Python program to read
# json file


import json
import subprocess
import time
from glob import glob
import os
from app.initdb import engine, User, Template, Pattern
from sqlalchemy.orm import sessionmaker

siamese_decode = {
    "S": "ℤ",
    "N": "ℂ",
    "A": "ℇ",
    "a": "℈",
    "C": "ℌ",
    "M": "℣"
}


def replace_siamese_decode(label_list):
    res = []
    for label in label_list:
        for key in siamese_decode.keys():
            if key in label:
                label = label.replace(key, siamese_decode[key])
        res.append(label)
    return res


def initjsondata(jsondata):
    shapes = jsondata['shapes']
    del jsondata['imageData']
    pattern = {}
    for index in range(len(shapes)):
        label = shapes[index]['label']
        if '__' in label:
            label_list = label.split(',')
            label = label_list[0]
            shapes[index]['label'] = label
            for i in range(len(label_list)):
                label_list[i] = label_list[i].replace('__', '')
            pattern[label] = label_list
        elif '=' in label:
            label_list = label.split('=')
            label = label_list[0]
            shapes[index]['label'] = label
            label_list.pop(0)
            label_list = replace_siamese_decode(label_list)
            pattern[label] = label_list
    extra = {}
    if 'extra' in jsondata.keys():
        extra = jsondata['extra']
    special_handle = {}
    if 'special' in jsondata.keys():
        special = jsondata['special']
        for index in range(len(special)):
            label = special[index]['label']
            values = special[index]['values']
            special_handle[label] = values
    output_handle = {}
    if 'output' in jsondata.keys():
        output = jsondata['output']
        for index in range(len(output)):
            label = output[index]['label']
            values = output[index]['values']
            output_handle[label] = values
    return jsondata, pattern, extra, special_handle, output_handle


def update_temp():
    # engine是2.2中创建的连接
    Session = sessionmaker(bind=engine)

    # 创建Session类实例
    session = Session()

    src_dir_list = glob('/home/attnroot/ftp/upload/new_template/' + '*')
    examples = {}
    for src_file in src_dir_list:
        # Opening JSON file
        f = open(src_file, )

        # returns JSON object as
        # a dictionary
        fpath, fname = os.path.split(src_file)
        name = fname.split('.')
        data = json.load(f)
        temp_name = name[0]
        newdata = {name[0]: data}
        our_template = session.query(Template).filter_by(name=temp_name).first()

        if not our_template:
            temp, pattern, extra, special_handle, output_handle = initjsondata(data)

            # 创建Template类实例
            ed_template = Template(name=temp_name, config=temp)
            ed_pattern = Pattern(name=temp_name, config=pattern, extra=extra, special_handle=special_handle, output_handle=output_handle)
            # 将该实例插入到users表
            session.add(ed_template)
            session.add(ed_pattern)
            session.commit()
            # Closing file
            f.close()
        else:
            temp, pattern, extra, special_handle, output_handle = initjsondata(data)
            our_template.config = temp
            our_pattern = session.query(Pattern).filter_by(name=temp_name).first()
            our_pattern.config = pattern
            our_pattern.extra = extra
            our_pattern.special_handle = special_handle
            our_pattern.output_handle = output_handle
            session.commit()
            f.close()
        # subprocess.call(args='./update_temp.sh', shell=True, cwd='/home/xli/OCR/invoice_ocr/app/invoice_template/')
        # from app.invoice_template.template import get_examples
        # from app.handler import invoice_direction_filter, get_direction_filter
        # invoice_direction_filter = get_direction_filter(get_examples())
        # from app.main import invoice_direction_filter
        # print(invoice_direction_filter)
        # invoice_direction_filter = get_direction_filter(get_examples())
        # print(invoice_direction_filter)

# patterns = session.query(Pattern).all()
# data1 = {
#      "shapes": []
#  }
# data = {
#     "__代码": ["代码", "代码:"],
#     "__号码": ["号码", "号码:"],
#     "__日期": ["日期", "日期:"],
#     "__时间": ["时间", "时间:"],
#     "__金额": ["金额", "金额:", "￥", "价格", "价格:", "票价", "票价:"],
#     "ticket_code": ["ℂ{12}"],
#     "ticket_id": ["ℂ{8}"],
#     "date": ["ℂℂℂℂℂℂℂℂ", "ℂℂℂℂ/ℂℂ/ℂℂ", "ℂℂℂℂ年ℂℂ月ℂℂ日", "ℂℂℂℂ-ℂℂ-ℂℂ"],
#     "time": ["ℂℂ:ℂℂ", "ℂℂ:ℂℂ:ℂℂ"],
#     "money": ["ℂ{1,5}", "ℂ{1,5}.ℂℂ", "ℂ{1,5}元", "ℂ{1,5}.ℂℂ元", "℣{2,15}"]
# }
# # 创建Template类实例
# train_ticket_tw_template = Pattern(name='common', config=data)
# common_template = Template(name='common', config=data1)
# # 将该实例插入到users表
# session.add(train_ticket_tw_template)
# session.add(common_template)
# session.commit()
#
# # 一次插入多条记录形式
# session.add_all(
#     [User(name='wendy', fullname='Wendy Williams', password='foobar'),
#      User(name='mary', fullname='Mary Contrary', password='xxg527'),
#      User(name='fred', fullname='Fred Flinstone', password='blah')]
# )
#
# 当前更改只是在session中，需要使用commit确认更改才会写入数据库

