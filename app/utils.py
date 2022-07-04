import numpy as np
import requests as req
from io import BytesIO
import base64
from PIL import Image
import cv2
import math
import sys
import re
import itertools
import pyzbar.pyzbar as pyzbar
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array
from keras import backend as K
import skimage
import skimage.io
from skimage.measure import regionprops
from skimage.morphology import label
import time
import random
import os
import imutils

from app.key_dicts import ALPHABET
import subprocess
import os.path as osp

alphabet = ALPHABET + u'卍'


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def activate_box(cond):
    label_img = label(cond)
    props = regionprops(label_img)
    for i in props:
        cord = i['BoundingBox']
        mask = i.convex_image
        x_gap = cord[3] - cord[1]
        y_gap = cord[2] - cord[0]
        if ((x_gap > 2) or (y_gap > 2)) and y_gap / x_gap < 4:
            box = [(cord[0], cord[1]), (cord[2], cord[1]), (cord[2], cord[3]), (cord[0], cord[3])]
            box_m = [max(math.ceil((cord[1] - 1) * 4), 0),
                     max(math.ceil((cord[0] - 1) * 4), 0),
                     max(math.ceil((cord[3] + 1) * 4), 0),
                     max(math.ceil((cord[2] + 1) * 4), 0)]
            yield box, box_m, mask


def handle_img(im):
    img = im.convert('RGB')
    scale = img.size[1] * 1.0 / 32
    w = int(img.size[0] / scale)
    img = img.resize((w, 32), Image.BILINEAR)
    img_array = np.asarray(img) / 255
    return np.expand_dims(np.transpose(img_array, (1, 0, 2)).astype('float32'), 0)


def mask_croped_img(croped_im, mask, crop_box):
    bb_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    mask = np.transpose(mask)
    resized_mask = skimage.transform.resize(mask, bb_size) * 255
    resized_mask = Image.fromarray(np.transpose(np.uint8(resized_mask)))
    im_size = croped_im.size
    flat_im = np.reshape(np.array(croped_im), (im_size[0] * im_size[1], 3))
    H, edges = np.histogramdd(flat_im, bins=(8, 8, 8))
    color_bin_idx = np.unravel_index(H.argmax(), H.shape)
    color = [int((edges[i][color_bin_idx[i] + 1] + edges[i][color_bin_idx[i]]) / 2) for i in range(3)]
    background = Image.new("RGB", bb_size, tuple(color))
    return Image.composite(croped_im, background, resized_mask)


def text_to_labels(text):
    ret = []
    for char in text:
        find_idx = alphabet.find(char)
        if find_idx != -1:
            ret.append(find_idx)
    return ret


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def labels_to_text_(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("卍")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def decode_original(out):
    ret = []
    out_best = list(np.argmax(out, 1))
    out_str = labels_to_text_(out_best)
    ret.append(out_str)
    return ret


def decode_mask(out, mask=np.array([])):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:] * mask, 1)) if mask.any() else list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        out_str = labels_to_text(out_best)
        ret.append(out_str)
    return ret


def read_img_from_url(url, read_type="pil"):
    response = req.get(url)
    if read_type == "pil":
        return Image.open(BytesIO(response.content))
    elif read_type == "skimage":
        return skimage.io.imread(BytesIO(response.content))


def read_img_from_base64(base64_code, read_type="pil"):
    base64_code = base64_code.replace('<img src="', '')
    base64_code = base64_code.replace('" alt=""/>', '')
    base64_code = base64_code.replace("<img src='", '')
    base64_code = base64_code.replace("' alt=''/>", '')
    text = re.findall(r'data:image/.*;base64,', base64_code)
    if text:
        base64_code = base64_code.replace(text[0], '')

    if read_type == "pil":
        return Image.open(BytesIO(base64.b64decode(base64_code)))
    elif read_type == "skimage":
        return skimage.io.imread(BytesIO(base64.b64decode(base64_code)))
    elif read_type == "base64":
        return base64_code


def resize_image(im, max_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def remove_punctuation(line):
    rule = re.compile(u'[^0-9]')
    line = rule.sub('', line)
    return line


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def center_transform(affine, input_shape, img_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
    top, left, bottom, right = 0, 0, hi, wi
    ''' 对比例特殊的图像进行裁剪
    if wi/hi/anisotropy < wo/ho: # input image too narrow, extend width
        w     = hi*wo/ho*anisotropy
        left  = (wi-w)/2
        right = left + w
    else: # input image too wide, extend height
        h      = wi*ho/wo/anisotropy
        top    = (hi-h)/2
        bottom = top + h
    '''
    center_matrix = np.array([[1, 0, -ho / 2], [0, 1, -wo / 2], [0, 0, 1]])
    scale_matrix = np.array([[(bottom - top) / ho, 0, 0], [0, (right - left) / wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi / 2], [0, 1, wi / 2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))


def transform_img(x, affine, img_shape):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)


def read_for_rotation(img, img_shape=(224, 224, 3)):
    x = img_to_array(img.convert('RGB'))
    data = np.zeros((1,) + img_shape, dtype=K.floatx())
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = center_transform(t, x.shape, img_shape)
    x = transform_img(x, t, img_shape)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    data[0] = x
    return data


def _cal_center_point(points):
    x = [points[i][0] for i in range(4)]
    y = [points[i][1] for i in range(4)]
    center = [int(sum(x) / 4), int(sum(y) / 4)]
    left_up = [point for point in points if (point[0] < center[0]) and (point[1] < center[1])]
    left_down = [point for point in points if (point[0] < center[0]) and (point[1] > center[1])]
    right_down = [point for point in points if (point[0] > center[0]) and (point[1] > center[1])]
    right_up = [point for point in points if (point[0] > center[0]) and (point[1] < center[1])]
    return center, [left_up[0], left_down[0], right_down[0], right_up[0]]


def _cal_edge_length(points):
    distanse = [int(math.sqrt((points[i][0] - points[i + 1][0]) ** 2 + (points[i][1] - points[i + 1][1]) ** 2)) for i in
                range(3)]
    return min(distanse)


def prepare_images(img):
    is_list = False
    if isinstance(img, list):
        results_list = []
        for index, pil_img in enumerate(img):
            raw_path = '/home/attnroot/attn_ocr/app/qrcode/rawImages'
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
            img_path = '/home/attnroot/attn_ocr/app/qrcode/rawImages/raw_' + str(index) + '.jpg'
            pil_img.save(img_path)
            qrcode_result = predict_bar_image([img_path])
            results = []
            if qrcode_result:
                if len(qrcode_result) > 0:
                    for result in qrcode_result:
                        data = result['data']
                        type = result['type']
                        if type is not 'QRCODE':
                            results.append({type: data})
            results_list.append(results)
        is_list = True
        return is_list, results_list
    else:
        img_path = '/home/attnroot/attn_ocr/app/qrcode/raw.jpg'
        img.save(img_path)
        qrcode_result = predict_bar_image([img_path])
        results = []
        if qrcode_result:
            if len(qrcode_result) > 0:
                for result in qrcode_result:
                    data = result['data']
                    type = result['type']
                    if type is not 'QRCODE':
                        results.append({type: data})
        return is_list, results


def qrcode_dewarp(pil_img):
    is_list, results = prepare_images(pil_img)
    # 将训练日志写入out.log与err.log文件
    mode = 'w'
    dst_path = '/home/attnroot/attn_ocr/app/qrcode/'
    outlog = open(osp.join(dst_path, 'out.log'), mode=mode, encoding='utf-8')
    errlog = open(osp.join(dst_path, 'err.log'), mode=mode, encoding='utf-8')
    outlog.write("This log file path is {}\n".format(
        osp.join(dst_path, 'out.log')))
    outlog.write("注意：标志为WARNING/INFO类的仅为警告或提示类信息，非错误信息\n")
    errlog.write("This log file path is {}\n".format(
        osp.join(dst_path, 'err.log')))
    errlog.write("注意：标志为WARNING/INFO类的仅为警告或提示类信息，非错误信息\n")

    if is_list:
        train = subprocess.Popen(
            args='/home/attnroot/anaconda3/envs/invoiceocr/bin/python /home/attnroot/attn_ocr/app/qrcode/infer_qr.py --image_dir=/home/attnroot/attn_ocr/app/qrcode/rawImages',
            shell=True, stdout=outlog, stderr=errlog, universal_newlines=True,
            encoding='utf-8')
    else:
        train = subprocess.Popen(
            args='/home/attnroot/anaconda3/envs/invoiceocr/bin/python /home/attnroot/attn_ocr/app/qrcode/infer_qr.py --image_file=/home/attnroot/attn_ocr/app/qrcode/raw.jpg',
            shell=True, stdout=outlog, stderr=errlog, universal_newlines=True,
            encoding='utf-8')
    while True:
        flag = 1
        if train.poll() is None:
            flag = 0
        if flag:
            break
    if is_list:
        for index, result in enumerate(results):
            txt_path = '/home/attnroot/attn_ocr/app/qrcode/rawImages/raw_' + str(index) + '_result.txt'
            result = get_qrcode_results(txt_path, result)
            results[index] = result
    else:
        txt_path = "/home/attnroot/attn_ocr/app/qrcode/raw_result.txt"
        results = get_qrcode_results(txt_path, results)
    return results


def get_qrcode_results(txt_path, results):
    f = open(txt_path, "r")
    raw_result = f.readlines()
    if len(raw_result) > 0:
        index = 1
        for line in raw_result:
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            type = 'QRCODE' + str(index)
            results.append({type: line})
            index += 1
        f.close()
    # os.remove('/home/attnroot/attn_ocr/app/qrcode/raw.jpg')
    # os.remove("/home/attnroot/attn_ocr/app/qrcode/raw_result.txt")
    return results


def predict_bar_image(image_list, batch_size=1):
    # 读取图片并将其转化为灰度图片
    batch_loop_cnt = math.ceil(float(len(image_list)) / batch_size)
    for i in range(batch_loop_cnt):
        result_list = []
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(image_list))
        batch_image_list = image_list[start_index:end_index]
        image = cv2.imread(batch_image_list[0])
        image1 = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算图像中x和y方向的Scharr梯度幅值表示
        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

        # x方向的梯度减去y方向的梯度
        gradient = cv2.subtract(gradX, gradY)
        # 获取处理后的绝对值
        gradient = cv2.convertScaleAbs(gradient)
        # cv2.imwrite("gradient.png", gradient)

        # 对处理后的结果进行模糊操作
        blurred = cv2.blur(gradient, (9, 9))
        # 将其转化为二值图片
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("thresh.png", thresh)

        # 构建一个掩码并将其应用在二值图片中
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite("closed1.png", closed)

        # 执行多次膨胀和腐蚀操作
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        # cv2.imwrite("closed2.png", closed)

        # 在二值图像中寻找轮廓, 然后根据他们的区域大小对该轮廓进行排序，保留最大的一个轮廓
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        areas = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(areas):
            for area in areas:
                # 计算最大的轮廓的最小外接矩形
                rect = cv2.minAreaRect(area)
                box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
                box = np.int0(box)

                # 找出四个顶点的x，y坐标的最大最小值。新图像的高=maxY-minY，宽=maxX-minX。
                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                xmin = min(Xs)
                xmax = max(Xs)
                ymin = min(Ys)
                ymax = max(Ys)
                hight = ymax - ymin
                width = xmax - xmin
                # 找出条形码长方形轮廓
                if abs(width - hight) < max(width, hight) / 3:
                    continue
                else:
                    break
            xmin = 0 if int(xmin) - 20 < 0 else int(xmin) - 20
            ymin = 0 if int(ymin) - 20 < 0 else int(ymin) - 20
            xmax = int(xmax) + 20
            ymax = int(ymax) + 20
            roi = image1[ymin:ymax, xmin: xmax]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("crop_raw.jpg", roi)
            # cv2.imwrite("crop_gray.jpg", gray)
            barcodes = pyzbar.decode(gray)
            if len(barcodes) == 0:
                gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                # cv2.imwrite("crop_resize.jpg", gray)
                barcodes = pyzbar.decode(gray)
                if len(barcodes) == 0:
                    result_list = []
                    continue
            # 这里循环，因为画面中可能有多个二维码
            for barcode in barcodes:
                # 条形码数据为字节对象，所以如果我们想在输出图像上画出来，就需要先将它转换成字符串
                barcodeData = barcode.data.decode("UTF-8")
                barcodeType = barcode.type
                result_dic = {'type': barcodeType, 'data': barcodeData}
                result_list.append(result_dic)
            # 向终端打印条形码数据和条形码类型
            # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        # img_name = os.path.basename(batch_image_list[0]).split('.')[0]
        # img_path = os.path.dirname(batch_image_list[0])
        # file_path = img_path + '/' + img_name + '_bar_result.txt'
        # out = open(file_path, 'w')
        # for result in result_list:
        #     type = result['type']
        #     data = result['data']
        #     out.write("[INFO] Found {} barcode: {}".format(type, data) + '\n')
        # out.close()
    return result_list


def debug(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            score = fn(*args, **kwargs)
            print("{} cost {}".format(fn.__name__, time.time() - start))
            return score
        except Exception as e:
            print("{} except {}".format(fn.__name__, repr(e)))

    return wrapper
