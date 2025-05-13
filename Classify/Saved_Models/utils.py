import lmdb
import fnmatch
import os
import cv2
import numpy as np
import json
import base64
import requests

class ScreenInfo:
    def __init__(self, dictfilename):
        self.classnames = []
        with open(dictfilename,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                if line not in self.classnames:
                    self.classnames.append(line)
            f.close()
    
    def classname2idx(self, classname):
        for index, key in enumerate(self.classnames):
            if key == classname:
                return index
        return -1
    
    def idx2classname(self, classidx):
        if classidx >= len(self.classnames):
            return ''
        else:
            return self.classnames[classidx]

class CharInfo:
    def __init__(self, dictfilename):
        self.charset = []
        with open(dictfilename,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                infos = line.split('\t')
                if len(infos) != 2:
                    continue
                if infos[0] not in self.charset:
                    self.charset.append(infos[0])
            f.close()
        self.charset = ['BLANK'] + self.charset + ['PAD'] + ['UNKNOWN']
        self.blank_id = 0
        self.pad_id = len(self.charset) - 2
        self.unknown_id = len(self.charset) - 1

    def idx2char(self, index):
        if index >= len(self.charset):
            return ''
        else:
            return self.charset[index]
    
    def char2idx(self, word):
        for index, key in enumerate(self.charset):
            if key == word:
                return index
        return -1
    
    def encode(self, sentence):
        text_idxs = []
        for word in sentence:
            text_idxs.append(self.char2idx(word))
        return text_idxs
    
    def decode(self, textidxs):
        sentences = []
        for textidx in textidxs:
            sentence = ''
            for idx in textidx:
                word = self.charset[idx]
                if word == 'BLANK' or word == 'PAD' or word == 'UNKNOWN':
                    continue
                else:
                    sentence += word
            sentences.append(sentence)
        return sentences
    
    def ctc_decode(self, textindexs, confidences=None):
        results = []
        for idx, indexs in enumerate(textindexs):
            confidence = []
            text = ''
            if confidences is not None:
                confidence = confidences[idx]
            pre_txt_idx = -1
            prob = 0
            txt_num = 0
            for tidx, txtidx in enumerate(indexs):
                if txtidx == self.blank_id or txtidx == self.pad_id or txtidx == self.unknown_id:
                    pre_txt_idx = txtidx
                    continue
                if txtidx == pre_txt_idx:
                    continue
                else:
                    text += str(self.idx2char(txtidx))
                    pre_txt_idx = txtidx
                    if len(confidence) > 0:
                        prob += confidence[tidx]
                    txt_num += 1
            if txt_num > 0:
                prob /= txt_num
            results.append([text, prob])
        return results

def findAllfiles(folder, patterns):
    imgfilenames = []
    for root, subdir, files in os.walk(folder):
        for file in files:
            for pattern in patterns:
                if fnmatch.fnmatch(file, pattern):
                    imgfilenames.append(os.path.join(root, file))
    return imgfilenames


def read_dict(dictfilename):
    contents = []
    with open(dictfilename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            contents.append(line)
        f.close()

def binary2cv(buffer):
    image = cv2.imdecode(np.asarray(bytearray(buffer),dtype=np.uint8),cv2.IMREAD_COLOR)
    return image


def base642image(base64str):
    image = base64.b64decode(base64str)
    image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), 1)
    return image

def image2base64(image):
    base64_str = cv2.imencode('.jpg',image)[1].tobytes()
    base64_str = base64.b64encode(base64_str).decode()
    return base64_str


def encoder_image_preprocess(image, normwidth, normheight):
    mean = np.array([0.48145466, 0.4578275, 0.40821073],dtype=np.float32).reshape(3,1,1)
    std = np.array([0.26862954, 0.26130258, 0.27577711],dtype=np.float32).reshape(3,1,1)
    normimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normimage = cv2.resize(image,(normwidth, normheight))
    normimage = normimage.transpose(2,0,1)
    normimage = normimage.astype(np.float32)
    normimage /= 255
    normimage = (normimage - mean) / std
    return normimage
