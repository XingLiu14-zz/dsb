# -*- coding: utf-8 -*-
import sys
import os
from os.path import join, expanduser

# ADD pytorch_mask_rcnn to sys path
sys.path.append('./pytorch-mask-rcnn')

import pandas as pd
import numpy as np
import imageio
import skimage
import torch
import model as modellib

from dsb_config import DsbConfig, InferenceConfig
from dsb_dataset import DsbDataset


#ROOT_DIR = os.getcwd()
#HOME_DIR = expanduser('~')
DSB_DATA_DIR = '../'
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
COCO_MODEL_PATH = os.path.join('./logs/', "mask_rcnn_dsb_0050.pth")

test_dir = join(DSB_DATA_DIR, 'stage2_test')
test_ids = os.listdir(test_dir)

dataset_test = DsbDataset()
dataset_test.load_dataset(test_ids, test_dir)
dataset_test.prepare()

dsb_config = DsbConfig()

model = modellib.MaskRCNN(config=dsb_config, model_dir='./logs')
model = model.cuda()
model.load_state_dict(torch.load(COCO_MODEL_PATH), strict=False)

raw_predictions = []
for test_id in dataset_test.image_ids:
    if test_id%1000 == 0:
        print("test id",test_id)
    if test_id == 1196:
        raw_predictions.append(None)
        continue
    test_image1 = dataset_test.load_image(test_id, 0)
    pred = model.detect([test_image1])
    if pred is None:
    	raw_predictions.append(None)
    	continue
    #print("pred1========",pred)
    pred = pred[0]
    #print("preds1=================",pred)
    sc = pred['scores']
    #print("scores=================",sc)
    pred = pred['masks']
    #print("pred2==================",pred)
    raw_predictions.append((pred, sc))

assert len(test_ids) == len(raw_predictions)

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def numpy2encoding_no_overlap(predicts, img_name, scores):
    sum_predicts = np.sum(predicts, axis=2)
    rows, cols = np.where(sum_predicts>=2)

    for i in zip(rows, cols):
        instance_indicies = np.where(np.any(predicts[i[0],i[1],:]))[0]
        highest = instance_indicies[0]
        predicts[i[0],i[1],:] = predicts[i[0],i[1],:]*0
        predicts[i[0],i[1],highest] = 1

    ImageId = []
    EncodedPixels = []
   
    #print(predicts.shape)
    for i in range(predicts.shape[2]):
        rle = rle_encoding(predicts[:,:,i])
        if len(rle)>0:
            ImageId.append(img_name)
            EncodedPixels.append(rle)
    return ImageId, EncodedPixels

new_test_ids = []
rles = []
for id, raw_pred in zip(test_ids, raw_predictions):
    if raw_pred is None:
        new_test_ids += [id]
        rles += [(1,0)]
        continue
    ids, rle = numpy2encoding_no_overlap(raw_pred[0], id, raw_pred[1])
    new_test_ids += ids
    rles += rle
print("new_test_ids shape", len(new_test_ids))
print("rles shape",len(rles))
print("rles shape 1", len(rles[0]))
print("len new_test_id",len(new_test_ids))
print("len rles",len(rles))
assert len(new_test_ids) == len(rles)

rles = [ " ".join( str(v) for v in k ) for k in rles ]
df = pd.DataFrame({ 'ImageId' : new_test_ids , 'EncodedPixels' : rles})
df.to_csv('./stage2_sub_all50.csv', index=False, columns=['ImageId', 'EncodedPixels'])
