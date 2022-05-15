
##
# Code to parse anotation and convert bounding box to yolo format for training
#

import os
import sys
import cv2
import json

# Functiont o parse json(annotations) file and extract
# per-image annotations
def parse_json(json_path):
    with open(json_path) as fp:
        data = json.load(fp)
    data_proc = {}
    for d in data['annotations']:
        if d['image_id'] in data_proc:
            data_proc[d['image_id']].append([d['bbox'], d['category_id']])
        else:
            data_proc[d['image_id']] = [[d['bbox'], d['category_id']]]
    return data_proc

# function to  convert bdboxes to yolo format
def convert(path, data):
    for ids in data:
        image_path = os.path.join(path, 'image_%09d.jpg' % (ids+1))
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        with open(image_path[:-4]+'.txt', 'w') as fp:
            for d in data[ids]:
                b, c = d
                cx = b[0]+(b[2]/2)
                cy = b[1]+(b[3]/2)
                fp.write('%d %f %f %f %f\n' % (c-1, cx/w, cy/h, b[2]/w, b[3]/h))

def main():
    ann_path = sys.argv[1]
    data = parse_json(os.path.join(ann_path))
    convert(os.path.dirname(ann_path).replace('annotations', 'images/train'), data)

if __name__=='__main__':
    main()

## 
# run: scripts/convert2yolo.py <path-to-json-file>
# 
# @author: Danish
# @date: 2022/05/15
