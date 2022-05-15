
##
# Code to Visualize annotations
# 1. from jsons (bbbox)
# 2. from labels(yolo format)


import os
import sys
import cv2
import json
from collections import defaultdict
from matplotlib import pyplot as plt


# code to parse json
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

# code to overlay annotations from json on image and show
def visualize_json(path, data, class_map, show=True):
    #cv2.namedWindow('image', 0)
    colors = [(0, 255, 0), (0, 0, 255)]
    class_dist = defaultdict(int)
    min_max_wh = [99999, 0, 99999, 0]
    for ids in data:
        image_path = os.path.join(path, 'image_%09d.jpg' % (ids+1))
        if not os.path.exists(image_path):
            image_path = os.path.join(path[:-5]+'eval', 'image_%09d.jpg' % (ids+1))
        image = cv2.imread(image_path)
        for d in data[ids]:
            b, c = d
            if c == 1 and b[2] < min_max_wh[0]:
                min_max_wh[0] = b[2]
            if c == 1 and b[2] >= min_max_wh[1]:
                min_max_wh[1] = b[2]
            if c == 2 and b[3] < min_max_wh[2]:
                min_max_wh[2] = b[3]
            if c == 2 and b[3] >= min_max_wh[3]:
                min_max_wh[3] = b[3]
            class_dist[class_map[c-1]] += 1
            if show:
                cv2.rectangle(image, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), colors[c-1], 2)
        if show:
            cv2.imwrite('ground_truth/image_%09d.jpg' % (ids+1), image)
            cv2.imshow('image', image)
            cv2.waitKey(0)
    print ('min_max_wh:', min_max_wh)
    print ('N-Samples:', len(data))
    names = list(class_dist.keys())
    values = list(class_dist.values())
    plt.bar(range(len(class_dist)), values, tick_label=names)
    plt.show()


# code to overlay annotations from labels on image and show
def visualize_yolo(path, class_map, show=True):
    #cv2.namedWindow('image', 0)
    class_dist = defaultdict(int)
    min_max_wh = [99999, 0, 99999, 0]
    for files in os.listdir(path):
        if '.jpg' not in files:
            continue
        imgpath = os.path.join(path, files)
        txtpath = os.path.join(path.replace('images', 'labels'), files[:-4]+'.txt')
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (640, 640))
        H, W = image.shape[:2]
        with open(txtpath) as fp:
            data = []
            for lines in fp.readlines():
                vals = list(map(float, lines.strip().split()))
                c = int(vals[0])
                class_dist[class_map[c]] += 1
                w = int(vals[3]*W)
                h = int(vals[4]*H)
                x = int((vals[1]*W)-(w/2))
                y = int((vals[2]*H)-(h/2))
                if c == 0 and h < min_max_wh[0]:
                    min_max_wh[0] = h
                if c == 0 and h >= min_max_wh[1]:
                    min_max_wh[1] = h
                if c == 1 and w < min_max_wh[2]:
                    min_max_wh[2] = w
                if c == 1 and w >= min_max_wh[3]:
                    min_max_wh[3] = w
                if not show:
                    continue
                if c == 0.0:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if show:
                cv2.imshow('image', image)
                cv2.waitKey(0)
    print ('min_max_wh:', min_max_wh)
    names = list(class_dist.keys())
    values = list(class_dist.values())
    plt.bar(range(len(class_dist)), values, tick_label=names)
    plt.show()

def main():
    ann_path = sys.argv[1]
    show = False
    if len(sys.argv) > 1:
        show = True
    if 'annotations' in ann_path:
        data = parse_json(os.path.join(ann_path))
        visualize_json(os.path.dirname(ann_path).replace('annotations', 'images/train'), data,
                            {0: 'person', 1: 'car'}, show)
    else:
        visualize_yolo(ann_path, {0: 'person', 1: 'car'}, show)

if __name__=='__main__':
    main()


##
# run: python scripts/visualize.py trainval/annotations/bbox-annotations.json
#      python scripts/visualize.py trainval/images/train/
#
# @author: Danish
# @date: 2022/05/15
