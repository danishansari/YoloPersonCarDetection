
##
# Code to up/down scale image and annotations
# If the anontations(bdbox) are big as compared image resolution
# then the images are down-sampled otherwise upsampled
#


import os
import sys
import cv2
import random
from collections import defaultdict


# function to find the index of the majority bouding in bins[0, 1, 2, 3, 4, 5, 6, 7]
# index will decide up/down scale of images
def get_bbox_bin(bboxes, nbins, h_range=(3, 639)):
    bin_val = (h_range[1]-h_range[0])/nbins
    bins = [i*bin_val for i in range(nbins)]
    bins.append(640)
    box_bin_count = defaultdict(int)
    for i, box in enumerate(bboxes):
        c, b = box
        if c != 0:
            continue
        for j, bn in enumerate(bins):
            if b[3] < bn:
                box_bin_count[j] += 1
                break
    return max(zip(box_bin_count.values(), box_bin_count.keys()))[1]

# function to up/down scale images based on scale-factor(sf) 
# if sf is -ve, images are down scaled otherwise up scaled
# with a random pixel value chosen from the bin it belongs
def scale_image_bbox(fname, image, bboxes, nbins, sf):
    H, W = image.shape[:2]
    bin_sz = H//nbins
    scale_pixel = int(random.uniform(bin_sz*(sf-1), bin_sz*sf))
    if W+scale_pixel > 0 and H+scale_pixel > 0:
        image_scaled = cv2.resize(image, (W+scale_pixel, H+scale_pixel))
        h, w = image_scaled.shape[:2]
        cv2.imwrite('augmented/aug_%d_%s' % (abs(scale_pixel), fname), image_scaled)
        with open('augmented/aug_%d_%s.txt' % (abs(scale_pixel), fname[:-4]), 'w') as fp:
            for box in bboxes:
                c, b = box
                nx = (w/W)*b[0] 
                ny = (h/H)*b[1]
                nw = (w/W)*b[2]
                nh = (h/H)*b[3]
                cx = nx + (nw/2)
                cy = ny + (nh/2)
                fp.write('%d %f %f %f %f\n' % (c, cx/w, cy/h, nw/w, nh/h))
                #cv2.rectangle(image_scaled, (int(nx),int(ny)), (int(nx+nw), int(ny+nh)), (0, 255, 0), 2)
            #cv2.imshow('scale', image_scaled)
            #cv2.waitKey(0)

# function to decide bin and augment image
def augment(fname, image, bboxes, nbins=8):
    bin_no = get_bbox_bin(bboxes, nbins)
    for i in range(1, nbins+1):
        if i == bin_no:
            continue
        if i < bin_no:
            scale_image_bbox(fname, image, bboxes, nbins, i*-1)
        else:
            scale_image_bbox(fname, image, bboxes, nbins, i-bin_no)
    

# funtion to iterate over all images and augment (scale)
def main():
    image_size = (640, 640)
    for files in os.listdir(sys.argv[1]):
        if '.jpg' not in files:
            continue
        imgpath = os.path.join(sys.argv[1], files)
        txtpath = os.path.join(sys.argv[1].replace('images', 'labels'), files[:-4]+'.txt')
        image = cv2.imread(imgpath)
        image = cv2.resize(image, image_size)
        H, W = image.shape[:2]
        bdboxes = []
        with open(txtpath) as fp:
            for lines in fp.readlines():
                vals = list(map(float, lines.strip().split()))
                c = int(vals[0])
                w = vals[3]*W
                h = vals[4]*W
                x = vals[1]*W - (w/2)
                y = vals[2]*H - (h/2)
                bdboxes.append([c, list(map(int, [x, y, w, h]))])
        augment(files, image, bdboxes)

if __name__=='__main__':
    main()

##
# run: create a directory 'augmented'
#      python scripts/scale_augment.py trainval/images/train/
#   
# @author: Danish
# @date: 2022/05/15
 
