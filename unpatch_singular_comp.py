#
# Brendan Martin
# unpatch.py

import os, sys
#import numpy as np
#from PIL import Image
from ultralytics import YOLO
import csv
from PIL import Image, ImageDraw
import numpy as np
#import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import shuffle, randint
import torch
import torchvision
import time
import argparse
import math
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint

# um/pixel length
UM_PER_PIXEL = 0.7784
UM_PER_PATCH = UM_PER_PIXEL * 832
#0.5945 µm2/pixel
MAX_DET = 30000
Image.MAX_IMAGE_PIXELS = 100000000

DEVICE = torch.device('cuda:0')


def scale_boxes(boxes, num_images, img_ind, img_scale):
    boxes[:,(0,2)] = boxes[:,(0,2)] + (img_scale[0]/2)*img_ind[0] # x
    boxes[:,(1,3)] = boxes[:,(1,3)] + (img_scale[1]/2)*img_ind[1] # y
    return boxes
    
def scale_masks(masks, num_images, img_ind, img_scale):
    for m in range(len(masks)):
        masks[m] = masks[m] + (img_scale/2)*img_ind
    return masks

# Credit: Ultralytics YOLOv8 model (https://github.com/ultralytics/ultralytics)
def box_inter_union(boxes1, boxes2):
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
    
def local_nms(box_results, mask_results):
    if box_results[1][1].numel() == 0:
        return torch.tensor([], device=DEVICE), []
    
    keep_bool_master = []
    tmp_boxes = []

    tmp_boxes.append(box_results[0][0])
    tmp_boxes.append(box_results[0][1])
    tmp_boxes.append(box_results[0][2])
    tmp_boxes.append(box_results[1][0])

    tmp_boxes.append(box_results[1][2])
    tmp_boxes.append(box_results[2][0])
    tmp_boxes.append(box_results[2][1])
    tmp_boxes.append(box_results[2][2])
    tmp_boxes_torch = torch.cat(tmp_boxes)
    
    # If no neighboring predictions, skip
    if torch.numel(tmp_boxes_torch)==0:
        return torch.tensor([], device=DEVICE), []
    
    for box in box_results[1][1]:
        
        intersection = box_inter_union(box[:4].unsqueeze(0), tmp_boxes_torch[:,:4])[0]
        surf_area = torchvision.ops.box_area( box[:4].unsqueeze(0) )[0]
        comp = ((surf_area - intersection)/surf_area) < 0.1

        if torch.any( comp ):
            area = torchvision.ops.box_area( tmp_boxes_torch[comp[0],:4] )
            
            # If boxes overlap significantly, keep larger box
            if torch.all(surf_area > area):
                keep_bool_master.append( True )
            else:
                keep_bool_master.append( False )
        else:
            keep_bool_master.append( True )
        
        
    if not any(keep_bool_master):
        return torch.tensor([], device=DEVICE), []
    
    box_results = box_results[1][1][keep_bool_master]
    mask_results = [ mask_results[1][1][i] for i in range(len(keep_bool_master)) if keep_bool_master[i] ]
    return box_results, mask_results
    
    
def inference(model, idx, img, size, out_dir):
    empty_tensor = torch.tensor([], device=DEVICE)
    
    # divide size of image by size of patch/2
    num_patches = ( np.array(img.size) / (size/2) ).astype(int)
    
    # Run inference on each image
    box_results = []
    mask_results = []
    
    box_results.append( [empty_tensor for _ in range(0, img.size[1], size//2)] )
    box_results[-1] += [empty_tensor, empty_tensor]
    mask_results.append( [[] for _ in range(0, img.size[1], size//2)] )
    mask_results[-1] += [[], []]
    
    for y0 in range(0, img.size[1], size//2):
        box_results.append([ empty_tensor ])
        mask_results.append([ [] ])
        for x0 in range(0, img.size[0], size//2):
            x1, y1 = x0+size, y0+size
            
            # crop image
            img_crop = img.crop((x0,y0,x1,y1))
            yc=math.ceil(y0/(size//2))
            xc=math.ceil(x0/(size//2))
            results = model( img_crop, verbose=False, device=0 )
            img_ind = np.array((xc,yc))
            
            # Scale the predictions back to their proper size
            for r in range(len(results)):
                boxes = results[r].boxes.data.clone()
                
                if boxes.numel() != 0: # if osteoclasts detected
                    boxes = scale_boxes(boxes, num_patches, img_ind, (size,size))
                    masks = scale_masks(results[r].masks.xy, num_patches, img_ind, np.array((size,size)))
                    box_results[-1].append( boxes )
                    mask_results[-1].append( masks )
                else:
                    box_results[-1].append( empty_tensor )
                    mask_results[-1].append( [] )
                    
        box_results[-1].append( empty_tensor )
        mask_results[-1].append( [] )
    
    box_results.append( [empty_tensor for _ in range(0, img.size[1], size//2)] )
    box_results[-1] += [empty_tensor, empty_tensor]
    mask_results.append( [[] for _ in range(0, img.size[1], size//2)] )
    mask_results[-1] += [[], []]
    
    objects_found = True if box_results else False
    
    if objects_found:
        new_box_results = []
        new_mask_results = []
        for r in range(1,len(box_results)-1):
            for c in range(1,len(box_results[r])-1):
                
                # If empty
                if torch.numel(box_results[r][c])==0:
                    continue
                    
                output = local_nms([ b[c-1:c+2] for b in box_results[r-1:r+2] ], [ m[c-1:c+2] for m in mask_results[r-1:r+2] ])
                
                new_box_results.append(output[0])
                new_mask_results += output[1]
                
        box_results = torch.cat(new_box_results)
        mask_results = new_mask_results
    
    with open("{f}/img_comp_{id}.txt".format(f=out_dir, id=idx), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(box_results)):
            writer.writerow( box_results[i].tolist() + mask_results[i].flatten().tolist() )
            
    # Draw boxes on original image
    img1 = ImageDraw.Draw(img, 'RGBA')
    
    for i, box in enumerate(box_results):
        box = box[:4].type(torch.int)
        shape = [(box[0], box[1]), (box[2], box[3])]
        img1.rectangle(shape, outline="green", width=3)
        mask = mask_results[i].astype(int).flatten().tolist()
        if len(mask) >= 6:
            color = (randint(0,255),randint(0,255),randint(0,255))
            img1.polygon(mask, fill=color+(125,), outline="blue")
            
    img.save( "{f}/unpatch_comp_{id}.png".format(f=out_dir, id=idx) )
    
    if objects_found:
        return [{"boxes":box_results[:,:4], "scores":box_results[:,4], "labels":box_results[:,5].int()}]
    else:
        return [{"boxes":[], "scores":[], "labels":[]}]

def load_annot(path):
    # Load the ground truth annotations
    bbf = csv.reader(open(path, newline=''), delimiter=',')
    bbs = []
    for row in bbf:
        if row[1] == "bb_x":
            continue
        
        bbs.append( [int(row[1]), int(row[2]), int(row[1])+int(row[3]), int(row[2])+int(row[4])] ) #xyxy
#        bbs.append( [ int(float(coor)) for coor in row[5:] ] )
    bbs = torch.tensor(bbs)
    bbs = [{ "boxes":bbs, "labels":torch.zeros(bbs.shape[0], dtype=torch.int) }]
    return bbs

def main(argv):
    
    # move the sys args into parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_foldername", type=str, default="img")
    parser.add_argument("--out_foldername", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="out")
    parser.add_argument("--ratio", type=float, default=0.7784) #um per pixel
    args = parser.parse_args()
    
    
    um_per_pixel = args.ratio
    patch_size = int( UM_PER_PATCH/um_per_pixel )
    
    out_dir = args.out_foldername
    
    img_dir = args.img_foldername
    img_files = os.listdir(img_dir)
#    img_files = [img for img in img_files if img.endswith(".tif")]

    model_path = args.model_path
    model = YOLO(model_path)
    
    # check if out_dir exists and create if it doesn't
    if not os.path.exists( out_dir ):
        os.makedirs( out_dir )
        
    idx = 0
    for img_filename in img_files[idx:]:
    
        print(img_filename.split("_")[1][:3], end=',')
        
        img = Image.open( os.path.join(img_dir, img_filename) )
        
        pred = inference(model, idx, img, patch_size, out_dir)
        
        idx += 1
        
    
    return
    
    

if __name__ == '__main__':
    main(sys.argv)
