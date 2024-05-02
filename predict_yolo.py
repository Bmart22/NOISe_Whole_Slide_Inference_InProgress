#
# Brendan Martin
# predict_yolo.py

import os, sys
#import numpy as np
#from PIL import Image
from ultralytics import YOLO
import csv
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import shuffle

def print_metrics(metrics):

    print("Bounding Boxes")
    print("MaP: ", metrics.box.map)    # map50-95(B)
    print("MaP50: ", metrics.box.map50)  # map50(B)
    print("MaP75: ", metrics.box.map75)  # map75(B)
    print("MaPs: ", metrics.box.maps)   # a list contains map50-95(B) of each category
    print("Segmentation")
    print("MaP: ", metrics.seg.map)    # map50-95(M)
    print("MaP50: ", metrics.seg.map50)  # map50(M)
    print("MaP75: ", metrics.seg.map75)  # map75(M)
    print("MaPs: ", metrics.seg.maps)   # a list contains map50-95(M) of each category
    return
    


def main(argv):
    
    model = YOLO("results/yolo/3_1_large/best.pt")
    
    num_img = 5
    input_img_dir = ""
    
    
#    input_img_dir = "data/oc_test/images/"
    file_list = os.listdir(input_img_dir)
    file_list = [f for f in file_list if f.endswith(".png")]
    file_list = [f for f in file_list if (f.startswith("img_3") or f.startswith("img_9"))]
    
    
    shuffle(file_list)
    file_list = file_list[:num_img]
    
    pred = []
    ground = []
    for file_ind, file in enumerate(file_list):
        file = os.path.join(input_img_dir, file)
        og_img = np.asarray( Image.open(file) ).copy()
        
        # Plot original image
        Image.fromarray(og_img).save(f"samples/{file_ind}_original.png")
        
        
        # Plot results image
        results = model([file])
        for i, r in enumerate(results):
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = im_bgr[..., ::-1]
        pred.append( im_rgb )
        
        Image.fromarray(im_rgb).save(f"samples/{file_ind}_prediction.png")
        
        #load labels
        file = file.replace("images","labels")
        file = file.replace(".png",".txt")
        
        bbf = csv.reader(open(file, newline=''), delimiter=' ')
        
        for i, row in enumerate(bbf):
            #reshape and rescale coordinates
            points = [float(row[i]) for i in range(1, len(row))]
            points = np.array(points).reshape((-1,2))
            points = (points * og_img.shape[:2]).astype(int)
            
            #fill contour in original image
            cv2.fillPoly(og_img, pts=np.int32([points]), color=(255,0,0))
            
        ground.append( og_img )
        Image.fromarray(og_img).save(f"samples/{file_ind}_ground_truth.png")
    
    
    return

    

if __name__ == '__main__':
    main(sys.argv)
