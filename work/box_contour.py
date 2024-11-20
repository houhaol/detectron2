import numpy as np
import cv2
import glob
import tqdm
import json

json_file = ('./inference_images/box_info.json')

mask_path = sorted(glob.glob("/home/houhao/workspace/videoReconstr/webRemoval/data/20210119_digiLabFinal/masks/00365.png"))
gt_json = ('./box_gt.json')
record = {}
for mask_file in tqdm.tqdm(mask_path):
    mask = cv2.imread(mask_file, 0)
    mask_rgb = cv2.imread(mask_file, 1)
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    box = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        box.append([x,y,x+w, y+h])
        mask_rgb = cv2.rectangle(mask_rgb,(x,y),(x+w,y+h),(255,255,0),2)
    record[(mask_file.split('/')[-1]).split('.p')[0]+'.jpg'] =box 

    cv2.imwrite('tmp.png', mask_rgb)

# with open(gt_json, 'w') as f:
#     json.dump(record, f)
    # box_info = json.load(f)


# import pdb; pdb.set_trace()
