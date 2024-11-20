import cv2 
import json
import glob
import tqdm
import numpy as np

json_file = ('./box_refine.json')
save_file = ('./inference/box_score.json')

with open(json_file) as f:
    box_info = json.load(f)
    for i, img in tqdm.tqdm(enumerate(box_info.keys())):
        rgb_img = cv2.imread('./data/20210204_Digi_generated/test_images/'+img)
        boxes = np.array(box_info[img]['boxes'])
        scores = np.array(box_info[img]['scores'])
        
        for j, box in enumerate(boxes):
            x1,y1,x2,y2 = box
            cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            cv2.putText(rgb_img, 'score:' +str(j), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        cv2.imwrite('./sort_out/%05d.jpg' % i, rgb_img)