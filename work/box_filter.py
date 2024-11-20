import json
import numpy as np
import tqdm
import time
import cv2


from sort import *

# json_file = ('./inference_0.5/box_score.json')
json_file = ('./box_refine_nms.json')
save_file = ('./box_info_consecutive_all.json')


# SORT tracker
mot_tracker = Sort(max_age=50, 
                min_hits=2,
                iou_threshold=0.001) #create instance of the SORT tracker


record = {}
tmp = []
with open(json_file) as f:
    box_info = json.load(f)
    for i, img in tqdm.tqdm(enumerate(box_info.keys())):
        # print("*"*10)
        # print('Next image start')
        rgb_img = cv2.imread('./data/20210204_Digi_generated/test_images/'+img)
        try:
            dets = np.array(box_info[img]['box_scores'])
        except:
            boxes = np.array(box_info[img]['boxes'])
            scores = np.array(box_info[img]['scores']).reshape(-1, 1)
            try:
                dets = np.concatenate((boxes, scores), axis=1)
            except:
                dets = np.zeros((0,5))
        if dets.size ==0:
            dets = np.zeros((0,5))
        trackers = mot_tracker.update(dets)
        for d in trackers:
            x1,y1,x2,y2 = d[:4]
            
            unique_id = str(int(d[4]))
            if unique_id not in record:
                patch = rgb_img[int(y1): int(y2), int(x1): int(x2)]
                record[unique_id] = {'boxes': [int(x1), int(y1), int(x2), int(y2)], 'frameID': '%05d.jpg' % i}
                cv2.imwrite('./patches/%02d.jpg' % int(d[4]), patch)
            
            cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            cv2.putText(rgb_img, 'id:' +str(int(d[4])), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        cv2.imwrite('./sort_out/%05d.jpg' % i, rgb_img)
        
with open(save_file, 'w') as f2:
    json.dump(record, f2)

# import pdb; pdb.set_trace()
# record = {}
# with open(json_file) as f:
#     box_info = json.load(f)
#     for i, img in tqdm.tqdm(enumerate(box_info.keys())):
#         print("*"*10)
#         print('Next image start')
#         record[img] = {}
#         record[img]['boxes'] = [] 
#         record[img]['box_center'] = []
#         boxes = box_info[img]['Boxes']
#         box_center = box_info[img]['Box_center']
#         if i == 113:
#             import pdb; pdb.set_trace()
#         for index, center in enumerate(box_center):
#             Notfound = False
#             for j in range(1, 6):
#                 frame_id = "%05d" % (i+j) + '.jpg'
#                 if frame_id not in box_info:
#                     continue 
#                 tmp_center = np.array(box_info[frame_id]['Box_center'])
#                 if tmp_center.size ==0:
#                     # print('no box in this frame')
#                     continue
#                 center_array = np.array(center)
#                 diff = abs(np.sum(tmp_center - center_array, axis=1))
#                 if np.where(diff < 100)[0].size == 0:
#                     # no sequnential box within 10 pixel range 
#                     Notfound = True
#             if not Notfound:
#                 # this box is well identified within next 10 frames
#                 record[img]['boxes'].append(boxes[index]) 
#                 record[img]['box_center'].append(center) 
#         print('detected {} box in {}'.format(len(record[img]['boxes']), img))
        
    
#     with open(save_file, 'w') as f2:
#         json.dump(record, f2)
#     import pdb; pdb.set_trace()
