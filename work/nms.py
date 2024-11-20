import numpy as np
import json
import tqdm
import cv2

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, -1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# json_file = ('./inference_0.5/box_score.json')
json_file = ('./box_refine.json')
save_file = ('./box_refine_nms.json')

nms_result = {}
with open(json_file) as f:
    box_info = json.load(f)
    for i, img in tqdm.tqdm(enumerate(box_info.keys())):
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
        
        nms_res = nms(dets, thresh=0.5)
        for res in nms_res:
            x1,y1,x2,y2,score = dets[res]
            cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            # cv2.putText(rgb_img, 'score:' +str(scores[i]), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        nms_result[img] = {'boxes': boxes[nms_res].tolist(), 'scores': scores[nms_res].flatten().tolist()}
        cv2.imwrite('./sort_out/%05d.jpg' % i, rgb_img)

with open(save_file, 'w') as f2:
    json.dump(nms_result, f2)