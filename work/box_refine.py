import json
import numpy as np
import tqdm
import time
import cv2
import copy

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

json_file = ('./inference/box_score.json')
save_file = ('./box_refine_test.json')
# json_file = ('./clasic_detect.json')
# save_file = ('./clasic_refine.json')


def merge_box(bb1, bb2):
    centorid1_x, centorid1_y  = bb1[0] + (bb1[2] - bb1[0])/2, bb1[1] + (bb1[3] - bb1[1])/2
    centorid2_x, centorid2_y  = bb2[0] + (bb2[2] - bb2[0])/2, bb2[1] + (bb2[3] - bb2[1])/2
    diff_x = abs(centorid1_x - centorid2_x)
    diff_y = abs(centorid1_y - centorid2_y)
    print(diff_x, diff_y)
    if (diff_x + diff_y) < 300:
#         print("merge these two bounding box")
        min_x = min(bb1[0], bb2[0])
        min_y = min(bb1[1], bb2[1])
        max_x = max(bb1[2], bb2[2])
        max_y = max(bb1[3], bb2[3])
        return (min_x, min_y, max_x, max_y)
    else:
        return None

# def merge_box_xywh(bb1, bb2):
#     centorid1_x, centorid1_y  = bb1[0] + (bb1[2])/2, bb1[1] + (bb1[3])/2
#     centorid2_x, centorid2_y  = bb2[0] + (bb2[2])/2, bb2[1] + (bb2[3])/2
#     diff_x = abs(centorid1_x - centorid2_x)
#     diff_y = abs(centorid1_y - centorid2_y)
#     # print(diff_x, diff_y)
#     if (diff_x + diff_y) < 300:
# #         print("merge these two bounding box")
#         min_x = min(bb1[0], bb2[0])
#         min_y = min(bb1[1], bb2[1])
#         max_x = max(bb1[2]+bb1[0], bb2[2]+bb2[0])
#         max_y = max(bb1[3]+bb1[1], bb2[3]+bb2[1])
#         return (min_x, min_y, max_x, max_y)
#     else:
#         return None

with open(json_file) as f:
    box_info = json.load(f)
    tmp_box = copy.deepcopy(box_info)
    for i, img in tqdm.tqdm(enumerate(box_info.keys())):
        print("*"*10)
        print('Next image start')
        boxes = np.array(box_info[img]['boxes'])
        scores = np.array(box_info[img]['scores'])
        index_tmp = []
        for j, box in enumerate(boxes):
            for k in range(j+1, boxes.shape[0]):
                box2 = boxes[k]
                result = merge_box(box, box2)
                # import pdb; pdb.set_trace()
                if result is not None:
                    index_tmp.append(j)
                    index_tmp.append(k)
                    # try:
                    tmp_box[img]['boxes'].append(list(result))
                    tmp_box[img]['scores'].append(max(scores[j], scores[k]))
                    # except:
                    #     # import pdb; pdb.set_trace()
                    # rgb_img = cv2.imread("/home/houhao/workspace/videoReconstr/webRemoval/data/20210119_digiLabFinal/frames/00000.jpg")
                    # # for d in tmp_box[img]['boxes']:
                    # for d in box_info[img]['boxes']:
                    #     x1,y1,x2,y2 = d
                    #     cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                    # cv2.imwrite('tmp.jpg', rgb_img)
                    # import pdb; pdb.set_trace()
        
        for index in sorted(list(set(index_tmp)), reverse=True):
            print('delet box {} at img {}'.format(index, img))
            del tmp_box[img]['boxes'][index]
            del tmp_box[img]['scores'][index]
        

# import pdb; pdb.set_trace()

with open(save_file, 'w') as f2:
    json.dump(tmp_box, f2, cls=NpEncoder)
