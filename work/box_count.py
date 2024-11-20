import pathlib
import json

# with open('box_gt.json') as infile:
#     gt_boxes = json.load(infile)

with open('./inference_0.7/box_score.json') as infile:
# with open('box_refine.json') as infile:
    pred_boxes = json.load(infile)

hits = 0
# import pdb; pdb.set_trace()
for key, values in pred_boxes.items():
    # tmp = [i for i in values['scores'] if i>=0.5]
    # if len(tmp) == 0:
    if len(values['boxes']) == 0:
        hits += 1

print("{} are not tracked, percentage is {}".format(hits, hits/len(pred_boxes)))