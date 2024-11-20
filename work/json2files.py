import pathlib
import json

with open('box_gt.json') as infile:
    gt_boxes = json.load(infile)

# with open('./inference/box_score.json') as infile:
with open('box_refine_nms.json') as infile:
    pred_boxes = json.load(infile)

gt_path = './trash2/gt/'
dets_path = './trash2/dets/'

pathlib.Path(gt_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(dets_path).mkdir(parents=True, exist_ok=True)

# gt generator
for key, value in gt_boxes.items():
    name = key.split('.jpg')[0] + '.txt'
    with open(gt_path + name, 'w') as f:
        for v in value:
            f.write('trash' + ' ')
            f.write(str(v[0]) + ' ')
            f.write(str(v[1]) + ' ')
            f.write(str(v[2] - v[0]) + ' ')
            f.write(str(v[3] - v[1]))

            # for j, vv in enumerate(v):
            #     f.write(str(vv))
            #     if j != 3:
            #         f.write(' ')
            f.write('\n')

# dets generator
for key, values in pred_boxes.items():
    name = key.split('.jpg')[0] + '.txt'
    with open(dets_path + name, 'w') as f:
        boxes = values['boxes']
        for i in range(len(values['scores'])):
            f.write('trash' + ' ' + str(round(values['scores'][i], 2)) + ' ')
            v = values['boxes'][i]
            f.write(str(int(v[0])) + ' ')
            f.write(str(int(v[1])) + ' ')
            # f.write(str(int(v[2])) + ' ')
            # f.write(str(int(v[3])))
            f.write(str(int(v[2] - v[0])) + ' ')
            f.write(str(int(v[3] - v[1])))
            # for j, vv in enumerate(values['boxes'][i]):
            #     f.write(str(int(vv)))
                
            #     if j != 3:
            #         f.write(' ')
            f.write('\n')