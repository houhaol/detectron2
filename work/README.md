# Steps

###### 1. 
Put data under data dir. Data structure looks like: \
```
.
│   _train.json
│   _val.json    
└───Dataset_name
│   └───train
│   │   │   00000.jpg
│   │   │   ...
│   │
│   └───val
│       │   00000.jpg
│       │   ...
```
###### 2. train
Modify dataset path in train.py, necessary training parameters, then start training
`python train.py`

###### 3. (Optional) Merge validation loss in tensorboard
`python plot.py`

###### 4. Inference on test dataset
e.g.
`python demo.py --config-file ../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml --input ./data/20210204_Digi_generated/test_images/*.jpg --output ./inference/ --confidence-threshold 0.7 --model ./output/model_final.pth`



