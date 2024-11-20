import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets.coco import convert_to_coco_json
train_data = load_coco_json("./data/20210204_Digi_train.json", "./data/20210204_Digi_generated/images/", "train")
val_data = load_coco_json("./data/20210204_Digi_val.json", "./data/20210204_Digi_generated/valid_images/", "val")

# convert_to_coco_json(dataset_name=train_data, output_file="./data/20210204_Digi_coco_train.json", allow_cached=True)
# convert_to_coco_json(dataset_name=val_data, output_file="./data/20210204_Digi_coco_val.json", allow_cached=True)

DatasetCatalog.register("train", lambda: load_coco_json("./data/20210204_Digi_train.json", "./data/20210204_Digi_generated/images/", "train"))
MetadataCatalog.get("train").set(thing_classes=["trash"],
                                                    json_file="./data/20210204_Digi_train.json",
                                                    image_root="./data/20210204_Digi_generated/images/")
DatasetCatalog.register("val", lambda: load_coco_json("./data/20210204_Digi_val.json", "./data/20210204_Digi_generated/valid_images/", "val"))
MetadataCatalog.get("val").set(thing_classes=["trash"],
                                                json_file="./data/20210204_Digi_val.json",
                                                image_root="./data/20210204_Digi_generated/valid_images/")





# # training
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("./detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
cfg.SOLVER.MAX_ITER = 6000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = (2000,4000)
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.0001

cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


from detectron2.evaluation import COCOEvaluator
from loss_hook import *
import logging
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()