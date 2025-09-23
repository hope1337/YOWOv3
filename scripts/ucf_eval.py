import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob
import json

from math import sqrt

from cus_datasets.build_dataset import build_dataset
from cus_datasets.collate_fn import collate_fn
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from utils.box import non_max_suppression, box_iou
from evaluator.eval import compute_ap
from cus_datasets.ucf.transforms import UCF_transform
from utils.flops import get_info
from collections import defaultdict
import psutil
from utils.utils import load_dict_json, save_dict_json
from boxmot import BoostTrack
from pathlib import Path
from PIL import Image
import warnings
from tqdm import tqdm
import logging

logging.getLogger("boxmot.motion.cmc.ecc").setLevel(logging.ERROR)

@torch.no_grad()
def eval(config):

    ###############################################
    dataset = build_dataset(config, phase='testvmAP')
    
    dataloader = data.DataLoader(dataset, 64, False, collate_fn=collate_fn
                                 , num_workers=8, pin_memory=True)
    
    model = build_yowov3(config)
    get_info(config, model)
    model.to("cuda")
    model.eval()
    ###############################################

    # Configure
    #iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    iou_v = torch.tensor([0.5]).cuda()
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    
    def nested_dict():
        return {"boxes": [], "img": None}

    # defaultdict cho video_name → frame_id → default dict
    # {video_name: {frame_id: {boxes: [[x1, y1, x2, y2, conf, cls], []... ], img: img_path}}} absolute coordinate
    ann_dict = defaultdict(lambda: defaultdict(nested_dict))
    pred_dict = defaultdict(lambda: defaultdict(nested_dict))

    with torch.no_grad():
        height = config['img_size']
        width  = config['img_size']
        scale_img = torch.tensor((width, height, width, height))
        for batch_clip, batch_bboxes, batch_labels, batch_meta_data in p_bar:
            batch_clip = batch_clip.to("cuda")
            targets = []
            for i, (bboxes, labels, meta_data) in enumerate(zip(batch_bboxes, batch_labels, batch_meta_data)):
                target = torch.Tensor(bboxes.shape[0], 6)
                target[:, 0] = i
                target[:, 1] = labels
                target[:, 2:] = bboxes
                targets.append(target)

                if config['calc_vmAP'] is True:
                    video_name = meta_data[0]
                    frame_id = meta_data[1] 
                    ann_dict[video_name][frame_id]["img"] = meta_data[2]
                    pseudo_conf = torch.ones(bboxes.shape[0]).unsqueeze(1)
                    ann_dict[video_name][frame_id]["boxes"] = torch.cat((bboxes * scale_img, pseudo_conf, labels.unsqueeze(1)), dim=1).tolist()

            targets = torch.cat(targets, dim=0).to("cuda")

            # Inference
            outputs = model(batch_clip)

            # NMS
            targets[:, 2:] *= scale_img.cuda()  # to pixels
            outputs = non_max_suppression(outputs, 0.005, 0.5)
            
            if config['calc_vmAP'] is True:
                for i, output in enumerate(outputs):
                    video_name = batch_meta_data[i][0]
                    frame_id = batch_meta_data[i][1]
                    pred_dict[video_name][frame_id]['boxes'] = output.cpu().tolist()
                    pred_dict[video_name][frame_id]['img'] = batch_meta_data[i][2]

            # Metrics
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                    continue

                detections = output.clone()
                #util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                # Evaluate
                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()  # target boxes
                    #tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                    #tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                    #tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                    #tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                    #util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                    correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                    correct = correct.astype(bool)

                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    for j in range(len(iou_v)):
                        x = torch.where((iou >= iou_v[j]) & correct_class)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                            matches = matches.cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        # Compute metrics
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

        # Print results
        print('%10.3g' * 3 % (m_pre, m_rec, mean_ap), flush=True)

        # Return results
        model.float()  # for training
        #return map50, mean_ap
        print(map50, flush=True)
        print(flush=True)
        print("=================================================================", flush=True)
        print(flush=True)
        print(mean_ap, flush=True)
        
        #############################################################################################################
        if config['calc_vmAP'] is False:
            return
        
        print()
        print("Calculating video-mAP...")
        print()

        save_dict_json(ann_dict, os.path.join(config['temp_folder'], "annotations.json"))
        print("annotation information saved!")
        ann_dict.clear()
        
        save_dict_json(pred_dict, os.path.join(config['temp_folder'], "predict.json"))
        print("prediction information saved!")
        pred_dict.clear()
        
        ann_dict = load_dict_json(os.path.join(config['temp_folder'], "annotations.json"))
        pred_dict = load_dict_json(os.path.join(config['temp_folder'], "predict.json"))
        
        ################################# 
        # Calculate video-mAP
        def create_tubes(info_dict):  
            def id_dict():
                return defaultdict(list)
            tubes = defaultdict(id_dict)
            video_list = info_dict.keys()
            for video_name in tqdm(video_list, desc="Processing videos"):
                tt_list = list(info_dict[video_name].keys())
                frame_list = []
                for tt in tt_list:
                    frame_list.append(int(tt))
                frame_list.sort()
                #print(frame_list)
                tracker = BoostTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device='cuda', half=False)
                for frame in frame_list:
                    boxes = info_dict[video_name][str(frame)]['boxes']
                    img_path = info_dict[video_name][str(frame)]['img']

                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (config['img_size'], config['img_size']))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    #print(type(image))
                    res = tracker.update(np.array(boxes), image)
                        
                    #tracker.plot_results(image, show_trajectories=True)
                    
                    for box in res:
                        tubes[video_name][int(box[4])].append({
                            'box': list(box[:4]),
                            'conf': box[5],
                            'class': int(box[6]),
                            'frame_id': frame
                        })
            return tubes
        
        #1. Create tubes
        print("Creating annotation tubes...")
        ann_tubes = create_tubes(ann_dict)
        ann_dict.clear()
        print("Annotation tubes created!")
        print()

        print("Creating prediction tubes...")
        pred_tubes = create_tubes(pred_dict)
        pred_dict.clear()
        print("Prediction tubes created!")
        print()
        
        
        save_dict_json(ann_tubes, os.path.join(config['temp_folder'], "ann_tubes.json"))
        ann_tubes.clear()
        save_dict_json(pred_tubes, os.path.join(config['temp_folder'], "pred_tubes.json"))
        pred_tubes.clear()
        
        #2. Convert format
        # {video_name: {id: {'box': [], 'conf':, 'class': , 'frame_id'}}}
        ann_tubes = load_dict_json(os.path.join(config['temp_folder'], "ann_tubes.json"))
        pred_tubes = load_dict_json(os.path.join(config['temp_folder'], "pred_tubes.json"))
        
        #a. convert ann
        print("Converting annotation format...")
        ann_vmAP = {
            "videos": [],
            "annotations": [],
            "categories": []
        }

        # For "videos" field
        video_list = list(ann_tubes.keys())
        videoname2index = {}
        for i, video_name in enumerate(video_list):
            ann_vmAP['videos'].append({
                "files_name": video_name,
                "width": config['img_size'],
                "height": config['img_size'],
                'id': i
            })
            videoname2index[video_name] = i

            
        # For "annotations" field
        id = 0
        for video_name in video_list:
            tube_id_tt = list(ann_tubes[video_name].keys())
            tube_list = []
            for tid in tube_id_tt:
                tube_list.append(int(tid))
                
            for tube_id in tube_list:
                tube = {
                    "id": id,
                    "video_id": videoname2index[video_name],
                    "category_id": None,
                    "track": []
                }
                for item in ann_tubes[video_name][str(tube_id)]:
                    tube['track'].append({
                        'bbox': item['box'],
                        'frame': item['frame_id'],
                        'outside': 0,
                        'occluded': 0
                    })
                    class_id = item['class']
                tube['category_id'] = class_id
                ann_vmAP['annotations'].append(tube)
                id += 1            

        # For "categories" field
        for class_id in config['idx2name'].keys():
            ann_vmAP['categories'].append({
                'id': int(class_id),
                'name': config['idx2name'][class_id]
            })
        save_dict_json(ann_vmAP, os.path.join(config['temp_folder'], 'final_ann.json'))
        ann_vmAP.clear()
        print("Annotation format converted!")
        print()
        
        #b. convert pred
        print("Converting prediction format...")
        id = 0
        pred_vmAP = []
        video_list = list(pred_tubes.keys())
        for video_name in video_list:
            tube_id_tt = list(pred_tubes[video_name].keys())
            tube_list = []
            for tid in tube_id_tt:
                tube_list.append(int(tid))
                
            for tube_id in tube_list:
                tube = {
                    "id": id,
                    "video_id": videoname2index[video_name],
                    "category_id": None,
                    "track": []
                }
                for item in pred_tubes[video_name][str(tube_id)]:
                    tube['track'].append({
                        'bbox': item['box'],
                        'frame': item['frame_id'],
                        'confidence': item['conf'],
                        'outside': 0,
                        'occluded': 0
                    })
                    class_id = item['class']
                tube['category_id'] = class_id
                id += 1
                pred_vmAP.append(tube)
                
        save_dict_json(pred_vmAP, os.path.join(config['temp_folder'], 'final_pred.json'))
        pred_vmAP.clear()
        print("Prediction format converted!")
        #################################
        