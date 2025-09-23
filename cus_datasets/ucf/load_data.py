import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from .transforms import Augmentation, UCF_transform
from PIL import Image
            
class UCF_dataset(data.Dataset):

    def __init__(self, root_path, split_path, 
                 data_path, ann_path, clip_length, 
                 sampling_rate, img_size, phase, 
                 transform=Augmentation()):
        self.root_path     = root_path                                        # path to root folder
        self.split_path    = os.path.join(root_path, split_path)              # path to split file
        self.data_path     = os.path.join(root_path, data_path)               # path to data folder
        self.ann_path      = os.path.join(root_path, ann_path)                # path to annotation foler
        self.transform     = transform
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate
        self.phase         = phase

        with open(self.split_path, 'r') as f:
            self.lines = f.readlines()

        self.nSample       = len(self.lines)
        self.img_size      = img_size

    def __len__(self):
        return self.nSample
    
    def __getitem__(self, index, get_origin_image=False):
        key_frame_path = self.lines[index].rstrip()                   # e.g : labels/Basketball/v_Basketball_g08_c01/00070.txt
        # for linux, replace '/' by '\' for window 
        split_parts    = key_frame_path.split('/')                    # e.g : ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        key_frame_idx  = int(split_parts[-1].split('.')[-2])          # e.g : 70
        video_name     = split_parts[-2]                              # e.g : v_Basketball_g08_c01
        class_name     = split_parts[1]                               # e.g : Baseketball
        video_path     = os.path.join(self.data_path, class_name, video_name) 
        ann_path       = os.path.join(self.ann_path, class_name, video_name)
        # e.g : /home/manh/Datasets/UCF101-24/ucf24/rgb-images/Basketball/v_Basketball_g08_c01

        path = os.path.join(class_name, video_name) # e.g : Basketball/v_Basketball_g08_c01
        clip        = []
        boxes       = []
        meta_data   = None
        if self.phase == "testvmAP":
            meta_data = [path, key_frame_idx, os.path.join(video_path, '{:05d}.jpg'.format(key_frame_idx))]
            
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i*self.sampling_rate

            if cur_frame_idx < 1:
                cur_frame_idx = 1

            # get frame
            cur_frame_path = os.path.join(video_path, '{:05d}.jpg'.format(cur_frame_idx))
            #cur_frame      = cv2.imread(cur_frame_path)/255.0
            #H, W, C        = cur_frame.shape
            #cur_frame      = cv2.resize(cur_frame, self.img_size)
            #clip.append(cur_frame)
            cur_frame = Image.open(cur_frame_path).convert('RGB')
            clip.append(cur_frame)

        if get_origin_image == True or self.phase == "testvmAP":
            key_frame_path     = os.path.join(video_path, '{:05d}.jpg'.format(key_frame_idx))
            original_image     = cv2.imread(key_frame_path)

        # get annotation for key frame
        ann_file_name = os.path.join(ann_path, '{:05d}.txt'.format(key_frame_idx))
        boxes  = []
        labels = []

        with open(ann_file_name) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.rstrip().split(' ')
                label = int(line[0]) - 1

                if self.phase == 'train':
                    onehot_vector = np.zeros(24)
                    onehot_vector[label] = 1.
                    labels.append(onehot_vector)
                elif self.phase == 'test' or self.phase == 'testvmAP':
                    labels.append(label)

                box = [float(line[1]), float(line[2]), float(line[3]), float(line[4])] 
                boxes.append(box)

        boxes = np.array(boxes)

        if self.phase == 'train':   
            labels = np.array(labels)
        elif self.phase == 'test' or self.phase == 'testvmAP':
            labels = np.expand_dims(np.array(labels), axis=1)    

        targets = np.concatenate((boxes, labels), axis=1)
        clip, targets = self.transform(clip, targets)
        
        boxes = targets[:, :4]

        if self.phase == 'train':
            labels = targets[:, 4:]
        elif self.phase == 'test' or self.phase == 'testvmAP':
            labels = targets[:, -1]

        if get_origin_image == True : 
            return original_image, clip, boxes, labels
        elif self.phase == "testvmAP":
            return clip, boxes, labels, meta_data
        else:
            return clip, boxes, labels
        
        
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv() 
    
    root_path = os.getenv("YOWO_UCF_ROOT_PATH")
    split_path = "trainlist.txt"
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = 16
    sampling_rate = 1

    dataset = UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate, img_size=(224, 224))
    
    for i in range(13000, dataset.__len__()):
        original_image, clip, boxes, labels = dataset.__getitem__(i, get_origin_image=True)
        original_image = clip[:, -1, :, :].squeeze(1).permute(1, 2, 0).contiguous().numpy()

        for box, label in zip(boxes, labels):
            H, W, C = original_image.shape

            pt1 = (int(box[0] * W), int(box[1] * H))
            pt2 = (int(box[2] * W), int(box[3] * H))

            cv2.rectangle(original_image, pt1, pt2, 1, 1, 1)
            print(label)
    
        cv2.imshow('img', original_image)
        k = cv2.waitKey()
        if k == ord('q'):
            break


def build_ucf_dataset(config, phase):
    root_path     = config['data_root']
    data_path     = "rgb-images"
    ann_path      = "labels"
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size      = config['img_size']

    if phase == 'train':
        split_path    = "trainlist.txt"
        return UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate, img_size, transform=Augmentation(img_size=img_size), phase=phase)
    elif phase == 'test' or phase == 'testvmAP':
        split_path    = "testlist.txt"
        return UCF_dataset(root_path, split_path, data_path, ann_path
                          , clip_length, sampling_rate, img_size, transform=UCF_transform(img_size=img_size), phase=phase)