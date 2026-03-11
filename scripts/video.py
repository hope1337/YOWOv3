
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

from math import sqrt

from cus_datasets.build_dataset import build_dataset
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from PIL import Image
from utils.flops import get_info

class live_transform():
    """
    Args:
        clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
        boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    Return:
        clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
        boxes : not change
    """

    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip, mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        W, H = img.size
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img

def detect(config, video_path=None, video_fps=24):
    """
    video_fps: FPS GỐC của video (cần biết để tính toán vị trí frame)
    """
    model = build_yowov3(config) 
    get_info(config, model)
    model.to("cuda")
    model.eval()
    mapping = config['idx2name']
    img_size = config['img_size']

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        # Nếu là Webcam thì bản thân nó đã là real-time, không cần logic này
        cap = cv2.VideoCapture(0)
        video_fps = 30 # Webcam thường là 30

    # Lấy tổng số frame của video để tránh seek quá lố
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_list = []
    transform = live_transform(img_size)
    
    # --- BIẾN QUẢN LÝ THỜI GIAN ---
    start_program_time = time.time() # Mốc thời gian bắt đầu chạy
    
    # Biến đếm FPS hiệu năng
    processed_count = 0
    start_process_metric = 0 
    current_avg_fps = 0.0

    while True:
        # 1. TÍNH TOÁN VỊ TRÍ FRAME CẦN ĐỌC DỰA TRÊN THỜI GIAN THỰC
        now = time.time()
        elapsed_time = now - start_program_time
        
        # Công thức: Frame hiện tại = Thời gian đã trôi qua * FPS của video
        target_frame_idx = int(elapsed_time * video_fps)

        # Kiểm tra nếu hết video
        if video_path and target_frame_idx >= total_frames:
            print("Hết video (theo thời gian thực)...")
            break
            # Hoặc muốn loop lại thì reset:
            # start_program_time = time.time()
            # continue

        # 2. NHẢY CÓC (SEEK) ĐẾN FRAME ĐÓ
        if video_path:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        
        # 3. ĐỌC FRAME
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = cv2.resize(frame, (img_size, img_size))
        origin_image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_list.append(transform(origin_image_pil))
        
        if (len(frame_list) > 16):
            frame_list.pop(0)
        
        # Buffering (Vẫn cần chờ đủ 16 frame đầu tiên)
        if (len(frame_list) < 16):
            cv2.putText(display_frame, f"Filling Buffer: {len(frame_list)}/16", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow('img', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # --- BẮT ĐẦU TÍNH FPS HIỆU NĂNG ---
        if start_process_metric == 0:
            start_process_metric = time.time()

        clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous()
        clip = clip.unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            outputs = model(clip)
            outputs = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)[0]

        # --- TÍNH TOÁN THÔNG SỐ ---
        processed_count += 1
        metric_elapsed = time.time() - start_process_metric
        if metric_elapsed > 0:
            current_avg_fps = processed_count / metric_elapsed

        if outputs is not None:
             draw_bounding_box(display_frame, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)

        # Hiển thị thông tin
        # Dòng 1: Frame thực tế của video đang hiển thị
        cv2.putText(display_frame, f"Video Frame: {target_frame_idx}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Dòng 2: Tốc độ xử lý của Model
        cv2.putText(display_frame, f"Model FPS: {current_avg_fps:.2f}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('img', display_frame)
        
        # Lưu ý: Ở đây KHÔNG CẦN time.sleep nữa
        # Vì vòng lặp sau quay lại, 'elapsed_time' đã tăng lên và 'target_frame_idx' sẽ tự nhảy cóc
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(processed_count)

if __name__ == "__main__":
    config = build_config()
    # Thay đường dẫn video của bro vào đây
    video_test = 'data/ucf24/videos_generated/Basketball/v_Basketball_g01_c01.mp4'
    detect(config, video_path=video_test, target_fps=25)