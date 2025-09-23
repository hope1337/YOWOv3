import torch

def collate_fn(batch_data):
    clips  = []
    boxes  = []
    labels = []
    meta_data = []
    flag = (len(batch_data[0]) == 4)

    for b in batch_data:
        clips.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        if flag:
            meta_data.append(b[3])
    
    clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
    if flag: 
        return clips, boxes, labels, meta_data
    else:
        return clips, boxes, labels