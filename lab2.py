import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model with pretrained weights and preprocessing transforms
weights = Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model.to(device).eval()
transform = weights.transforms()

# Initialize tracker
tracker = BoostTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)

# Start video capture
cap = cv2.VideoCapture(0)

with torch.inference_mode():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB and prepare for model
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
        input_tensor = transform(tensor).to(device)

        # Run detection
        output = model([input_tensor])[0]
        scores = output['scores'].cpu().numpy()
        keep = scores >= 0.5

        # Prepare detections for tracking
        boxes = output['boxes'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()
        filtered_scores = scores[keep]
        detections = np.concatenate([boxes, filtered_scores[:, None], labels[:, None]], axis=1)

        # Update tracker and draw results
        #   INPUT:  M X (x, y, x, y, conf, cls)
        #   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
        res = tracker.update(detections, frame)
        tracker.plot_results(frame, show_trajectories=True)

        # Show output
        cv2.imshow('BoXMOT + Torchvision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()