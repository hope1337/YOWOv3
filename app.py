import streamlit as st
import torch
import torchvision.transforms.functional as FT
import cv2
import numpy as np
import time
from PIL import Image
import yaml # <--- Thêm dòng này
# ... các import khác

# Import các module của bạn (đảm bảo cấu trúc thư mục đúng như lúc chạy CLI)
from utils.box import draw_bounding_box, non_max_suppression
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config

# --- 1. ĐỊNH NGHĨA CLASS HELPER (Giữ nguyên logic của bạn) ---
class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip):
        # Chuẩn hóa theo ImageNet (hoặc theo config của bạn)
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

@st.cache_resource
def load_model(config_path, checkpoint_path=None):
    # --- THAY ĐỔI: Tự load config từ file YAML thay vì dùng build_config() mặc định ---
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file config tại: {config_path}")
        st.stop()
    # ---------------------------------------------------------------------------------

    # Tiếp tục logic cũ
    model = build_yowov3(cfg)
    
    # Nếu cần load checkpoint (weights)
    if checkpoint_path:
         model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
        
    model.to("cuda")
    model.eval()
    return model, cfg

# --- 3. GIAO DIỆN CHÍNH ---
def main():
    st.set_page_config(layout="wide", page_title="YOWOv3 Video Action Detection")
    st.title("🎥 YOWOv3 Action Detection Dashboard")

    # --- SIDEBAR: CÁC SETTING ---
    st.sidebar.header("⚙️ Settings")
    
    # Input paths
    config_path = st.sidebar.text_input("Config Path (-cf)", value="weights/ucf_config.yaml")
    video_path = st.sidebar.text_input("Video Path (-vd)", value="data/ucf24/videos_generated/Basketball/v_Basketball_g01_c01.mp4")
    
    # Model params
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Control buttons
    start_btn = st.sidebar.button("▶️ Start Detection")
    stop_btn = st.sidebar.button("⏹️ Stop")

    # --- LOGIC XỬ LÝ ---
    if start_btn:
        try:
            # Load Model
            with st.spinner("Loading Model..."):
                model, config = load_model(config_path)
                img_size = config['img_size']
                mapping = config['idx2name']
                transform = live_transform(img_size)

            # Setup Video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Cannot open video: {video_path}")
                return

            # Setup giao diện 2 cột
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Video")
                orig_placeholder = st.empty() # Placeholder để update ảnh
            with col2:
                st.subheader("Processed Output (Detection)")
                proc_placeholder = st.empty() # Placeholder để update ảnh
            
            # Biến buffer và fps
            frame_list = []
            fps_metric = st.sidebar.empty() # Hiển thị FPS ở sidebar
            
            # --- Trước vòng lặp, khai báo các biến đếm ---
            prev_time = time.time()
            frame_count = 0 

            while cap.isOpened():
                # Check nút Stop
                if stop_btn:
                    st.warning("Đã dừng video.")
                    break

                ret, frame = cap.read()
                if not ret:
                    st.info("Video finished.")
                    break

                # 1. Hiển thị Video Gốc (Bên trái)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # 2. Xử lý cho Model
                display_frame = cv2.resize(frame.copy(), (img_size, img_size))
                
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_list.append(transform(pil_img))

                if len(frame_list) > 16:
                    frame_list.pop(0)

                # Khi đủ buffer 16 frame thì mới detect
                if len(frame_list) == 16:
                    clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous()
                    clip = clip.unsqueeze(0).to("cuda")

                    with torch.no_grad():
                        outputs = model(clip)
                        outputs = non_max_suppression(outputs, conf_threshold=conf_thresh, iou_threshold=iou_thresh)[0]
                    
                    # Vẽ Box nếu có phát hiện
                    if outputs is not None:
                        draw_bounding_box(display_frame, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)
                    
                    # --- PHẦN KHÔI PHỤC: TÍNH VÀ IN FPS LÊN HÌNH ---
                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    fps = 1 / exec_time if exec_time > 0 else 0
                    prev_time = curr_time
                    frame_count += 1
                    
                    # In số Frame
                    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # In FPS (Màu xanh lá)
                    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # -----------------------------------------------

                    # Cập nhật metric ở sidebar (nếu muốn giữ)
                    fps_metric.metric("Model FPS", f"{fps:.2f}")

                    # Hiển thị Video Kết quả (Bên phải)
                    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    proc_placeholder.image(display_rgb, channels="RGB", use_container_width=True)
                
                else:
                    # Hiển thị màn hình chờ khi chưa đủ frame
                    waiting_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                    cv2.putText(waiting_img, f"Buffering: {len(frame_list)}/16", (10, img_size//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    proc_placeholder.image(waiting_img, channels="RGB")
                

            cap.release()

        except Exception as e:
            st.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()