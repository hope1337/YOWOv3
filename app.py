import streamlit as st
import torch
import torchvision.transforms.functional as FT
import cv2
import numpy as np
import time
import yaml  # <--- Quan trọng: Dùng để đọc file config
from PIL import Image

# Import các module của bạn
from utils.box import draw_bounding_box, non_max_suppression
from model.TSN.YOWOv3 import build_yowov3

# --- 1. CLASS HELPER (Giữ nguyên) ---
class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip):
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

# --- 2. HÀM LOAD MODEL (Đã sửa lỗi đọc Config) ---
@st.cache_resource
def load_model(config_path):
    # Đọc trực tiếp file YAML từ đường dẫn user nhập
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file config tại '{config_path}'. Vui lòng kiểm tra lại đường dẫn.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi đọc file config: {e}")
        st.stop()

    model = build_yowov3(cfg)
    # Lưu ý: Nếu cần load weights riêng, bạn có thể thêm logic ở đây
    # model.load_state_dict(torch.load('path/to/weights.pth', map_location='cuda'))
        
    model.to("cuda")
    model.eval()
    return model, cfg

# --- 3. GIAO DIỆN CHÍNH ---
def main():
    st.set_page_config(layout="wide", page_title="YOWOv3 Action Detection")
    st.title("🎥 YOWOv3 Action Detection Dashboard")

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Settings")
    
    # Đường dẫn (Default trỏ đúng vào file weights của bạn)
    config_path = st.sidebar.text_input("Config Path (-cf)", value="weights/ucf_config.yaml")
    video_path = st.sidebar.text_input("Video Path (-vd)", value="data/ucf24/videos_test/Skijet/v_Skijet_g07_c03.mp4")
    
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)
    
    start_btn = st.sidebar.button("▶️ Start Detection")
    stop_btn = st.sidebar.button("⏹️ Stop")

    if start_btn:
        # --- BƯỚC 1: LOAD MODEL ---
        with st.spinner("Đang load model..."):
            model, config = load_model(config_path)
            img_size = config['img_size']
            mapping = config['idx2name']
            transform = live_transform(img_size)

        # --- BƯỚC 2: MỞ VIDEO ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Cannot open video: {video_path}")
            return

        # Setup giao diện 2 cột
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Video")
            orig_placeholder = st.empty()
        with col2:
            st.subheader("Processed Output (Real-time Sync)")
            proc_placeholder = st.empty()
        
        # --- BƯỚC 3: CHUẨN BỊ BIẾN CHO VÒNG LẶP ---
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 25 # Fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_list = []
        fps_metric = st.sidebar.empty() # Chỗ hiển thị FPS bên sidebar
        
        start_program_time = time.time() # Mốc thời gian bắt đầu
        prev_model_time = time.time()    # Mốc tính FPS model
        processed_count = 0
        
        # --- BƯỚC 4: VÒNG LẶP CHÍNH (Đã sync thời gian) ---
        while True:
            # Check nút Stop (Streamlit cần cách trick khác để stop mượt, nhưng tạm thời logic này OK)
            if stop_btn: 
                break

            # 4.1. Tính vị trí frame cần đọc
            now = time.time()
            elapsed_time = now - start_program_time
            target_frame_idx = int(elapsed_time * video_fps)

            if target_frame_idx >= total_frames:
                st.success("Video finished.")
                break

            # 4.2. Nhảy (Seek) đến frame đó
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret: break

            # 4.3. Hiển thị Video Gốc (Trái)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # 4.4. Xử lý Model
            display_frame = cv2.resize(frame.copy(), (img_size, img_size))
            pil_img = Image.fromarray(frame_rgb)
            
            frame_list.append(transform(pil_img))
            if len(frame_list) > 16:
                frame_list.pop(0)

            # Chỉ detect khi đủ buffer
            if len(frame_list) == 16:
                clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous()
                clip = clip.unsqueeze(0).to("cuda")

                with torch.no_grad():
                    outputs = model(clip)
                    outputs = non_max_suppression(outputs, conf_threshold=conf_thresh, iou_threshold=iou_thresh)[0]
                
                # Vẽ Box
                if outputs is not None:
                    draw_bounding_box(display_frame, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)

                # --- TÍNH TOÁN FPS MODEL (Mới - Giống CLI cũ) ---
                processed_count += 1 # Cần khai báo biến này = 0 ở trước while
                curr_time = time.time()
                total_time_elapsed = curr_time - start_program_time # Hoặc start_process_metric
                model_fps = processed_count / total_time_elapsed if total_time_elapsed > 0 else 0
                
                # Vẽ thông tin lên hình (Phải)
                cv2.putText(display_frame, f"Video Frame: {target_frame_idx}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Model FPS: {model_fps:.2f}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Cập nhật Sidebar metric
                fps_metric.metric("Model FPS", f"{model_fps:.2f}")

                # Hiển thị Video Kết quả
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                proc_placeholder.image(display_rgb, channels="RGB", use_container_width=True)
            
            else:
                # Màn hình chờ buffer
                waiting = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                cv2.putText(waiting, f"Buffering: {len(frame_list)}/16", (10, img_size//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                proc_placeholder.image(waiting, channels="RGB")

        cap.release()

if __name__ == "__main__":
    main()