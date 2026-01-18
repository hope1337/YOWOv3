import cv2
import os

def images_to_video(image_folder, output_video_path, fps):
    # Lấy danh sách ảnh và sắp xếp
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort() # Quan trọng: Sắp xếp để video không bị nhảy frame

    if not images:
        print(f"[Bỏ qua] Folder trống: {image_folder}")
        return

    # Đọc ảnh đầu tiên để lấy kích thước
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    # cv2.destroyAllWindows() # Không cần thiết khi chạy script hàng loạt

def process_dataset(input_root, output_root, fps=24):
    # Duyệt qua các folder Action (VD: Basketball, Tennis...)
    if not os.path.exists(input_root):
        print(f"Lỗi: Không tìm thấy đường dẫn {input_root}")
        return

    actions = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    print(f"Tìm thấy {len(actions)} loại hành động. Bắt đầu xử lý...")

    for action in actions:
        action_src_path = os.path.join(input_root, action)
        action_dst_path = os.path.join(output_root, action)
        
        # 1. Tạo folder Action tương ứng ở đầu ra (giữ cấu trúc thư mục)
        os.makedirs(action_dst_path, exist_ok=True)

        # Duyệt qua các folder Video con (VD: v_Basketball_g01_c01)
        video_folders = [d for d in os.listdir(action_src_path) if os.path.isdir(os.path.join(action_src_path, d))]
        
        for video_folder in video_folders:
            src_frames_path = os.path.join(action_src_path, video_folder)
            
            # Tên file video đầu ra (VD: v_Basketball_g01_c01.mp4)
            # Video sẽ nằm trong: data/ucf24/videos_generated/Basketball/v_Basketball_g01_c01.mp4
            dst_video_path = os.path.join(action_dst_path, video_folder + ".mp4")

            # Kiểm tra nếu video đã tồn tại thì bỏ qua (tiết kiệm thời gian nếu chạy lại)
            if os.path.exists(dst_video_path):
                print(f"[Đã có] {dst_video_path}")
                continue
            
            print(f"Đang tạo: {dst_video_path}")
            images_to_video(src_frames_path, dst_video_path, fps)

    print("\n>>> HOÀN TẤT TOÀN BỘ! <<<")

# --- CẤU HÌNH ---
# Đường dẫn tương đối từ folder YOWOv3
INPUT_DIR = 'data/ucf24/rgb-images'       
OUTPUT_DIR = 'data/ucf24/videos_generated' # Folder chứa kết quả
FPS = 25 # UCF24 gốc thường là 25fps, fen có thể chỉnh lại

# Chạy
process_dataset(INPUT_DIR, OUTPUT_DIR, FPS)