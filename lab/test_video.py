import cv2
import os

def get_video_names_from_file(file_path):
    """
    Đọc file list (train hoặc test) và trả về tập hợp (set) các tên video.
    """
    video_names = set()
    
    if not os.path.exists(file_path):
        print(f"[Cảnh báo] Không tìm thấy file: {file_path}")
        return set()

    print(f"-> Đang đọc file: {file_path} ...")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Format: labels/Action/VideoName/Frame.txt
            parts = line.split('/')
            if len(parts) >= 3:
                video_name = parts[2]
                video_names.add(video_name)
    
    return video_names

def images_to_video(image_folder, output_video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort() 

    if not images: return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None: return
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def process_dataset_strict(input_root, output_root, train_list_path, test_list_path, fps=24):
    # 1. Lấy danh sách video từ cả 2 file
    print("--- BƯỚC 1: KIỂM TRA DỮ LIỆU ---")
    train_videos = get_video_names_from_file(train_list_path)
    test_videos = get_video_names_from_file(test_list_path)

    print(f"   + Số lượng video Train: {len(train_videos)}")
    print(f"   + Số lượng video Test : {len(test_videos)}")

    # 2. Kiểm tra trùng lặp (Intersection)
    overlap = train_videos.intersection(test_videos)
    
    if len(overlap) > 0:
        print(f"\n[CẢNH BÁO NGUY HIỂM] Phát hiện {len(overlap)} video nằm trong cả Train và Test!")
        print(f"Danh sách trùng: {list(overlap)[:5]} ...") 
        # Tùy chọn: Xóa video trùng khỏi danh sách Test để đảm bảo an toàn
        test_videos = test_videos - overlap
        print("-> Đã tự động LOẠI BỎ các video trùng này khỏi danh sách cần tạo.")
    else:
        print("\n[OK] Dữ liệu sạch. Không có video nào bị trùng giữa Train và Test.")

    if not test_videos:
        print("Không có video test nào để xử lý.")
        return

    # 3. Bắt đầu tạo video
    print("\n--- BƯỚC 2: TẠO VIDEO TEST ---")
    if not os.path.exists(input_root):
        print(f"Không tìm thấy thư mục ảnh gốc: {input_root}")
        return

    actions = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    count = 0
    for action in actions:
        action_src_path = os.path.join(input_root, action)
        video_folders = [d for d in os.listdir(action_src_path) if os.path.isdir(os.path.join(action_src_path, d))]
        
        for video_folder in video_folders:
            # --- BỘ LỌC NGHIÊM NGẶT ---
            # Chỉ xử lý nếu video nằm trong danh sách Test (đã lọc trùng)
            if video_folder not in test_videos:
                continue 
            
            # Check ngược lại lần nữa: Nếu nó nằm trong Train thì bỏ qua ngay (Double check)
            if video_folder in train_videos:
                continue

            action_dst_path = os.path.join(output_root, action)
            os.makedirs(action_dst_path, exist_ok=True)
            
            dst_video_path = os.path.join(action_dst_path, video_folder + ".mp4")
            src_frames_path = os.path.join(action_src_path, video_folder)

            if os.path.exists(dst_video_path):
                continue

            print(f"[{count+1}] Creating: {video_folder}")
            images_to_video(src_frames_path, dst_video_path, fps)
            count += 1

    print(f"\n>>> HOÀN TẤT! Đã tạo {count} video TEST vào: {output_root} <<<")

# --- CẤU HÌNH ---
INPUT_DIR = 'data/ucf24/rgb-images'       
OUTPUT_DIR = 'data/ucf24/videos_test'      
TRAIN_LIST = 'data/ucf24/trainlist.txt'     # File danh sách Train
TEST_LIST = 'data/ucf24/testlist.txt'       # File danh sách Test
FPS = 25

# Chạy
process_dataset_strict(INPUT_DIR, OUTPUT_DIR, TRAIN_LIST, TEST_LIST, FPS)