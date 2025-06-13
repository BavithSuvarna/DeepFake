import cv2
import os

def extract_frames(video_path, save_dir, every_n=10):
    os.makedirs(save_dir, exist_ok=True)

    # Determine how many frames already exist
    existing_frames = [f for f in os.listdir(save_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    existing_indices = sorted([int(f.split("_")[1].split(".")[0]) for f in existing_frames])
    start_index = existing_indices[-1] + 1 if existing_indices else 0
    start_frame = start_index * every_n

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip frames until start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    index = start_index

    print(f"Resuming from frame {start_frame} (index {start_index}) for {os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= total_frames:
            break
        if frame_count % every_n == 0:
            filename = os.path.join(save_dir, f"frame_{index}.jpg")
            cv2.imwrite(filename, frame)
            index += 1
        frame_count += 1

    cap.release()

def extract_from_folder(video_folder, output_root, every_n=10):
    os.makedirs(output_root, exist_ok=True)
    for filename in os.listdir(video_folder):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(video_folder, filename)
            video_name = os.path.splitext(filename)[0]
            save_dir = os.path.join(output_root, video_name)
            print(f"Processing {video_path}...")
            extract_frames(video_path, save_dir, every_n)
    print(f"✅ All videos in {video_folder} processed.\n")

# Extract frames from all videos in 'dataset/fake' and 'dataset/real'
extract_from_folder("dataset/fake", "frames/fake", every_n=10)
extract_from_folder("dataset/real", "frames/real", every_n=10)
