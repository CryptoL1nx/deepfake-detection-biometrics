import cv2
import os

def extract_frames(video_folder, output_folder, frames_per_video=5):
    os.makedirs(output_folder, exist_ok=True)
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    print(f'Found {len(videos)} videos in {video_folder}')
    total_frames = 0
    for video_file in videos:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // frames_per_video)
        saved = 0
        for i in range(frames_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                name = f"{video_file[:-4]}_frame{i}.jpg"
                cv2.imwrite(os.path.join(output_folder, name), frame)
                saved += 1
        cap.release()
        total_frames += saved
        print(f'  {video_file}: {saved} frames extracted')
    print(f'Total frames extracted: {total_frames}')

# Extract frames from deepfake videos
print("Extracting deepfake frames...")
extract_frames(
    r'C:\cs599\deepfake\faceforensics\manipulated_sequences\FaceSwap\c40\videos',
    r'C:\cs599\deepfake\sample_images\fake'
)

# Extract frames from original real videos
print("\nExtracting real video frames...")
extract_frames(
    r'C:\cs599\deepfake\faceforensics\original_sequences\youtube\c40\videos',
    r'C:\cs599\deepfake\sample_images\ff_real'
)

print("\nDone!")