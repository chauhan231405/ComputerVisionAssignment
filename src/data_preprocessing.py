import cv2
import os

input_video_path = r'/config/workspace/data/input_video.mp4'
output_folder = r'/config/workspace/data/images'

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()
print(f'Extracted {frame_count} frames to {output_folder}')
