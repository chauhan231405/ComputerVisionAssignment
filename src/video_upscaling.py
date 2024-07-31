# upscaling_video.py
import sys
import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from models.edsr import EDSR

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR(num_channels=3, num_blocks=8, num_features=32, scale_factor=2).to(device)
model.load_state_dict(torch.load(r'/config/workspace/src/models/trained_model.pth'))
model.eval()

# Define input and output paths
input_folder = r'/config/workspace/data/images'
output_video_path = r'/config/workspace/data/upscaled_video.mp4'

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Frames per second
frame_size = (1280, 720)  # Final output resolution
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Prepare transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ensure consistent size for model input
    transforms.ToTensor()
])

# Process and upscale frames
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith('.png'):
        frame_path = os.path.join(input_folder, filename)
        image = Image.open(frame_path).convert('RGB')  # Convert to RGB
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image).squeeze().cpu()

        output_image = transforms.ToPILImage()(output)
        output_frame = output_image.resize(frame_size, Image.Resampling.LANCZOS)  # Resize to final frame size

        output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Debug output
        print(f'Processing frame: {filename}, shape: {output_frame.shape}')
        
        video_writer.write(output_frame)

video_writer.release()

# Check if file was created successfully
if os.path.exists(output_video_path):
    print(f'Upscaled video saved to {output_video_path}')
else:
    print(f'Failed to save video to {output_video_path}')
