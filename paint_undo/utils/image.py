import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from a video file and save them as images.
    
    Parameters:
    video_path (str): Path to the video file.
    output_dir (str): Directory where extracted frames will be saved.
    frame_interval (int): Interval between frames to extract (default is 1, extract every frame).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Construct the output frame file path
            frame_file = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            # Save the frame as an image file
            cv2.imwrite(frame_file, frame)
            saved_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} and saved to {output_dir}")