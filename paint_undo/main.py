import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip
from rembg import remove
import shutil
import base64

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def post_process_frame(frame, width, height):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return the original frame
    if not contours:
        return frame
    
    # Calculate center of the image
    center_x, center_y = width // 2, height // 2
    
    # Function to calculate score based on size and position
    def calculate_score(contour):
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        max_distance = np.sqrt(width**2 + height**2) / 2
        position_score = 1 - (distance_from_center / max_distance)
        return area * position_score
    
    # Find the contour with the highest score
    best_contour = max(contours, key=calculate_score)
    
    # Create a mask for the best contour
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], 0, 255, -1)
    
    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

def extract_frames(video_path, start_time, end_time, target_fps=None):
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    original_fps = clip.fps
    
    if target_fps is None or target_fps >= original_fps:
        frames = [frame for frame in clip.iter_frames()]
        return frames, original_fps
    else:
        frame_interval = original_fps / target_fps
        frames = [frame for i, frame in enumerate(clip.iter_frames()) if i % frame_interval < 1]
        return frames, target_fps

def remove_background(frames, output_dir, fps):
    bg_removed_frames = []
    post_processed_frames = []
    bg_removed_dir = os.path.join(output_dir, "bg_removed_frames")
    post_processed_dir = os.path.join(output_dir, "post_processed_frames")
    create_directory(bg_removed_dir)
    create_directory(post_processed_dir)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, frame in enumerate(frames):
        # Remove background
        bg_removed_frame = remove(frame)
        bg_removed_frames.append(bg_removed_frame)
        cv2.imwrite(os.path.join(bg_removed_dir, f"bg_removed_frame_{i:04d}.png"), cv2.cvtColor(bg_removed_frame, cv2.COLOR_RGBA2BGRA))
        
        # Post-process frame
        height, width = frame.shape[:2]
        post_processed_frame = post_process_frame(bg_removed_frame, width, height)
        post_processed_frames.append(post_processed_frame)
        cv2.imwrite(os.path.join(post_processed_dir, f"post_processed_frame_{i:04d}.png"), cv2.cvtColor(post_processed_frame, cv2.COLOR_RGBA2BGRA))
        
        # Update progress bar and status text
        progress = (i + 1) / len(frames)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {i+1}/{len(frames)}")
    
    # Clear the status text
    status_text.empty()
    
    return post_processed_frames

def create_video_with_audio(frames, audio_clip, output_path, fps):
    temp_video = ImageSequenceClip(frames, fps=fps)
    final_video = temp_video.set_audio(audio_clip)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


def display_video(video_path, width=300):
    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Calculate height maintaining aspect ratio
    height = int((width / orig_width) * orig_height)

    # Read video file and encode to base64
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    b64 = base64.b64encode(video_bytes).decode()

    # Display the video
    st.markdown(f"""
        <video width="{width}" height="{height}" controls>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """, 
        unsafe_allow_html=True
    )

def main():
    st.title("Video Processing App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Create output directory
        video_name = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join("output", video_name)
        create_directory(output_dir)

        # Display the uploaded video
        st.subheader("Uploaded Video")
        display_video(video_path)

        # Get video info
        clip = VideoFileClip(video_path)
        duration = clip.duration
        original_fps = clip.fps
        clip.close()

        # Frame extraction range slider
        st.subheader("Select Frame Extraction Range")
        start_time, end_time = st.slider("Select range", 0.0, duration, (0.0, duration))

        # FPS control
        st.subheader("FPS Control")
        target_fps = st.number_input("Target FPS (leave empty to use original FPS)", min_value=1, max_value=int(original_fps), value=int(original_fps/2))
        if target_fps == "":
            target_fps = None

        # Extract Frames
        if st.button("Extract Frames"):
            with st.spinner("Extracting frames... This may take a while."):
                frames, used_fps = extract_frames(video_path, start_time, end_time, target_fps)
                st.session_state['frames'] = frames
                st.session_state['used_fps'] = used_fps
                
                # Create and save cropped video
                cropped_clip = VideoFileClip(video_path).subclip(start_time, end_time)
                cropped_video_path = os.path.join(output_dir, "cropped_video.mp4")
                cropped_clip.write_videofile(cropped_video_path)
                
            st.success(f"Extracted {len(frames)} frames at {used_fps} FPS and saved cropped video")
            st.subheader("Cropped Video")
            display_video(cropped_video_path)

        # Background removal and post-processing
        if 'frames' in st.session_state and st.button("Remove Background and Post-process"):
            with st.spinner("Removing background and post-processing... This may take a while."):
                if 'post_processed_frames' not in st.session_state:
                    post_processed_frames = remove_background(st.session_state['frames'], output_dir, st.session_state['used_fps'])
                    st.session_state['post_processed_frames'] = post_processed_frames
                
                # Create video with audio
                audio_clip = AudioFileClip(video_path).subclip(start_time, end_time)
                post_processed_video_path = os.path.join(output_dir, "post_processed_video.mp4")
                create_video_with_audio(st.session_state['post_processed_frames'], audio_clip, post_processed_video_path, st.session_state['used_fps'])
            
            st.success("Background removal and post-processing completed and saved")
            st.subheader("Post-processed Video")
            display_video(post_processed_video_path)

        # Display processed frames
        if 'post_processed_frames' in st.session_state:
            st.subheader("Processed Frames")
            frame_index = st.slider("Select frame", 0, len(st.session_state['post_processed_frames']) - 1, 0)
            st.image(st.session_state['post_processed_frames'][frame_index])

        # One-click process
        if st.button("Process Video (All Steps)"):
            with st.spinner("Processing video... This may take a while."):
                # Extract frames if not already done
                if 'frames' not in st.session_state:
                    frames, used_fps = extract_frames(video_path, start_time, end_time, target_fps)
                    st.session_state['frames'] = frames
                    st.session_state['used_fps'] = used_fps

                # Remove background and post-process if not already done
                if 'post_processed_frames' not in st.session_state:
                    post_processed_frames = remove_background(st.session_state['frames'], output_dir, st.session_state['used_fps'])
                    st.session_state['post_processed_frames'] = post_processed_frames
                
                # Create video with audio
                audio_clip = AudioFileClip(video_path).subclip(start_time, end_time)
                final_video_path = os.path.join(output_dir, "final_video.mp4")
                create_video_with_audio(st.session_state['post_processed_frames'], audio_clip, final_video_path, st.session_state['used_fps'])
            
            st.success("Video processing completed and saved")
            st.subheader("Final Processed Video")
            display_video(final_video_path)

        # Clean up the temporary file
        os.unlink(video_path)

        st.success(f"All processed files have been saved in the '{output_dir}' directory")

if __name__ == "__main__":
    main()