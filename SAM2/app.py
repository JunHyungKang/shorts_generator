import gradio as gr

import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def preprocess_image(image):
    return image, gr.State([]), gr.State([]), image

def get_point(point_type, tracking_points, trackings_input_label, first_frame_path, evt: gr.SelectData):
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")

    tracking_points.value.append(evt.index)
    print(f"TRACKING POINT: {tracking_points.value}")

    if point_type == "include":
        trackings_input_label.value.append(1)
    elif point_type == "exclude":
        trackings_input_label.value.append(0)
    print(f"TRACKING INPUT LABEL: {trackings_input_label.value}")

    # Open the image and get its dimensions
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction = 0.02  # You can adjust this value as needed
    radius = int(fraction * min(w, h))

    # Create a transparent layer to draw on
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(tracking_points.value):
        if trackings_input_label.value[index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, 'RGBA')
    selected_point_map = Image.alpha_composite(transparent_background, transparent_layer)

    return tracking_points, trackings_input_label, selected_point_map

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


def post_process_mask(mask, smooth_factor=15, dilate_factor=5):
    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)

    # Apply morphological operations to smooth the mask
    kernel = np.ones((smooth_factor, smooth_factor), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilate the mask slightly to cover any small gaps
    kernel = np.ones((dilate_factor, dilate_factor), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply Gaussian blur to further smooth the edges
    mask = cv2.GaussianBlur(mask, (smooth_factor, smooth_factor), 0)

    # Threshold the mask to make it binary again
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def show_masks(image, masks, scores):
    extracted_images = []  # List to store filenames of extracted object images

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Post-process the mask
        processed_mask = post_process_mask(mask)

        # Create a white background
        extracted_image = np.ones_like(image) * 255

        # Copy the masked area from the original image
        extracted_image[processed_mask > 0] = image[processed_mask > 0]

        # Save the extracted image
        extracted_filename = f"extracted_image_{i+1}.png"
        Image.fromarray(extracted_image).save(extracted_filename)
        extracted_images.append(extracted_filename)

    return extracted_images[0] if extracted_images else None


def sam_process(input_image, checkpoint, tracking_points, trackings_input_label):
    image = Image.open(input_image)
    image = np.array(image.convert("RGB"))

    if checkpoint == "tiny":
        sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
    elif checkpoint == "samll":
        sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
    elif checkpoint == "base-plus":
        sam2_checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"
    elif checkpoint == "large":
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(image)

    input_point = np.array(tracking_points.value)
    input_label = np.array(trackings_input_label.value)

    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    extracted_result = show_masks(image, masks, scores)

    return extracted_result


with gr.Blocks() as demo:
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    trackings_input_label = gr.State([])
    with gr.Column():
        gr.Markdown("# SAM2 Image Predictor")
        gr.Markdown("This is a simple demo for image segmentation with SAM2.")
        gr.Markdown("""Instructions: 
        
        1. Upload your image 
        2. With 'include' point type selected, Click on the object to mask
        3. Switch to 'exclude' point type if you want to specify an area to avoid
        4. Submit !
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="input image", interactive=False, type="filepath", visible=False)                 
                points_map = gr.Image(
                    label="points map", 
                    type="filepath",
                    interactive=True
                )
                with gr.Row():
                    point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include")
                    clear_points_btn = gr.Button("Clear Points")
                checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus", "large"], value="tiny")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                output_result = gr.Image(label="Extracted Object")
                output_result_mask = gr.Image(label="Mask Result")

    clear_points_btn.click(
        fn = preprocess_image,
        inputs = input_image, 
        outputs = [first_frame_path, tracking_points, trackings_input_label, points_map],
        queue=False
    )

    points_map.upload(
        fn = preprocess_image, 
        inputs = [points_map], 
        outputs = [first_frame_path, tracking_points, trackings_input_label, input_image],
        queue = False
    )

    points_map.select(
        fn = get_point, 
        inputs = [point_type, tracking_points, trackings_input_label, first_frame_path], 
        outputs = [tracking_points, trackings_input_label, points_map], 
        queue = False
    )

    submit_btn.click(
        fn=sam_process,
        inputs=[input_image, checkpoint, tracking_points, trackings_input_label],
        outputs=[output_result],
    )

demo.launch(show_api=False, show_error=True)
