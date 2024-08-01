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
    
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size
    transparent_layer = np.zeros((h, w, 4))
    for index, track in enumerate(tracking_points.value):
        if trackings_input_label.value[index] == 1:
            cv2.circle(transparent_layer, track, 20, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, 20, (255, 0, 0, 255), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
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

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    combined_images = []  # List to store filenames of images with masks overlaid
    mask_images = []      # List to store filenames of separate mask images

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # ---- Original Image with Mask Overlaid ----
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)  # Draw the mask with borders
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')

        # Save the figure as a JPG file
        combined_filename = f"combined_image_{i+1}.jpg"
        plt.savefig(combined_filename, format='jpg', bbox_inches='tight')
        combined_images.append(combined_filename)

        plt.close()  # Close the figure to free up memory

        # ---- Separate Mask Image (White Mask on Black Background) ----
        # Create a black image
        mask_image = np.zeros_like(image, dtype=np.uint8)
        
        # The mask is a binary array where the masked area is 1, else 0.
        # Convert the mask to a white color in the mask_image
        mask_layer = (mask > 0).astype(np.uint8) * 255
        for c in range(3):  # Assuming RGB, repeat mask for all channels
            mask_image[:, :, c] = mask_layer

        # Save the mask image
        mask_filename = f"mask_image_{i+1}.png"
        Image.fromarray(mask_image).save(mask_filename)
        mask_images.append(mask_filename)

        plt.close()  # Close the figure to free up memory

    return combined_images, mask_images

def sam_process(input_image, tracking_points, trackings_input_label):
    image = Image.open(input_image)
    image = np.array(image.convert("RGB"))

    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

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
    logits = logits[sorted_ind]

    print(masks.shape)

    results, mask_results = show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=False)
    print(results)

    return results[0], mask_results[0]

with gr.Blocks() as demo:
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    trackings_input_label = gr.State([])
    with gr.Column():
        gr.Markdown("# SAM2 Image Predictor")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="input image", interactive=False, type="filepath", visible=False)
                with gr.Row():
                    point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include")
                    clear_points_btn = gr.Button("Clear Points")
                with gr.Column():
                    points_map = gr.Image(
                        label="points map", 
                        type="filepath",
                        interactive=True
                    )
                    submit_btn = gr.Button("Submit")
            with gr.Column():
                output_result = gr.Image()
                output_result_mask = gr.Image()
    
    clear_points_btn.click(
        fn = preprocess_image,
        inputs = input_image, 
        outputs = [first_frame_path, tracking_points, trackings_input_label, points_map],
        queue=False
    )
    
    points_map.upload(
        preprocess_image, 
        points_map, 
        [first_frame_path, tracking_points, trackings_input_label, input_image],
        queue=False
    )

    points_map.select(
        get_point, 
        [point_type, tracking_points, trackings_input_label, first_frame_path], 
        [tracking_points, trackings_input_label, points_map], 
        queue=False
    )


    submit_btn.click(
        fn = sam_process,
        inputs = [input_image, tracking_points, trackings_input_label],
        outputs = [output_result, output_result_mask]
    )
demo.launch()