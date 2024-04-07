import numpy as np
import gradio as gr
import cv2
from imread_from_url import imread_from_url
from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings

model_path = "models/yolov8l-worldv2.onnx"
embed_path = "data/class_embeddings.npz"

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.1, iou_thres=0.5)

# Initialize DetectionDrawer
drawer = DetectionDrawer(class_list)

img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Interior_design_865875.jpg/800px-Interior_design_865875.jpg"
img = imread_from_url(img_url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_objects(img, threshold, class_name):

    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    yoloworld_detector.conf_threshold = threshold/100

    # Get class index
    class_index = np.where(class_list == class_name)[0][0]

    # Get class embedding
    class_embedding = class_embeddings[:, [class_index], :]

    # Detect Objects
    boxes, scores, class_ids = yoloworld_detector(bgr_img, class_embedding)
    class_ids += class_index

    # Draw detections
    combined_img = drawer(img, boxes, scores, class_ids)

    return combined_img

demo = gr.Interface(
    detect_objects,
    [
        gr.Image(value=img, label="Input Image"),
        gr.Slider(1, 100, step=1, value=10, label="Threshold (%)"),
        gr.Dropdown(class_list.tolist(), label="Select Class", value=class_list[0])
    ],
    "image",
    title="YOLO-World Open Vocabulary Object Detection",
    description="Demo to showcase the open vocabulary object detection using YOLO-World model in ONNX. The model has been exported with one class embedding as input. Select a class from the dropdown to detect objects in the image."
)
if __name__ == "__main__":
    demo.launch()