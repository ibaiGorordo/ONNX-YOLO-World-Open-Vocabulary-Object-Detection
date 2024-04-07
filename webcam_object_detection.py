import cv2

from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings

model_path = "models/yolov8l-worldv2.onnx"
embed_path = "data/glasses_embeddings.npz"

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)
print("Detecting classes:", class_list)

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.1, iou_thres=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize DetectionDrawer
drawer = DetectionDrawer(class_list)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect Objects
    boxes, scores, class_ids = yoloworld_detector(frame, class_embeddings)

    # Draw detections
    combined_img = drawer(frame, boxes, scores, class_ids)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
