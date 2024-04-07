import cv2
from cap_from_youtube import cap_from_youtube

from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings

model_path = "models/yolov8x-worldv2.onnx"
embed_path = "data/video_embeddings.npz"

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.1, iou_thres=0.5)
num_classes = yoloworld_detector.num_classes
print("Number of classes:", num_classes, "Number of classes loaded:", len(class_list))

# Initialize video
# cap = cv2.VideoCapture("input.mp4")
videoUrl = 'https://youtu.be/Atkp8mklOh0?si=MsFhQJZJDsjyQTqF'
cap = cap_from_youtube(videoUrl, resolution='720p')
start_time = 0 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

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

    # Draw what label we are detecting
    # cv2.putText(combined_img, f"Detecting: {class_list}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detected Objects", combined_img)
    # out.write(combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
