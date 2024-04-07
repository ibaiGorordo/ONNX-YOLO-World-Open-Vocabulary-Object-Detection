import cv2
from imread_from_url import imread_from_url
from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings

model_path = "models/yolov8l-worldv2.onnx"
embed_path = "data/panda_embeddings.npz"

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)
print("Detecting classes:", class_list)

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.001, iou_thres=0.5)

# Initialize DetectionDrawer
drawer = DetectionDrawer(class_list)

img_url = "https://upload.wikimedia.org/wikipedia/commons/3/33/IFC_mall_2013_Chinese_New_Year_%28Hong_Kong%29.jpg"
img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids = yoloworld_detector(img, class_embeddings)

# Draw detections
combined_img = drawer(img, boxes, scores, class_ids)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", combined_img)
cv2.waitKey(0)

# cv2.imwrite("doc/img/detected_objects.png", combined_img)
