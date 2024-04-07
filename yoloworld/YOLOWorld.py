import time
import cv2
import numpy as np
import onnxruntime

from yoloworld.nms import nms


def read_class_embeddings(embed_path):
    data = np.load(embed_path)
    return data["class_embeddings"], data["class_list"]


class YOLOWorld:

    def __init__(self, path, conf_thres=0.3, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image, class_embeddings):
        return self.detect_objects(image, class_embeddings)

    def detect_objects(self, image, class_embeddings):

        if class_embeddings.shape[1] != self.num_classes:
            raise ValueError(f"Number of classes in the class embeddings should be {self.num_classes}")

        input_tensor = self.prepare_input(image)

        # Perform yoloworld on the image
        outputs = self.inference(input_tensor, class_embeddings)

        return self.process_output(outputs)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor, class_embeddings):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: input_tensor, self.input_names[1]: class_embeddings})

        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = YOLOWorld.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.num_classes = model_inputs[1].shape[1]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from yoloworld.DetectionDrawer import DetectionDrawer
    from imread_from_url import imread_from_url

    model_path = "../models/yolov8l-worldv2.onnx"
    embed_path = "../data/panda_embeddings.npz"

    # Load class embeddings
    class_embeddings, class_list = read_class_embeddings(embed_path)

    # Initialize YOLO-World object detector
    yoloworld_detector = YOLOWorld(model_path, conf_thres=0.3, iou_thres=0.5)

    # Initialize DetectionDrawer
    drawer = DetectionDrawer(class_list)

    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)

    # Detect Objects
    boxes, scores, class_ids = yoloworld_detector(img, class_embeddings)

    # Draw detections
    combined_img = drawer(img, boxes, scores, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
