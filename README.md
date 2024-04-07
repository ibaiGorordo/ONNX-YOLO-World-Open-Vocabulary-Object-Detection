# ONNX-YOLO-World-Open-Vocabulary-Object-Detection

![!ONNX-YOLO-World-Open-Vocabulary-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLO-World-Open-Vocabulary-Object-Detection/raw/main/doc/img/gradio_demo.gif)

# Important
- If you know what labels you are going to detect, use the official export method. This repository is just to show how a model can be exported with the ability to accept the class embeddings as input maintaining the open vocabulary feature in ONNX.
- It is necessary to especify the number of classes when exporting the model. Setting a dynamic number of classes gives an error when running the model.
- For classes that are not in the pretrained datasets (e.g. COCO, Objects365...), the score can be very low (less than 1%), try reducing the confidence threshold to detect them.

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```shell
git clone https://github.com/ibaiGorordo/ONNX-YOLO-World-Open-Vocabulary-Object-Detection.git
cd ONNX-YOLO-World-Open-Vocabulary-Object-Detection
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
- Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lbVu_yA8GnWIbuU4STFvq0WS0p8bqTej?usp=sharing): If you don't want to install additional libraries
- Otherwise, use the `export_ultralytics_model.py` script to export the model to ONNX format. Select the number of classes you want as input. Default number of classes is 1.

# Class embeddings
- The Google Colab notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lbVu_yA8GnWIbuU4STFvq0WS0p8bqTej?usp=sharing) also includes the class embeddings generation.
- Otherwise, use the `save_class_embeddings.py` script to generate the class embeddings. 
- The class embeddings can be obtained using Openai CLIP model. The embeddings are stored in the `.npz` format, and it also includes the list of classes.
- The number of class embeddings in the `.npz` file does not need to match the number of classes in the model as long as you only pass the correct number of class embeddings to the model during inference.

# Original YOLO-World model
The original YOLO-World model can be found in this repository: [YOLO-World Repository](https://github.com/AILab-CVC/YOLO-World)
- The License of the models is GPL-3.0 license: [License](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE)

# Examples

 * **Image inference**:
 ```shell
 python image_object_detection.py
 ```

Exported using: 
 ```shell
python .\save_class_embeddings.py panda  --output_name panda_embeddings.npz
python .\export_ultralytics_model.py --model_name yolov8l-worldv2.pt --num_classes 1
 ```

 * **Gradio Image inference**:

It showcases the model's ability to detect different classes during inference. The model is exported with one class embedding as input, but the class can be selected from the list includded in the class embeddings file.
 ```shell
 python gradio_image_selected_class_detection.py
 ```

Exported using: 
 ```shell
python .\save_class_embeddings.py dog book-shelf chair table display keyboard earth-globe printer clock 
python .\export_ultralytics_model.py --model_name yolov8l-worldv2.pt --num_classes 1
 ```

 * **Webcam inference**:
 ```shell
 python webcam_object_detection.py
 ```

Exported using: 
 ```shell
python .\save_class_embeddings.py glasses  --output_name glass_embeddings.npz
python .\export_ultralytics_model.py --model_name yolov8l-worldv2.pt --num_classes 1
 ```

 * **Video inference**: https://youtu.be/U0857S7x1zc
 ```shell
 python video_object_detection.py
 ```

Exported using: 
 ```shell
python .\save_class_embeddings.py person car bike trash-can traffic-light traffic-cone van bus truck street-sign tree  --output_name video_embeddings.npz
python .\export_ultralytics_model.py --model_name yolov8x-worldv2.pt --num_classes 11
 ```

 ![!YOLO-World detection video](https://github.com/ibaiGorordo/ONNX-YOLO-World-Open-Vocabulary-Object-Detection/raw/main/doc/img/yoloworld_video.gif)

  *Original video: [https://youtu.be/Atkp8mklOh0?si=MsFhQJZJDsjyQTqF](https://youtu.be/Atkp8mklOh0?si=MsFhQJZJDsjyQTqF)*

# References:
* YOLO-World model: [https://github.com/AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
* Ultralytics: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* Ultralytics YOLO-World model: [https://docs.ultralytics.com/models/yolo-world/](https://docs.ultralytics.com/models/yolo-world/)
* Gradio: [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)