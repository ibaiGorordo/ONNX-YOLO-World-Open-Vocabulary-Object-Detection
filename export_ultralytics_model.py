from argparse import ArgumentParser
import torch
from ultralytics import YOLOWorld

from export import ModelExporter


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="yolov8l-worldv2.pt")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    img_width = args.img_width
    img_height = args.img_height
    num_classes = args.num_classes
    model_name = args.model_name
    output_dir = args.output_dir
    device = args.device

    # Initialize model
    yoloModel = YOLOWorld(model_name)
    yoloModel.set_classes([""] * num_classes)

    # Initialize model exporter
    export_model = ModelExporter(yoloModel, device)

    # Export model
    export_model.export(output_dir, model_name, img_width, img_height, num_classes)


if __name__ == "__main__":
    main()
