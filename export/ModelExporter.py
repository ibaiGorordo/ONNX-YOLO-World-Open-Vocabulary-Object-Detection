from copy import deepcopy
import torch
from ultralytics import YOLOWorld


class ModelExporter(torch.nn.Module):
    def __init__(self, yoloModel, device='cpu'):
        super(ModelExporter, self).__init__()
        model = yoloModel.model
        model = deepcopy(model).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        self.model = model
        self.device = device

    def forward(self, x, txt_feats):
        return self.model.predict(x, txt_feats=txt_feats)

    def export(self, output_dir, model_name, img_width, img_height, num_classes):

        x = torch.randn(1, 3, img_width, img_height, requires_grad=False).to(self.device)
        txt_feats = torch.randn(1, num_classes, 512, requires_grad=False).to(self.device)

        print(x.shape, txt_feats.shape)

        # Export model
        output_path = f"{output_dir}/{model_name.replace('.pt', '.onnx')}"
        with torch.no_grad():
            torch.onnx.export(self,
                              (x, txt_feats),
                              output_path,
                              do_constant_folding=True,
                              opset_version=12,
                              input_names=["images", "txt_feats"],
                              output_names=["output"])

if __name__ == "__main__":
    # Initialize model
    yoloModel = YOLOWorld("yolov8l-worldv2.pt")
    yoloModel.set_classes([""] * 1)

    # Initialize model exporter
    export_model = ModelExporter(yoloModel, "cuda")

    # Export model
    x = torch.randn(1, 3, 640, 480, requires_grad=False).to("cuda")
    txt_feats = torch.randn(1, 1, 512, requires_grad=False).to("cuda")

    y = export_model(x, txt_feats)
