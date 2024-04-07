from copy import deepcopy
import torch


class ModelExporter(torch.nn.Module):
    def __init__(self, yoloModel, device='cpu'):
        super(ModelExporter, self).__init__()
        model = deepcopy(yoloModel).to(device)
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
        onnx_name = model_name.replace('.pth', '.onnx').replace('.pt', '.onnx')
        output_path = f"{output_dir}/{onnx_name}"
        with torch.no_grad():
            torch.onnx.export(self,
                              (x, txt_feats),
                              output_path,
                              do_constant_folding=True,
                              opset_version=17,
                              input_names=["images", "txt_feats"],
                              output_names=["output"])


if __name__ == "__main__":
    from ultralytics import YOLOWorld

    # Initialize model
    yoloModel = YOLOWorld("../yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth")
    yoloModel.set_classes([""] * 1)

    # Initialize model exporter
    export_model = ModelExporter(yoloModel, "cuda")

    # Export model
    x = torch.randn(1, 3, 640, 480, requires_grad=False).to("cuda")
    txt_feats = torch.randn(1, 1, 512, requires_grad=False).to("cuda")

    y = export_model(x, txt_feats)
