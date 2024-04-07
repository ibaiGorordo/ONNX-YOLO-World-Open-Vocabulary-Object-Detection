import clip
import torch


class TextEmbedder:
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        self.device = device
        self.clip_model, _ = clip.load(model_name, device=self.device)

    def __call__(self, text):
        return self.embed_text(text)

    def embed_text(self, text):
        if not isinstance(text, list):
            text = [text]

        text_token = clip.tokenize(text).to(self.device)
        txt_feats = [self.clip_model.encode_text(token).detach() for token in text_token.split(1)]
        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        txt_feats = txt_feats.unsqueeze(0)
        return txt_feats


if __name__ == "__main__":
    import numpy as np

    text_embedder = TextEmbedder()
    text = ["cat", "dog"]
    txt_feats = text_embedder(text).cpu().numpy()

    np.savez("../data/text_embeddings.npz", txt_feats=txt_feats, text=np.array(text))
