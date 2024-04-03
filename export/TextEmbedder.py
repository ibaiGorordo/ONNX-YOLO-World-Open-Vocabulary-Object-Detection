import clip
import torch


class TextEmbedder:
    def __init__(self, device='cpu'):
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

    def embed_text(self, text):
        if not isinstance(text, list):
            text = [text]

        text_token = clip.tokenize(text).to(self.device)
        txt_feats = [self.clip_model.encode_text(token).detach() for token in text_token.split(1)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        return txt_feats

