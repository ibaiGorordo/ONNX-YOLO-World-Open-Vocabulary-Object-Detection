from argparse import ArgumentParser
import numpy as np
import torch

from export import TextEmbedder


def main():
    parser = ArgumentParser()
    parser.add_argument("classes", type=str, nargs="+")
    parser.add_argument("--output", type=str, default="class_embeddings.npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize text embedder
    text_embedder = TextEmbedder(device=args.device)

    # Get text embeddings
    class_embeddings = text_embedder.embed_text(args.classes)

    # Convert to numpy array
    class_embeddings = class_embeddings.cpu().numpy()

    # Save class embeddings
    np.save(args.output, class_embeddings)


if __name__ == "__main__":
    main()
