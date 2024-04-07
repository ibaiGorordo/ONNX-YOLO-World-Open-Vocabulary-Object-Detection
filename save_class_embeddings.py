from argparse import ArgumentParser
import numpy as np
import torch

from yoloworld import TextEmbedder


def main():
    parser = ArgumentParser()
    parser.add_argument("classes", type=str, nargs="+",
                        help='List of classes separated by space. You can use "-" for multiple words per class. Example: cat dog street-light')
    parser.add_argument("--output_dir", type=str, default="data", help="Output file to save class embeddings")
    parser.add_argument("--output_name", type=str, default="class_embeddings.npz", help="Output file name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize text embedder
    text_embedder = TextEmbedder(device=args.device)

    # Replace - or _ with space in class names
    classes = [class_name.replace("-", " ").replace("_", " ") for class_name in args.classes]

    # Get text embeddings
    class_embeddings = text_embedder(classes)

    # Convert to numpy array
    class_embeddings = class_embeddings.cpu().numpy().astype(np.float32)

    # Save class embeddings and classes
    output_path = args.output_dir + "/" + args.output_name
    np.savez(output_path, class_embeddings=class_embeddings, class_list=np.array(args.classes))


if __name__ == "__main__":
    main()
