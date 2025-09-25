import argparse
from dataloader import load_cb513
from preprocessing import preprocess_dataset
from train import train_baseline, train_random_forest, train_mlp, train_with_embeddings
import analyze_results
from embed_sequences import run_embedding   # 👈 Import der Embedding-Funktion
import numpy as np
from embedding_utils import load_embeddings, summarize_embeddings   # 👈 summarize importiert


def run_training(encoding: str, group_split: bool = False, subset: int = None):
    df = load_cb513(n_proteins=subset)
    print("Loaded dataset:", df.shape)

    X, y = preprocess_dataset(df, encoding=encoding, window_size=17)
    print(f"Processed shapes: X={X.shape}, y={y.shape} (encoding={encoding})")

    # Train models
    train_baseline(X, y)
    train_random_forest(X, y)
    train_mlp(X, y, epochs=5, batch_size=512)


def run_analysis():
    analyze_results.main()


def run_training_with_embeddings(embed_file: str, model_type: str):
    X, y, ids = load_embeddings(embed_file)
    print(f"Loaded embeddings: {X.shape}, labels={len(y)}")
    train_with_embeddings(X, y, model_type=model_type)


def run_summarize(embed_file: str):
    summarize_embeddings(embed_file)


def main():
    parser = argparse.ArgumentParser(
        description="Protein Secondary Structure Prediction Project"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "analyze", "all", "embed", "train_embed", "summarize"],
        required=True,
        help="Choose whether to train models, analyze results, embed ProtBERT features, "
             "train on embeddings, summarize embeddings, or all",
    )
    parser.add_argument(
        "--encoding",
        choices=["onehot", "protbert", "esm2"],
        default="onehot",
        help="Feature encoding method for preprocessing",
    )
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="Use GroupKFold split instead of random split",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, only use first N proteins (useful for fast embedding tests)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="protbert_embeddings.npz",
        help="Output file name for embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embeddings: cpu or cuda",
    )
    parser.add_argument(
        "--embed-file",
        type=str,
        default="protbert_embeddings.npz",
        help="Path to embeddings file for --mode train_embed or summarize",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "rf", "mlp"],
        help="Model type when training on embeddings",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training(encoding=args.encoding, group_split=args.group_split, subset=args.subset)
    elif args.mode == "analyze":
        run_analysis()
    elif args.mode == "all":
        run_training(encoding=args.encoding, group_split=args.group_split, subset=args.subset)
        run_analysis()
    elif args.mode == "embed":
        run_embedding(n_proteins=args.subset, out_file=args.out, device=args.device)
    elif args.mode == "train_embed":
        run_training_with_embeddings(embed_file=args.embed_file, model_type=args.model)
    elif args.mode == "summarize":
        run_summarize(embed_file=args.embed_file)


if __name__ == "__main__":
    main()
