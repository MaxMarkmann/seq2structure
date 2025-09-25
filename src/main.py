import argparse
from dataloader import load_cb513
from preprocessing import preprocess_dataset
from train import (
    train_baseline, train_random_forest, train_mlp,
    train_baseline_groupkfold  # neuer Import
)
import analyze_results


def run_training(group_split: bool = False):
    df = load_cb513()
    print("Loaded dataset:", df.shape)

    X, y, groups = preprocess_dataset(df, window_size=17)
    print("Processed shapes:", X.shape, y.shape)

    if group_split:
        # Protein-level split (GroupKFold)
        train_baseline_groupkfold(X, y, groups)
        # später kannst du hier auch RandomForest/MLP mit GroupKFold ergänzen
    else:
        # Klassischer Residue-level split
        train_baseline(X, y)
        train_random_forest(X, y)
        train_mlp(X, y, epochs=5, batch_size=512)


def run_analysis():
    analyze_results.main()


def main():
    parser = argparse.ArgumentParser(
        description="Protein Secondary Structure Prediction Project"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "analyze", "all"],
        required=True,
        help="Choose whether to train models, analyze results, or both",
    )
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="Use GroupKFold (protein-level split) instead of residue-level split",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training(group_split=args.group_split)
    elif args.mode == "analyze":
        run_analysis()
    elif args.mode == "all":
        run_training(group_split=args.group_split)
        run_analysis()


if __name__ == "__main__":
    main()
