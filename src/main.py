import argparse
from dataloader import load_cb513
from preprocessing import preprocess_dataset
from train import train_baseline, train_random_forest
import analyze_results


def run_training():
    df = load_cb513()
    print("Loaded dataset:", df.shape)

    X, y = preprocess_dataset(df, window_size=17)
    print("Processed shapes:", X.shape, y.shape)

    # Train models
    train_baseline(X, y)
    train_random_forest(X, y)


def run_analysis():
    analyze_results.main()


def main():
    parser = argparse.ArgumentParser(
        description="Protein Secondary Structure Prediction Project"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "analyze"],
        required=True,
        help="Choose whether to train models or analyze results",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "analyze":
        run_analysis()


if __name__ == "__main__":
    main()
