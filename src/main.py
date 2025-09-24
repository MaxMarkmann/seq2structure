from dataloader import load_cb513
from preprocessing import preprocess_dataset
from train import train_baseline

def main():
    # 1. Load dataset
    df = load_cb513()
    print("Loaded dataset:", df.shape)

    # 2. Preprocess
    X, y = preprocess_dataset(df, window_size=17)
    print("Processed shapes:", X.shape, y.shape)

    # 3. Train baseline model
    model, score = train_baseline(X, y)
    print("Baseline accuracy:", score)

if __name__ == "__main__":
    main()
