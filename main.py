import argparse
from src.data_loader import load_raw_data, save_processed_data, load_processed_data
from src.labeling import apply_weak_supervision
from src.models import prepare_features, run_benchmark_and_save
from src.analyzer import analyze_top_features, plot_confusion_matrix, analyze_predictions_and_errors

def parse_args():
    parser = argparse.ArgumentParser(description="Guardian Environment News Classifier")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip raw data loading and weak supervision.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.skip_preprocessing:
        print("Loading cached data...")
        df_labeled = load_processed_data()
    else:
        print("Processing raw data with weak supervision...")
        df_raw = load_raw_data()
        df_labeled = apply_weak_supervision(df_raw)
        save_processed_data(df_labeled)

    tfidf, X_train_vec, X_test_vec, y_train, y_test, X_test_raw, y_test_raw = prepare_features(df_labeled)
    results, best_model = run_benchmark_and_save(X_train_vec, X_test_vec, y_train, y_test)

    analyze_top_features(best_model, tfidf)
    y_pred = best_model.predict(X_test_vec)
    plot_confusion_matrix(y_test, y_pred, best_model.classes_)
    analyze_predictions_and_errors(best_model, tfidf, X_test_raw, y_test_raw)

if __name__ == "__main__":
    main()