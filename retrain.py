from recommender_service import get_training_data, train_ranking_model

if __name__ == "__main__":
    df = get_training_data(limit=10000)
    if df.empty:
        print("No training data yet.")
    else:
        train_ranking_model(df)
        print("Model retrained and saved.")

