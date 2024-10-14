import pandas as pd
import os
from catboost import CatBoostClassifier

def read_test_feature(raw_data_path, feature_df):
    data_file = os.path.join(raw_data_path, "test_dataset.parquet")
    data_df = pd.read_parquet(data_file)
    test_df = pd.merge(data_df, feature_df, on='ADDRESS', how='left')
    test_x = test_df.drop(['ADDRESS'], axis=1)

    return data_df, test_x

def main():
    raw_data_path = "./data/raw_data/"
    feature_path = "./data/features/transactions_feature.parquet"
    feature_df = pd.read_parquet(feature_path)
    test_df, test_x = read_test_feature(raw_data_path, feature_df)
    save_model_path = "./saved_model/"
    best_model_path = os.path.join(save_model_path, "best_model.cbm")
    
    # Load CatBoost model
    model = CatBoostClassifier()
    model.load_model(best_model_path)
    
    # Make predictions
    predictions = model.predict(test_x)
    
    # Save prediction results
    test_df['PRED'] = predictions.astype(int)
    test_df.to_csv("pred.csv", index=False)
    print("Predictions saved to pred.csv")

if __name__ == "__main__":
    main()