import os
import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


def make_data_set(raw_data_path, feature_df):
    data_file = os.path.join(raw_data_path, "train_dataset.parquet")
    data_df = pd.read_parquet(data_file)
    train_df, val_df = train_test_split(data_df, test_size=0.075, random_state=42)
    train_df = pd.merge(train_df, feature_df, on='ADDRESS', how='left')
    val_df = pd.merge(val_df, feature_df, on='ADDRESS', how='left')
    train_df = shuffle(train_df, random_state=42)
    train_x = train_df.drop(['ADDRESS', 'LABEL'], axis=1)
    train_y = train_df['LABEL'].astype(int)
    val_x = val_df.drop(['ADDRESS', 'LABEL'], axis=1)
    val_y = val_df['LABEL'].astype(int)

    return train_x, train_y, val_x, val_y


def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    res = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return res

def select_important_features(model, train_x, threshold=0.01):
    feature_importances = model.get_feature_importance()
    print("model.get_feature_importance()", model.get_feature_importance())
    important_features = train_x.columns[feature_importances > threshold]
    return important_features

def train_catboost(train_x, train_y, val_x, val_y, save_model_path):
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=8,
        eval_metric='Logloss',
        random_seed=42,
        verbose=100,
        # l2_leaf_reg = 1,
        bagging_temperature = 0
    )
    
    # Hyperparameter tuning
    param_grid = {
        # 'iterations': [500, 1000]
        # 'learning_rate': [0.05, 0.1],
        # 'depth': [4, 6, 8, 10],
        # 'l2_leaf_reg': [1, 3, 5, 7, 9],
        # 'l2_leaf_reg': [1, 3],
        # 'bagging_temperature': [0, 0.5, 1]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='f1_macro', verbose=1, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    
    best_model = grid_search.best_estimator_
    
    # Save the best hyperparameters
    best_params = grid_search.best_params_
    with open(os.path.join(save_model_path, "best_params.json"), "w") as f:
        json.dump(best_params, f)
    
    # Retrain the model with the best parameters and evaluate on the validation set
    best_model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=50)
    
    # Get important features
    important_features = select_important_features(best_model, train_x)
    
    # Retrain the model using only important features
    train_x_important = train_x[important_features]
    val_x_important = val_x[important_features]
    
    best_model.fit(train_x_important, train_y, eval_set=(val_x_important, val_y), early_stopping_rounds=50)
    
    # Save the best model
    best_model.save_model(os.path.join(save_model_path, "best_model.cbm"))
    
    val_pred = best_model.predict(val_x_important)
    val_metrics = get_metrics(val_y, val_pred)
    for metric_name, value in val_metrics.items():
        print(f"val_metrics: {metric_name}: {value}")
    with open(os.path.join(save_model_path, "val_performance.json"), "w") as f:
        for metric_name, value in val_metrics.items():
            f.write(f"val_metrics: {metric_name}: {value}" + "\n")

def main():
    # Example data paths
    raw_data_path = "./data/raw_data/"
    feature_path = "./data/features/transactions_feature_partner.parquet"
    save_model_path = "./saved_model/"
    
    # Load data
    feature_df = pd.read_parquet(feature_path)
    train_x, train_y, val_x, val_y = make_data_set(raw_data_path, feature_df)
    
    # Train the model
    train_catboost(train_x, train_y, val_x, val_y, save_model_path)

if __name__ == "__main__":
    main()