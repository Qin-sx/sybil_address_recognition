import os
import pandas as pd
import FeatureEngineering as fe

def get_all_candidates(file_path):
    train_datafile = os.path.join(file_path, "train_dataset.parquet")
    test_datafile = os.path.join(file_path, "test_dataset.parquet")
    train_df = pd.read_parquet(train_datafile)
    test_df = pd.read_parquet(test_datafile)
    train_addresses = train_df['ADDRESS']
    test_addresses = test_df['ADDRESS']
    all_addresses = pd.concat([train_addresses, test_addresses], ignore_index=True)
    all_addresses_df = pd.DataFrame(all_addresses, columns=['ADDRESS'])
    return all_addresses_df

def merge_features(transactions_feature_file, token_transfers_feature_file):
    # Read transactions_feature.parquet file
    transactions_feature_df = pd.read_parquet(transactions_feature_file)
    
    # Read token_transfers_feature.parquet file
    token_transfers_feature_df = pd.read_parquet(token_transfers_feature_file)
    
    # Merge the two dataframes based on the ADDRESS column
    merged_df = pd.merge(transactions_feature_df, token_transfers_feature_df, on='ADDRESS', how='left')
    
    # Save the merged dataframe back to the original transactions_feature.parquet file
    merged_df.to_parquet(transactions_feature_file)

if __name__ == "__main__":
    file_path = "./data/raw_data/"
    output_path = "./data/features/"
    transaction_raw_file = os.path.join(file_path, "transactions.parquet")
    token_transfer_raw_file = os.path.join(file_path, "token_transfers.parquet")
    dex_swaps_raw_file = os.path.join(file_path, "dex_swaps.parquet")
    address_df = get_all_candidates(file_path)
    
    # Generate transaction features
    transaction_feature_df = fe.make_transaction_features(transaction_raw_file, output_path, address_df)
    transaction_feature_df.fillna(0, inplace=True)
    transaction_output_file_name = os.path.join(output_path, "transactions_feature.parquet")
    transaction_feature_df.to_parquet(transaction_output_file_name)


    # Generate token transfer features
    token_transfer_feature_df = fe.make_token_transfer_features(token_transfer_raw_file, output_path, address_df)
    token_transfer_feature_df.fillna(0, inplace=True)
    token_transfer_output_file_name = os.path.join(output_path, "token_transfers_feature.parquet")
    token_transfer_feature_df.to_parquet(token_transfer_output_file_name)

    # Merge token transfer features into the transaction features file
    merge_features(transaction_output_file_name, token_transfer_output_file_name)

    # Generate DEX swap features
    make_dex_feature_df = fe.make_dex_swap_features(dex_swaps_raw_file, output_path, address_df)
    make_dex_feature_df.fillna(0, inplace=True)
    make_dex_output_file_name = os.path.join(output_path, "dex_swaps_feature.parquet")
    make_dex_feature_df.to_parquet(make_dex_output_file_name)

    # Merge DEX swap features into the transaction features file
    merge_features(transaction_output_file_name, make_dex_output_file_name)
    