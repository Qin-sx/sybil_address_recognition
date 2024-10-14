import os
import pandas as pd
import FeatureEngineering as fe

def get_all_candidates(file_path, file_path_2):
    train_datafile = os.path.join(file_path, "train_dataset.parquet")
    test_datafile = os.path.join(file_path, "test_dataset.parquet")
    partner_datafile = os.path.join(file_path_2, "partners.parquet")
    train_df = pd.read_parquet(train_datafile)
    test_df = pd.read_parquet(test_datafile)
    partner_df = pd.read_parquet(partner_datafile)
    train_addresses = train_df['ADDRESS']
    test_addresses = test_df['ADDRESS']
    # partner_df.rename(columns={'PARTNER': 'ADDRESS'}, inplace=True) 
    partner_addresses = partner_df['ADDRESS']
    all_addresses = pd.concat([train_addresses, test_addresses, partner_addresses], ignore_index=True)
    all_addresses_df = pd.DataFrame(all_addresses, columns=['ADDRESS'])
    return all_addresses_df

def merge_features(transactions_feature_file, token_transfers_feature_file):
    transactions_feature_df = pd.read_parquet(transactions_feature_file)
    
    token_transfers_feature_df = pd.read_parquet(token_transfers_feature_file)
    
    merged_df = pd.merge(transactions_feature_df, token_transfers_feature_df, on='ADDRESS', how='left')
    
    merged_df.to_parquet(transactions_feature_file)

if __name__ == "__main__":
    file_path = "./data/raw_data/"
    output_path = "./data/features/"
    transaction_raw_file = os.path.join(file_path, "transactions.parquet")
    token_transfer_raw_file = os.path.join(file_path, "token_transfers.parquet")
    dex_swaps_raw_file = os.path.join(file_path, "dex_swaps.parquet")
    address_df = get_all_candidates(file_path, output_path)

    transaction_feature_df = fe.make_transaction_features(transaction_raw_file, output_path, address_df)
    transaction_feature_df.fillna(0, inplace=True)
    transaction_output_file_name = os.path.join(output_path, "transactions_feature_partner.parquet")
    transaction_feature_df.to_parquet(transaction_output_file_name)

    token_transfer_feature_df = fe.make_token_transfer_features(token_transfer_raw_file, output_path, address_df)
    token_transfer_feature_df.fillna(0, inplace=True)
    token_transfer_output_file_name = os.path.join(output_path, "token_transfers_feature.parquet")
    token_transfer_feature_df.to_parquet(token_transfer_output_file_name)

    merge_features(transaction_output_file_name, token_transfer_output_file_name)
