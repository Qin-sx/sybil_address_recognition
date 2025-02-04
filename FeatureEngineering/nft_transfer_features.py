import os
import pandas as pd

def calculate_nft_transfer_counts(data_df, filter_df, column_name, final_name):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_nft_zero_address_counts(data_df, filter_df, column_name, zero_column_name, final_name):
    zero_address_counts = data_df[data_df[zero_column_name] == '0x0000000000000000000000000000000000000000'][column_name].value_counts().reset_index()
    zero_address_counts.columns = ['ADDRESS', final_name]
    result_df = zero_address_counts[zero_address_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def make_nft_transfer_features(nft_transfer_raw_file, output_path, filter_df):
    data_df = pd.read_parquet(nft_transfer_raw_file)

    nft_from_counts_df = calculate_nft_transfer_counts(data_df, filter_df, 'NFT_FROM_ADDRESS', 'NFT_FROM_ADDRESS_COUNT')
    nft_to_counts_df = calculate_nft_transfer_counts(data_df, filter_df, 'NFT_TO_ADDRESS', 'NFT_TO_ADDRESS_COUNT')
    nft_from_zero_to_counts_df = calculate_nft_zero_address_counts(data_df, filter_df, 'NFT_FROM_ADDRESS', 'NFT_TO_ADDRESS', 'NFT_FROM_ZERO_TO_ADDRESS_COUNT')
    nft_to_zero_from_counts_df = calculate_nft_zero_address_counts(data_df, filter_df, 'NFT_TO_ADDRESS', 'NFT_FROM_ADDRESS', 'NFT_TO_ZERO_FROM_ADDRESS_COUNT')

    final_df = filter_df.copy()
    dataframes = [
        nft_from_counts_df,
        nft_to_counts_df,
        nft_from_zero_to_counts_df,
        nft_to_zero_from_counts_df
    ]

    for df in dataframes:
        final_df = pd.merge(final_df, df, on='ADDRESS', how='left')
    return final_df