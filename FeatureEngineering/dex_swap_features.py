import os
import pandas as pd
import numpy as np

def calculate_average_amount_in_usd(data_df, filter_df):
    data_df['AMOUNT_IN_USD'] = pd.to_numeric(data_df['AMOUNT_IN_USD'], errors='coerce').fillna(0)
    average_values = data_df.groupby('ORIGIN_FROM_ADDRESS')['AMOUNT_IN_USD'].mean()
    average_amount_in_usd_df = average_values.reset_index(name='DEX_SWAP_AVERAGE_AMOUNT_IN_USD')
    average_amount_in_usd_df.columns = ['ADDRESS', 'DEX_SWAP_AVERAGE_AMOUNT_IN_USD']
    average_amount_in_usd_df['DEX_SWAP_AVERAGE_AMOUNT_IN_USD'] = np.log1p(average_amount_in_usd_df['DEX_SWAP_AVERAGE_AMOUNT_IN_USD'])
    result_df = average_amount_in_usd_df[average_amount_in_usd_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_maximum_amount_in_usd(data_df, filter_df):
    data_df['AMOUNT_IN_USD'] = pd.to_numeric(data_df['AMOUNT_IN_USD'], errors='coerce').fillna(0)
    maximum_values = data_df.groupby('ORIGIN_FROM_ADDRESS')['AMOUNT_IN_USD'].max()
    maximum_amount_in_usd_df = maximum_values.reset_index(name='DEX_SWAP_MAXIMUM_AMOUNT_IN_USD')
    maximum_amount_in_usd_df.columns = ['ADDRESS', 'DEX_SWAP_MAXIMUM_AMOUNT_IN_USD']
    maximum_amount_in_usd_df['DEX_SWAP_MAXIMUM_AMOUNT_IN_USD'] = np.log1p(maximum_amount_in_usd_df['DEX_SWAP_MAXIMUM_AMOUNT_IN_USD'])
    result_df = maximum_amount_in_usd_df[maximum_amount_in_usd_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_average_amount_out_usd(data_df, filter_df):
    data_df['AMOUNT_OUT_USD'] = pd.to_numeric(data_df['AMOUNT_OUT_USD'], errors='coerce').fillna(0)
    average_values = data_df.groupby('ORIGIN_TO_ADDRESS')['AMOUNT_OUT_USD'].mean()
    average_amount_out_usd_df = average_values.reset_index(name='DEX_SWAP_AVERAGE_AMOUNT_OUT_USD')
    average_amount_out_usd_df.columns = ['ADDRESS', 'DEX_SWAP_AVERAGE_AMOUNT_OUT_USD']
    average_amount_out_usd_df['DEX_SWAP_AVERAGE_AMOUNT_OUT_USD'] = np.log1p(average_amount_out_usd_df['DEX_SWAP_AVERAGE_AMOUNT_OUT_USD'])
    result_df = average_amount_out_usd_df[average_amount_out_usd_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_maximum_amount_out_usd(data_df, filter_df):
    data_df['AMOUNT_OUT_USD'] = pd.to_numeric(data_df['AMOUNT_OUT_USD'], errors='coerce').fillna(0)
    maximum_values = data_df.groupby('ORIGIN_TO_ADDRESS')['AMOUNT_OUT_USD'].max()
    maximum_amount_out_usd_df = maximum_values.reset_index(name='DEX_SWAP_MAXIMUM_AMOUNT_OUT_USD')
    maximum_amount_out_usd_df.columns = ['ADDRESS', 'DEX_SWAP_MAXIMUM_AMOUNT_OUT_USD']
    maximum_amount_out_usd_df['DEX_SWAP_MAXIMUM_AMOUNT_OUT_USD'] = np.log1p(maximum_amount_out_usd_df['DEX_SWAP_MAXIMUM_AMOUNT_OUT_USD'])
    result_df = maximum_amount_out_usd_df[maximum_amount_out_usd_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_all_action_num(data_df, filter_df, columns=['ORIGIN_FROM_ADDRESS', 'ORIGIN_TO_ADDRESS', 'SENDER', 'TX_TO', 'TOKEN_IN', 'TOKEN_OUT'], final_name='DEX_SWAP_ALL_ACTIONS_COUNT'):
    all_addresses = pd.concat([data_df['ORIGIN_FROM_ADDRESS'], data_df['ORIGIN_TO_ADDRESS'], data_df['SENDER'], data_df['TX_TO'], data_df['TOKEN_IN'], data_df['TOKEN_OUT']])
    address_counts = all_addresses.value_counts().reset_index()
    address_counts.columns = ['ADDRESS', final_name]
    result_df = address_counts[address_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_each_deposit_action_num(data_df, filter_df, column_name="ORIGIN_TO_ADDRESS", final_name="DEX_SWAP_DEPOSIT_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_each_withdrawal_action_num(data_df, filter_df, column_name="ORIGIN_FROM_ADDRESS", final_name="DEX_SWAP_WITHDRAWAL_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_sender_action_num(data_df, filter_df, column_name="SENDER", final_name="DEX_SWAP_SENDER_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_tx_to_action_num(data_df, filter_df, column_name="TX_TO", final_name="DEX_SWAP_TX_TO_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_token_in_action_num(data_df, filter_df, column_name="TOKEN_IN", final_name="DEX_SWAP_TOKEN_IN_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_token_out_action_num(data_df, filter_df, column_name="TOKEN_OUT", final_name="DEX_SWAP_TOKEN_OUT_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df



def make_dex_swap_features(dex_swap_raw_file, output_path, filter_df):
    data_df = pd.read_parquet(dex_swap_raw_file)

    # average_amount_in_usd_df = calculate_average_amount_in_usd(data_df, filter_df)
    # print("average_amount_in_usd_df done")
    # maximum_amount_in_usd_df = calculate_maximum_amount_in_usd(data_df, filter_df)
    # print("maximum_amount_in_usd_df done")
    # average_amount_out_usd_df = calculate_average_amount_out_usd(data_df, filter_df)
    # print("average_amount_out_usd_df done")
    # maximum_amount_out_usd_df = calculate_maximum_amount_out_usd(data_df, filter_df)
    # print("maximum_amount_out_usd_df done")
    
    all_action_num = calculate_all_action_num(data_df, filter_df)
    each_deposit_action_num = calculate_each_deposit_action_num(data_df, filter_df)
    each_withdrawal_action_num = calculate_each_withdrawal_action_num(data_df, filter_df)
    print("all_action_num done")

    sender_action_num = calculate_sender_action_num(data_df, filter_df)
    tx_to_action_num  = calculate_tx_to_action_num(data_df, filter_df)
    # token_in_action_num = calculate_token_in_action_num(data_df, filter_df)
    # token_out_action_num = calculate_token_out_action_num(data_df, filter_df)
    print("all_action_num done")

    final_df = filter_df.copy()
    dataframes = [
        # average_amount_in_usd_df,
        # maximum_amount_in_usd_df,
        # average_amount_out_usd_df,
        # maximum_amount_out_usd_df,
        all_action_num,
        each_deposit_action_num,
        each_withdrawal_action_num,
        sender_action_num,
        tx_to_action_num
        # token_in_action_num,
        # token_out_action_num
    ]

    for df in dataframes:
        final_df = pd.merge(final_df, df, on='ADDRESS', how='left')
    return final_df