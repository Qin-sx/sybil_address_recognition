import os
import pandas as pd
import numpy as np

def calculate_all_transfer_actions(data_df, filter_df, columns=['FROM_ADDRESS', 'TO_ADDRESS', 'ORIGIN_FROM_ADDRESS', 'ORIGIN_TO_ADDRESS'], final_name='TOKEN_TRANSFER_ALL_ACTIONS_COUNT'):
    all_addresses = pd.concat([data_df['FROM_ADDRESS'], data_df['TO_ADDRESS'], data_df['ORIGIN_FROM_ADDRESS'], data_df['ORIGIN_TO_ADDRESS']])
    address_counts = all_addresses.value_counts().reset_index()
    address_counts.columns = ['ADDRESS', final_name]
    result_df = address_counts[address_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_each_deposit_action_num(data_df, filter_df, column_name="TO_ADDRESS", final_name="TOKEN_TRANSFER_DEPOSIT_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_each_withdrawal_action_num(data_df, filter_df, column_name="FROM_ADDRESS", final_name="TOKEN_TRANSFER_WITHDRAWAL_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_unique_addresses(data_df, filter_df):
    from_to_unique = data_df.groupby('FROM_ADDRESS')['TO_ADDRESS'].nunique().reset_index()
    from_to_unique.columns = ['ADDRESS', 'TOKEN_TRANSFER_UNIQUE_TO_INTERACTIONS']
    to_from_unique = data_df.groupby('TO_ADDRESS')['FROM_ADDRESS'].nunique().reset_index()
    to_from_unique.columns = ['ADDRESS', 'TOKEN_TRANSFER_UNIQUE_FROM_INTERACTIONS']
    combined_interactions = pd.merge(from_to_unique, to_from_unique, on='ADDRESS', how='outer').fillna(0)
    result_df = combined_interactions[combined_interactions['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_deposit_average_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    average_values = data_df.groupby('TO_ADDRESS')['RAW_AMOUNT'].mean()
    deposit_average_values_df = average_values.reset_index(name='TOKEN_TRANSFER_DEPOSIT_AVERAGE_VALUE')
    deposit_average_values_df.columns = ['ADDRESS', 'TOKEN_TRANSFER_DEPOSIT_AVERAGE_VALUE']
    deposit_average_values_df['TOKEN_TRANSFER_DEPOSIT_AVERAGE_VALUE'] = np.log1p(deposit_average_values_df['TOKEN_TRANSFER_DEPOSIT_AVERAGE_VALUE'])
    result_df = deposit_average_values_df[deposit_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_deposit_maximum_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    maximum_values = data_df.groupby('TO_ADDRESS')['RAW_AMOUNT'].max()
    deposit_maximum_values_df = maximum_values.reset_index(name='TOKEN_TRANSFER_DEPOSIT_MAXIMUM_VALUE')
    deposit_maximum_values_df.columns = ['ADDRESS', 'TOKEN_TRANSFER_DEPOSIT_MAXIMUM_VALUE']
    deposit_maximum_values_df['TOKEN_TRANSFER_DEPOSIT_MAXIMUM_VALUE'] = np.log1p(deposit_maximum_values_df['TOKEN_TRANSFER_DEPOSIT_MAXIMUM_VALUE'])
    result_df = deposit_maximum_values_df[deposit_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_withdrawal_average_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    average_values = data_df.groupby('FROM_ADDRESS')['RAW_AMOUNT'].mean()
    withdrawal_average_values_df = average_values.reset_index(name='TOKEN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE')
    withdrawal_average_values_df.columns = ['ADDRESS', 'TOKEN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE']
    withdrawal_average_values_df['TOKEN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE'] = np.log1p(withdrawal_average_values_df['TOKEN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE'])
    result_df = withdrawal_average_values_df[withdrawal_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_withdrawal_maximum_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    maximum_values = data_df.groupby('FROM_ADDRESS')['RAW_AMOUNT'].max()
    withdrawal_maximum_values_df = maximum_values.reset_index(name='TOKEN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE')
    withdrawal_maximum_values_df.columns = ['ADDRESS', 'TOKEN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE']
    withdrawal_maximum_values_df['TOKEN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE'] = np.log1p(withdrawal_maximum_values_df['TOKEN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE'])
    result_df = withdrawal_maximum_values_df[withdrawal_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_origin_each_deposit_action_num(data_df, filter_df, column_name="ORIGIN_TO_ADDRESS", final_name="TOKEN_ORIGIN_TRANSFER_DEPOSIT_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_origin_each_withdrawal_action_num(data_df, filter_df, column_name="ORIGIN_FROM_ADDRESS", final_name="TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_origin_deposit_average_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    average_values = data_df.groupby('ORIGIN_TO_ADDRESS')['RAW_AMOUNT'].mean()
    deposit_average_values_df = average_values.reset_index(name='TOKEN_ORIGIN_TRANSFER_DEPOSIT_AVERAGE_VALUE')
    deposit_average_values_df.columns = ['ADDRESS', 'TOKEN_ORIGIN_TRANSFER_DEPOSIT_AVERAGE_VALUE']
    deposit_average_values_df['TOKEN_ORIGIN_TRANSFER_DEPOSIT_AVERAGE_VALUE'] = np.log1p(deposit_average_values_df['TOKEN_ORIGIN_TRANSFER_DEPOSIT_AVERAGE_VALUE'])
    result_df = deposit_average_values_df[deposit_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_origin_deposit_maximum_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    maximum_values = data_df.groupby('ORIGIN_TO_ADDRESS')['RAW_AMOUNT'].max()
    deposit_maximum_values_df = maximum_values.reset_index(name='TOKEN_ORIGIN_TRANSFER_DEPOSIT_MAXIMUM_VALUE')
    deposit_maximum_values_df.columns = ['ADDRESS', 'TOKEN_ORIGIN_TRANSFER_DEPOSIT_MAXIMUM_VALUE']
    deposit_maximum_values_df['TOKEN_ORIGIN_TRANSFER_DEPOSIT_MAXIMUM_VALUE'] = np.log1p(deposit_maximum_values_df['TOKEN_ORIGIN_TRANSFER_DEPOSIT_MAXIMUM_VALUE'])
    result_df = deposit_maximum_values_df[deposit_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_origin_withdrawal_average_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    average_values = data_df.groupby('ORIGIN_FROM_ADDRESS')['RAW_AMOUNT'].mean()
    withdrawal_average_values_df = average_values.reset_index(name='TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE')
    withdrawal_average_values_df.columns = ['ADDRESS', 'TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE']
    withdrawal_average_values_df['TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE'] = np.log1p(withdrawal_average_values_df['TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_AVERAGE_VALUE'])
    result_df = withdrawal_average_values_df[withdrawal_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def calculate_origin_withdrawal_maximum_value(data_df, filter_df):
    data_df['RAW_AMOUNT'] = pd.to_numeric(data_df['RAW_AMOUNT'], errors='coerce')
    maximum_values = data_df.groupby('ORIGIN_FROM_ADDRESS')['RAW_AMOUNT'].max()
    withdrawal_maximum_values_df = maximum_values.reset_index(name='TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE')
    withdrawal_maximum_values_df.columns = ['ADDRESS', 'TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE']
    withdrawal_maximum_values_df['TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE'] = np.log1p(withdrawal_maximum_values_df['TOKEN_ORIGIN_TRANSFER_WITHDRAWAL_MAXIMUM_VALUE'])
    result_df = withdrawal_maximum_values_df[withdrawal_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df



def make_token_transfer_features(token_transfer_raw_file, output_path, filter_df):
    data_df = pd.read_parquet(token_transfer_raw_file)

    all_action_counts_df = calculate_all_transfer_actions(data_df, filter_df)
    print("all_action_counts_df done")
    deposit_action_num_df = calculate_each_deposit_action_num(data_df, filter_df)
    print("deposit_action_num_df done")
    withdrawal_action_num_df = calculate_each_withdrawal_action_num(data_df, filter_df)
    print("withdrawal_action_num_df done")
    unique_addresses_df = calculate_unique_addresses(data_df, filter_df)
    print("unique_addresses_df done")
    deposit_average_value_df = calculate_deposit_average_value(data_df, filter_df)
    print("deposit_average_value_df done")
    deposit_maximum_value_df = calculate_deposit_maximum_value(data_df, filter_df)
    print("deposit_maximum_value_df done")
    withdrawal_average_value_df = calculate_withdrawal_average_value(data_df, filter_df)
    print("withdrawal_average_value_df done")
    withdrawal_maximum_value_df = calculate_withdrawal_maximum_value(data_df, filter_df)
    print("withdrawal_maximum_value_df done")
    
    origin_each_deposit_action_num = calculate_origin_each_deposit_action_num(data_df, filter_df)
    origin_each_withdrawal_action_num = calculate_origin_each_withdrawal_action_num(data_df, filter_df)
    origin_deposit_average_value = calculate_origin_deposit_average_value(data_df, filter_df)
    origin_deposit_maximum_value = calculate_origin_deposit_maximum_value(data_df, filter_df)
    origin_withdrawal_average_value = calculate_origin_withdrawal_average_value(data_df, filter_df)
    origin_withdrawal_maximum_value = calculate_origin_withdrawal_maximum_value(data_df, filter_df)


    final_df = filter_df.copy()
    dataframes = [
        all_action_counts_df,
        deposit_action_num_df,
        withdrawal_action_num_df,
        unique_addresses_df,
        deposit_average_value_df,
        deposit_maximum_value_df,
        withdrawal_average_value_df,
        withdrawal_maximum_value_df,

        origin_each_deposit_action_num,
        origin_each_withdrawal_action_num,
        origin_deposit_average_value,
        origin_deposit_maximum_value,
        origin_withdrawal_average_value,
        origin_withdrawal_maximum_value
    ]

    for df in dataframes:
        final_df = pd.merge(final_df, df, on='ADDRESS', how='left')
    return final_df