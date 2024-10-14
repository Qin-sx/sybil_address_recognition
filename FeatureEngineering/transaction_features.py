import os
import pandas as pd

def calculate_all_action_num(data_df, filter_df, columns=['FROM_ADDRESS', 'TO_ADDRESS'], final_name='TRANSACTION_ALL_ACTIONS_COUNT'):
    all_addresses = pd.concat([data_df['FROM_ADDRESS'], data_df['TO_ADDRESS']])
    address_counts = all_addresses.value_counts().reset_index()
    address_counts.columns = ['ADDRESS', final_name]
    result_df = address_counts[address_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_each_deposit_action_num(data_df, filter_df, column_name="TO_ADDRESS", final_name="TRANSACTION_DEPOSIT_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_each_withdrawal_action_num(data_df, filter_df, column_name="FROM_ADDRESS", final_name="TRANSACTION_WITHDRAWAL_ACTION_COUNT"):
    transfer_counts = data_df[column_name].value_counts().reset_index()
    transfer_counts.columns = ['ADDRESS', final_name]
    result_df = transfer_counts[transfer_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_unique_addresses(data_df, filter_df):
    from_to_unique = data_df.groupby('FROM_ADDRESS')['TO_ADDRESS'].nunique().reset_index()
    from_to_unique.columns = ['ADDRESS', 'TRANSACTION_UNIQUE_TO_INTERACTIONS']
    to_from_unique = data_df.groupby('TO_ADDRESS')['FROM_ADDRESS'].nunique().reset_index()
    to_from_unique.columns = ['ADDRESS', 'TRANSACTION_UNIQUE_FROM_INTERACTIONS']
    combined_interactions = pd.merge(from_to_unique, to_from_unique, on='ADDRESS', how='outer').fillna(0)
    result_df = combined_interactions[combined_interactions['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_max_nonce_per_contract(data_df, filter_df):
    melted_df = pd.melt(data_df, id_vars=['NONCE'], value_vars=['FROM_ADDRESS', 'TO_ADDRESS'], var_name='TYPE', value_name='ADDRESS')
    melted_df = melted_df.drop(columns='TYPE')
    max_nonce_per_address = melted_df.groupby('ADDRESS')['NONCE'].max().reset_index()
    result_df = max_nonce_per_address[max_nonce_per_address['ADDRESS'].isin(filter_df['ADDRESS'])]
    result_df = result_df.rename(columns={'NONCE': 'TRANSACTION_MAX_NONCE'})
    result_df['TRANSACTION_MAX_NONCE'] = result_df['TRANSACTION_MAX_NONCE'].astype(int)
    return result_df

def calculate_deposit_average_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    average_values = data_df.groupby('TO_ADDRESS')['VALUE'].mean()
    deposit_average_values_df = average_values.reset_index(name='TRANSACTION_DEPOSIT_AVERAGE_VALUE')
    deposit_average_values_df.columns = ['ADDRESS', 'TRANSACTION_DEPOSIT_AVERAGE_VALUE']
    result_df = deposit_average_values_df[deposit_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_deposit_maximum_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    maximum_values = data_df.groupby('TO_ADDRESS')['VALUE'].max()
    deposit_maximum_values_df = maximum_values.reset_index(name='TRANSACTION_DEPOSIT_MAXIMUM_VALUE')
    deposit_maximum_values_df.columns = ['ADDRESS', 'TRANSACTION_DEPOSIT_MAXIMUM_VALUE']
    result_df = deposit_maximum_values_df[deposit_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_withdrawal_average_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    average_values = data_df.groupby('FROM_ADDRESS')['VALUE'].mean()
    withdrawal_average_values_df = average_values.reset_index(name='TRANSACTION_WITHDRAWAL_AVERAGE_VALUE')
    withdrawal_average_values_df.columns = ['ADDRESS', 'TRANSACTION_WITHDRAWAL_AVERAGE_VALUE']
    result_df = withdrawal_average_values_df[withdrawal_average_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_withdrawal_maximum_value(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    maximum_values = data_df.groupby('FROM_ADDRESS')['VALUE'].max()
    withdrawal_maximum_values_df = maximum_values.reset_index(name='TRANSACTION_WITHDRAWAL_MAXIMUM_VALUE')
    withdrawal_maximum_values_df.columns = ['ADDRESS', 'TRANSACTION_WITHDRAWAL_MAXIMUM_VALUE']
    result_df = withdrawal_maximum_values_df[withdrawal_maximum_values_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_deposit_zero_value_count(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    zero_value_df = data_df[data_df['VALUE'] == 0]
    zero_value_counts = zero_value_df.groupby('TO_ADDRESS').size()
    deposit_zero_value_counts_df = zero_value_counts.reset_index(name='TRANSACTION_DEPOSIT_ZERO_VALUE_COUNT')
    deposit_zero_value_counts_df.columns = ['ADDRESS', 'TRANSACTION_DEPOSIT_ZERO_VALUE_COUNT']
    result_df = deposit_zero_value_counts_df[deposit_zero_value_counts_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_withdrawal_zero_value_count(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    zero_value_df = data_df[data_df['VALUE'] == 0]
    zero_value_counts = zero_value_df.groupby('FROM_ADDRESS').size()
    withdrawal_zero_value_counts_df = zero_value_counts.reset_index(name='TRANSACTION_WITHDRAWAL_ZERO_VALUE_COUNT')
    withdrawal_zero_value_counts_df.columns = ['ADDRESS', 'TRANSACTION_WITHDRAWAL_ZERO_VALUE_COUNT']
    result_df = withdrawal_zero_value_counts_df[withdrawal_zero_value_counts_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_tx_fee_features(data_df, filter_df):
    data_df['TX_FEE'] = pd.to_numeric(data_df['TX_FEE'], errors='coerce')
    avg_tx_fee = data_df.groupby('FROM_ADDRESS')['TX_FEE'].mean().reset_index(name='TRANSACTION_AVG_TX_FEE')
    max_tx_fee = data_df.groupby('FROM_ADDRESS')['TX_FEE'].max().reset_index(name='TRANSACTION_MAX_TX_FEE')
    result_df = pd.merge(avg_tx_fee, max_tx_fee, on='FROM_ADDRESS', how='outer')
    result_df.columns = ['ADDRESS', 'TRANSACTION_AVG_TX_FEE', 'TRANSACTION_MAX_TX_FEE']
    result_df = result_df[result_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_gas_features(data_df, filter_df):
    data_df['EFFECTIVE_GAS_PRICE'] = pd.to_numeric(data_df['EFFECTIVE_GAS_PRICE'], errors='coerce')
    data_df['GAS_LIMIT'] = pd.to_numeric(data_df['GAS_LIMIT'], errors='coerce')
    data_df['GAS_USED'] = pd.to_numeric(data_df['GAS_USED'], errors='coerce')
    data_df['CUMULATIVE_GAS_USED'] = pd.to_numeric(data_df['CUMULATIVE_GAS_USED'], errors='coerce')
    
    avg_gas_price = data_df.groupby('FROM_ADDRESS')['EFFECTIVE_GAS_PRICE'].mean().reset_index(name='TRANSACTION_AVG_GAS_PRICE')
    max_gas_price = data_df.groupby('FROM_ADDRESS')['EFFECTIVE_GAS_PRICE'].max().reset_index(name='TRANSACTION_MAX_GAS_PRICE')
    avg_gas_limit = data_df.groupby('FROM_ADDRESS')['GAS_LIMIT'].mean().reset_index(name='TRANSACTION_AVG_GAS_LIMIT')
    max_gas_limit = data_df.groupby('FROM_ADDRESS')['GAS_LIMIT'].max().reset_index(name='TRANSACTION_MAX_GAS_LIMIT')
    avg_gas_used = data_df.groupby('FROM_ADDRESS')['GAS_USED'].mean().reset_index(name='TRANSACTION_AVG_GAS_USED')
    max_gas_used = data_df.groupby('FROM_ADDRESS')['GAS_USED'].max().reset_index(name='TRANSACTION_MAX_GAS_USED')
    avg_cumulative_gas_used = data_df.groupby('FROM_ADDRESS')['CUMULATIVE_GAS_USED'].mean().reset_index(name='TRANSACTION_AVG_CUMULATIVE_GAS_USED')
    max_cumulative_gas_used = data_df.groupby('FROM_ADDRESS')['CUMULATIVE_GAS_USED'].max().reset_index(name='TRANSACTION_MAX_CUMULATIVE_GAS_USED')
    
    result_df = avg_gas_price.merge(max_gas_price, on='FROM_ADDRESS')
    result_df = result_df.merge(avg_gas_limit, on='FROM_ADDRESS')
    result_df = result_df.merge(max_gas_limit, on='FROM_ADDRESS')
    result_df = result_df.merge(avg_gas_used, on='FROM_ADDRESS')
    result_df = result_df.merge(max_gas_used, on='FROM_ADDRESS')
    result_df = result_df.merge(avg_cumulative_gas_used, on='FROM_ADDRESS')
    result_df = result_df.merge(max_cumulative_gas_used, on='FROM_ADDRESS')
    
    result_df.columns = [
        'ADDRESS', 'TRANSACTION_AVG_GAS_PRICE', 'TRANSACTION_MAX_GAS_PRICE',
        'TRANSACTION_AVG_GAS_LIMIT', 'TRANSACTION_MAX_GAS_LIMIT',
        'TRANSACTION_AVG_GAS_USED', 'TRANSACTION_MAX_GAS_USED',
        'TRANSACTION_AVG_CUMULATIVE_GAS_USED', 'TRANSACTION_MAX_CUMULATIVE_GAS_USED'
    ]
    
    result_df = result_df[result_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_max_fee_per_gas_features(data_df, filter_df):
    data_df['MAX_FEE_PER_GAS'] = pd.to_numeric(data_df['MAX_FEE_PER_GAS'], errors='coerce')
    avg_max_fee_per_gas = data_df.groupby('FROM_ADDRESS')['MAX_FEE_PER_GAS'].mean().reset_index(name='TRANSACTION_AVG_MAX_FEE_PER_GAS')
    max_max_fee_per_gas = data_df.groupby('FROM_ADDRESS')['MAX_FEE_PER_GAS'].max().reset_index(name='TRANSACTION_MAX_MAX_FEE_PER_GAS')
    result_df = pd.merge(avg_max_fee_per_gas, max_max_fee_per_gas, on='FROM_ADDRESS', how='outer')
    result_df.columns = ['ADDRESS', 'TRANSACTION_AVG_MAX_FEE_PER_GAS', 'TRANSACTION_MAX_MAX_FEE_PER_GAS']
    result_df = result_df[result_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_max_priority_fee_per_gas_features(data_df, filter_df):
    data_df['MAX_PRIORITY_FEE_PER_GAS'] = pd.to_numeric(data_df['MAX_PRIORITY_FEE_PER_GAS'], errors='coerce')
    avg_max_priority_fee_per_gas = data_df.groupby('FROM_ADDRESS')['MAX_PRIORITY_FEE_PER_GAS'].mean().reset_index(name='TRANSACTION_AVG_MAX_PRIORITY_FEE_PER_GAS')
    max_max_priority_fee_per_gas = data_df.groupby('FROM_ADDRESS')['MAX_PRIORITY_FEE_PER_GAS'].max().reset_index(name='TRANSACTION_MAX_MAX_PRIORITY_FEE_PER_GAS')
    result_df = pd.merge(avg_max_priority_fee_per_gas, max_max_priority_fee_per_gas, on='FROM_ADDRESS', how='outer')
    result_df.columns = ['ADDRESS', 'TRANSACTION_AVG_MAX_PRIORITY_FEE_PER_GAS', 'TRANSACTION_MAX_MAX_PRIORITY_FEE_PER_GAS']
    result_df = result_df[result_df['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_max_zero_value_from_to_count(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    zero_value_df = data_df[data_df['VALUE'] == 0]
    
    from_to_zero_value_counts = zero_value_df.groupby(['FROM_ADDRESS', 'TO_ADDRESS']).size().reset_index(name='COUNT')
    max_from_to_zero_value_counts = from_to_zero_value_counts.groupby('FROM_ADDRESS')['COUNT'].max().reset_index(name='MAX_FROM_TO_ZERO_VALUE_COUNT')
    max_from_to_zero_value_counts.columns = ['ADDRESS', 'MAX_FROM_TO_ZERO_VALUE_COUNT']
    result_df = max_from_to_zero_value_counts[max_from_to_zero_value_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df

def calculate_max_zero_value_to_from_count(data_df, filter_df):
    data_df['VALUE'] = pd.to_numeric(data_df['VALUE'], errors='coerce')
    zero_value_df = data_df[data_df['VALUE'] == 0]
    
    to_from_zero_value_counts = zero_value_df.groupby(['TO_ADDRESS', 'FROM_ADDRESS']).size().reset_index(name='COUNT')
    max_to_from_zero_value_counts = to_from_zero_value_counts.groupby('TO_ADDRESS')['COUNT'].max().reset_index(name='MAX_TO_FROM_ZERO_VALUE_COUNT')
    max_to_from_zero_value_counts.columns = ['ADDRESS', 'MAX_TO_FROM_ZERO_VALUE_COUNT']
    result_df = max_to_from_zero_value_counts[max_to_from_zero_value_counts['ADDRESS'].isin(filter_df['ADDRESS'])]
    return result_df


def make_transaction_features(transaction_raw_file, output_path, filter_df):
    data_df = pd.read_parquet(transaction_raw_file)

    deposit_zero_value_count = calculate_deposit_zero_value_count(data_df, filter_df)
    withdrawal_zero_value_count = calculate_withdrawal_zero_value_count(data_df, filter_df)
    print("zero_value_count done")
    # max_zero_value_from_to_count = calculate_max_zero_value_from_to_count(data_df, filter_df)
    # max_zero_value_to_from_count = calculate_max_zero_value_to_from_count(data_df, filter_df)
    # print("max_zero_value_count done")

    all_action_counts_df = calculate_all_action_num(data_df, filter_df)
    print("all_action_counts_df done")
    deposit_action_num_df = calculate_each_deposit_action_num(data_df, filter_df)
    print("deposit_action_num_df done")
    withdrawal_action_num_df = calculate_each_withdrawal_action_num(data_df, filter_df)
    print("withdrawal_action_num_df done")
    unique_addresses_df = calculate_unique_addresses(data_df, filter_df)
    print("unique_addresses_df done")
    max_nonce_per_contract_df = calculate_max_nonce_per_contract(data_df, filter_df)
    print("max_nonce_per_contract_df done")
    deposit_average_value_df = calculate_deposit_average_value(data_df, filter_df)
    print("deposit_average_value_df done")
    deposit_maximum_value_df = calculate_deposit_maximum_value(data_df, filter_df)
    print("deposit_maximum_value_df done")
    withdrawal_average_value_df = calculate_withdrawal_average_value(data_df, filter_df)
    print("withdrawal_average_value_df done")
    withdrawal_maximum_value_df = calculate_withdrawal_maximum_value(data_df, filter_df)
    print("withdrawal_maximum_value_df done")
    tx_fee_features_df = calculate_tx_fee_features(data_df, filter_df)
    print("tx_fee_features_df done")
    gas_features_df = calculate_gas_features(data_df, filter_df)
    print("gas_features_df done")

    max_fee_per_gas_features_df = calculate_max_fee_per_gas_features(data_df, filter_df)
    print("max_fee_per_gas_features_df done")
    max_priority_fee_per_gas_features_df = calculate_max_priority_fee_per_gas_features(data_df, filter_df)
    print("max_priority_fee_per_gas_features_df done")
    
    final_df = filter_df.copy()
    dataframes = [
        deposit_zero_value_count,
        withdrawal_zero_value_count,

        # max_zero_value_from_to_count,
        # max_zero_value_to_from_count,

        all_action_counts_df,
        deposit_action_num_df,
        withdrawal_action_num_df,
        unique_addresses_df,
        max_nonce_per_contract_df,
        deposit_average_value_df,
        deposit_maximum_value_df,
        withdrawal_average_value_df,
        withdrawal_maximum_value_df,
        tx_fee_features_df,
        gas_features_df,
        max_fee_per_gas_features_df,
        max_priority_fee_per_gas_features_df
    ]

    for df in dataframes:
        final_df = pd.merge(final_df, df, on='ADDRESS', how='left')
    return final_df