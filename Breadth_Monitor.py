# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:29:34 2024

@author: Andy
"""
import pandas as pd
from datetime import datetime, timedelta
import sys
import yfinance as yf
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

#####################################
# Variables - Setup
#####################################

hoje = datetime.now()
reference_time = 18

# Where files and database are stored (same directory as MAIN PROGRAM)
data_folder = 'Data/Data_files'
codes_folder = 'Data/Codes_to_download'
plots_folder = 'Data/Plots'
pdf_folder = 'Data/PDF'
market_monitor_folder = 'Data/Mkt_Monitor'

yahoo_idx_components_dictionary = {
    1: {'idx_code': '^BVSP', 'market': 'bovespa', 'codes_csv': 'IBOV.csv'},
    2: {'idx_code': '^IXIC', 'market': 'nasdaq', 'codes_csv': 'NASDAQ.csv'},
    3: {'idx_code': '^FTSE', 'market': 'ftse100', 'codes_csv': 'none'},
    4: {'idx_code': '^FTMC', 'market': 'ftse250', 'codes_csv': 'FTSE250.csv'},
    5: {'idx_code': '^GSPC', 'market': 'sp500', 'codes_csv': 'SP500.csv'},
    6: {'idx_code': '^DJI', 'market': 'dow30', 'codes_csv': 'DOW.csv'},
    7: {'idx_code': 'GC=F', 'market': 'gold', 'codes_csv': 'none'},
    8: {'idx_code': 'BTC-USD', 'market': 'bitcoin', 'codes_csv': 'none'},
    9: {'idx_code': 'BRL=X', 'market': 'dollar', 'codes_csv': 'none'},
    10: {'idx_code': 'CL=F', 'market': 'crude', 'codes_csv': 'none'},
    11: {'idx_code': 'ZN=F', 'market': '10yrTnote', 'codes_csv': 'none'},
    12: {'idx_code': 'ZT=F', 'market': '2yrTnote', 'codes_csv': 'none'},
    13: {'idx_code': '^VIX', 'market': 'vix', 'codes_csv': 'none'},
    14: {'idx_code': '^IBX50', 'market': 'ibx50', 'codes_csv': 'none'}
    # 15: {'idx_code': 'IFIX.SA', 'market': 'fii', 'codes_csv': 'none'},
    # 16: {'idx_code': 'IDIV.SA', 'market': 'dividends', 'codes_csv': 'none'},
}

compare_index_list = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13]

# Create a new dictionary using dictionary comprehension
filtered_index_dict = {key: {'idx_code': value['idx_code'], 'market': value['market']}
                       for key, value in yahoo_idx_components_dictionary.items() if key in compare_index_list}

# Variables for breadth and plots
mas_to_use = [1, 5, 12, 25, 40, 50, 100, 200]  # Moving average: 1 is used as 'Close'
time_periods = [252, 63, 21]  # 1 year, 3 months and 1 month

# For table
rows = 20
percentiles = [0, 0.1, 0.2, 0.8, 0.9]
percentile_color = {0: 'red', 0.1: 'orange', 0.2: 'white', 0.8: 'lightgreen', 0.9: 'green'}
# Rows to use for ranking calculations
sample_size_for_ranking = 1000


# -----------------------------FUNCTIONS----------------------------------
##########################################################################
# Work with all indices or choose one
##########################################################################
def get_market_map(mkt_map):
    # Work with all indexes or just one
    for key, value in mkt_map.items():
        print(f'Key: {key}, Market: {value["market"]}')

    # Prompt the user to choose between all entries or a specific number
    choice_all_or_one = input("Use all (Enter) or select a number:")

    if not choice_all_or_one:
        # Process all entries by default
        map_to_use = mkt_map
    else:
        # Process a specific number
        selected_number = int(choice_all_or_one)
        if selected_number in mkt_map:
            map_to_use = {selected_number: mkt_map[selected_number]}
        else:
            print("Invalid selection. Exiting the program.")
            sys.exit()

    return map_to_use


##########################################################################
#  Get users choice: update, use existing or download new databases
##########################################################################
def get_user_choice():
    print('What do you want to do?')
    print('Update: 1, Use existing data: 2 or Create new databases: 3')
    while True:
        try:
            user_choice = int(input('Enter your choice (1: update, 2: use as is, or 3: download new): ') or 1)
            if user_choice in [1, 2, 3]:
                return user_choice  # Return the value if it's valid
            else:
                print('Invalid choice. Please enter 1, 2, or 3.')
        except ValueError:
            print('Invalid input. Please enter a number.')


##########################################################################
# Choose lookback period in days
##########################################################################
def get_lookback():
    while True:
        try:
            lookback_period = int(input("Days to look back (756/~3yr default): ") or 756)
            return lookback_period  # Return the lookback value if the input is successfully converted to an integer
        except ValueError:
            print("Invalid input. Please enter an integer.")


##########################################################################
# Setup START / END dates
##########################################################################
def download_until():
    global hoje
    global reference_time

    # 0: Mon, 1: Tue, 2: Wed, 3: Thur, 4: Fri, 5: Sat, 6: Sun
    if hoje.weekday() in {5, 6}:  # Weekend, download up to Friday
        dl_until = hoje - timedelta(days=(hoje.weekday() % 5 + 1))  # % = remainder

    # If weekday and market still open, data unavailable for today, use yesterday
    elif datetime.now().hour < reference_time:
        dl_until = datetime.today() - timedelta(days=1)

    # If weekday and market closed, use today
    else:
        dl_until = datetime.today()

    # print(f'New data will be downloaded until: {end_download_on.strftime("%A, %d %B %Y")}')

    return dl_until


##################################################################################
# Download historical data for the defined period for all tickers in selected market
##################################################################################
def download_components_data(mkt, ticker_list, first, last):
    # Download historical data for chosen market components

    print(f"{mkt} data will be downloaded from {first} until {last}")
    components_df = yf.download(ticker_list, start=first, end=last, rounding=True)

    # Convert the index to DateTimeIndex
    components_df.index = pd.to_datetime(components_df.index)

    components_df.to_csv(f'{data_folder}/EOD_{mkt}.csv')  # , index=False)
    components_df_file_saved = f'{data_folder}/EOD_{mkt}.csv'

    print(f"Downloaded and saved: {components_df_file_saved}")


##########################################################################
# Download historical data for the selected index
##########################################################################
def download_index_data(index_code, first, last):
    # Download historical data for chosen index

    global data_folder

    # Download historical data
    data = yf.download(index_code, start=first, end=last)
    data.index = pd.to_datetime(data.index)

    # Save to CSV file
    data.to_csv(f"{data_folder}/INDEX_{index_code}.csv")  # , index=False)

    print(f"Data for {index_code} downloaded from {first} until {last}\n"
          f"Saved to: {data_folder}/INDEX_{index_code}.csv")


##########################################################################
# Create all (or chosen) databases
##########################################################################
def create_databases(market_list, start, end):

    global yahoo_idx_components_dictionary

    print('Will overwrite any existing files.')

    # all_markets = [key for key in yahoo_idx_components_dictionary.keys()]

    # Always need to download ALL indexes
    for number in yahoo_idx_components_dictionary:
        mkt_details = yahoo_idx_components_dictionary[number]
        # Files to be used
        i_c = mkt_details['idx_code']
        # m_n = mkt_details['market']
        # tikrs = mkt_details['codes_csv']
        # Download and save index data
        download_index_data(i_c, start, end)

    # Only download component data from SELECTED index or ALL
    for number in market_list:
        mkt_details = market_list[number]
        # Files to be used
        # i_c = mkt_details['idx_code']
        m_n = mkt_details['market']
        tikrs = mkt_details['codes_csv']
        if tikrs != 'none':
            # First make dataframe of codes
            t_df = pd.read_csv(f'{codes_folder}/{tikrs}')
            # Next make the list for yahoo download
            t_list = t_df['Code'].tolist()  # codes_csv must have column "Code"
            # Use list to download components data
            download_components_data(m_n, t_list, start, end)


##########################################################################
# Align indexes of eod_df with index_df
##########################################################################
def align_indexes(df1, df2):  # Align DataFrames separately

    eoddf, idxdf = df2.align(df1, join='outer', axis=0)
    print(f'{df1} and {df2} indexes aligned')

    return idxdf, eoddf


##########################################################################
# Check if there are any NaN values in the entire DataFrame
##########################################################################
def has_nan(idx, eod, mkt):
    # .any().any() checks if there are any NaN values in the entire DataFrame
    if idx.isna().any().any():  # If TRUE...
        # Check if there are NaNs in rows
        # nan_in_rows = idx[idx.isna().any(axis=1)]
        print(f'{mkt} index df has rows with NaN values:')
        # print(f'{mkt} index df rows with NaN values:')
        # print(nan_in_rows)

    else:
        print(f'No Nan found in {mkt} index df')

    if eod.isna().any().any():  # If TRUE...
        # Check if there are NaNs in rows
        # nan_in_rows = eod[eod.isna().any(axis=1)]
        print(f'{mkt} components df has rows with NaN values:')
        # print(nan_in_rows)

    else:
        print(f'No Nan found in {mkt} components df')


##########################################################################
# Update and save selected market components
##########################################################################
def update_components_data(mkt, df):
    global data_folder
    global codes_folder
    global hoje
    global reference_time
    global until_date

    # updated_components = None
    # Read existing eod/components csv for update
    component_file = f'{data_folder}/EOD_{mkt}.csv'

    # Extract list of codes from data csv
    tickers_list = df.columns.get_level_values(1).unique().tolist()

    # Get last available date in dataframe
    last_date_in_csv = df.index[-1]

    # If download until date is SAME as last date in csv: do nothing
    if until_date.date() == last_date_in_csv.date():
        print(f"{mkt} update not required. Last date in file is today")
        # return updated_components  # Return NONE: GPT says to make sure to return something meaningful

    # If download until date is AFTER last date in csv: update
    elif until_date.date() > last_date_in_csv.date():
        # Download missing days
        start_update_on = last_date_in_csv + timedelta(days=1)
        missing_days_df = yf.download(tickers_list, start=start_update_on, end=until_date, rounding=True)
        # Convert the indexes to DateTimeIndex
        missing_days_df.index = pd.to_datetime(missing_days_df.index)
        print(f'Downloaded {mkt} from {last_date_in_csv.strftime("%A, %d %B %y")}'
              f' to {until_date.strftime("%A, %d %B %y")}')
        # Join update to original
        updated_components = pd.concat([df, missing_days_df], axis=0).loc[
            ~pd.concat([df, missing_days_df], axis=0).index.duplicated(keep='first')]
        # Check for Duplicate Indexes:
        print(f'Duplicated indexes = {updated_components.index.duplicated().any()}')
        # Get last date with data
        updated_until = updated_components.index[-1].strftime("%A, %d %B %y")
        # Save and check components file as csv
        updated_components.to_csv(component_file)
        if not updated_components.tail(10).isnull().values.any():
            print(f'{mkt} updated until: {updated_until}. Filename= {component_file}')
        else:
            print(f'{mkt} updated with NaN. Check update.')

    else:  # start_update_on AFTER until_date
        print(f'Error: {mkt} start date after end date')
        #  return updated_idx  # GPT says to make sure to return something meaningful


##########################################################################
# Update and save selected index data
##########################################################################
def update_index_data(idx, df):
    global data_folder
    global codes_folder
    global hoje
    global reference_time
    global until_date

    # updated_idx = None
    idx_file = f'{data_folder}/INDEX_{idx}.csv'

    # Yahoo seems to be a day late with volume on indexes so...
    # ...if last volume is zero: drop, download and rewrite last row.
    if df['Volume'].iloc[-1] == 0:
        df = df.drop(df.index[-1])
        print(f'{df.index[-1]} has zero volume. Dropping row and re-updating')

    start_update_on = df.index[-1] + timedelta(days=1)

    # If last day in df = update until date, no need to update
    if until_date.date() == df.index[-1].date():
        print('No need to update')
        #  return updated_idx  # GPT says to make sure to return something meaningful

    elif df.index[-1].date() < until_date.date():
        print(f'Last date in {idx} file: {start_update_on.strftime("%A, %d %B %y")}')
        print(f'Update {idx} until: {until_date.strftime("%A, %d %B %y")}')
        # Download missing days
        missing_days = yf.download(idx, start=start_update_on, end=until_date, rounding=True)
        # Convert the indexes to DateTimeIndex
        missing_days.index = pd.to_datetime(missing_days.index)
        # Join update to original
        updated_idx = pd.concat([df, missing_days], axis=0).loc[
            ~pd.concat([df, missing_days], axis=0).index.duplicated(keep='first')]
        # Check for Duplicate Indexes:
        print(f'Duplicated indexes = {updated_idx.index.duplicated().any()}')
        # Get last date with data
        updated_until = updated_idx.index[-1].strftime("%A %d %B %y")
        # Check and save index file as csv
        updated_idx.to_csv(idx_file)
        if not updated_idx.tail(10).isnull().values.any():
            print(f'{idx} updated until: {updated_until}. Filename= {idx_file}')
        else:
            print(f'{idx} updated with NaN. Check update.')
    else:  # start_update_on AFTER until_date
        print(f'Error: {idx} start date after end date')
        #  return updated_idx  # GPT says to make sure to return something meaningful


##########################################################################
# Update all (or chosen) databases
##########################################################################
def update_databases(market_list):

    global yahoo_idx_components_dictionary

    # Initialize before the loop to stop "Local variable 'x' might be referenced before assignment"
    last_date_in_comp_csv = None
    com_df = None
    last_date_in_ind_csv = None
    ind_df = None

    # Always need to update ALL indexes
    for number in yahoo_idx_components_dictionary:
        mkt_details = yahoo_idx_components_dictionary[number]
        # Files to be used
        i_c = mkt_details['idx_code']
        # m_n = mkt_details['market']
        # tikrs = mkt_details['codes_csv']

        # Read existing index file for update: idx_data_df
        idx_file = f'{data_folder}/INDEX_{i_c}.csv'
        ind_df = pd.read_csv(idx_file, header=0, index_col=0)
        ind_df.index = pd.to_datetime(ind_df.index)
        # Remove Duplicate Index Labels (if any)
        ind_df = ind_df[~ind_df.index.duplicated(keep='first')]
        # last_date_in_ind_csv = ind_df.index[-1]
        print(f'Updating index: {i_c}')
        update_index_data(i_c, ind_df)

    # Only update component data from SELECTED index or ALL
    for number in market_list:
        mkt_details = market_list[number]
        # Files to be used
        # i_c = mkt_details['idx_code']
        m_n = mkt_details['market']
        tikrs = mkt_details['codes_csv']
        print(f'Trying to update: {m_n}')
        # Update components data (where there is a list of tickers for that market)
        if tikrs != 'none':
            # Read existing eod/components csv for update
            component_csv = f'{data_folder}/EOD_{m_n}.csv'
            # Get current eod data: components_df
            com_df = pd.read_csv(component_csv, index_col=0, header=[0, 1])
            com_df.index = pd.to_datetime(com_df.index)
            # Remove Duplicate Index Labels (if any)
            com_df = com_df[~com_df.index.duplicated(keep='first')]

            # last_date_in_comp_csv = com_df.index[-1]
            print(f'Updating components: {m_n}')
            update_components_data(m_n, com_df)
        else:
            print(f'No components to update for {m_n}')

    return last_date_in_comp_csv, com_df, last_date_in_ind_csv, ind_df


##########################################################################
# -----------------BREADTH AND INDICATOR FUNCTIONS------------------------
##########################################################################
##########################################################################
# Simple test plot to see basic chart
##########################################################################
def plot_close_and_volume(df_idx, idx):
    global lookback
    global hoje

    # print(f'Type df_idx = {type(df_idx)}')
    # print('This is df_idx:')
    # print(df_idx)
    p_1 = df_idx.tail(lookback)
    date_labels = p_1.index.strftime("%d/%m/%y").tolist()
    p1 = p_1.reset_index(drop=True)

    # Extract start and end dates from df_idx
    start_date = p_1.index[0].strftime('%d/%m/%y')  # First index value as start date
    end_date = p_1.index[-1].strftime('%d/%m/%y')  # Last index value as end date

    # Create a figure and axis
    graph_name = f'{idx} - Close and Volume'
    fig, ax1 = plt.subplots(figsize=(17, 12))

    # Plot closing price on the first y-axis
    ax1.set_ylabel('Close (log scale)', color='blue')
    ax1.plot(p1.index, p1['Adj Close'])
    ax1.tick_params(axis='y')
    ax1.set_yscale('log')

    # Set x-ticks and x-tick labels for first y-axis

    ax1.set_xticks(p1.index[::xlabel_separation])
    ax1.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Add title as text to the plot using data coordinates
    ax1.text(p1.index[0], p1['Adj Close'].max(), graph_name, color='black', ha='left', va='bottom')

    # Create a second y-axis for volume
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volume', color='red')
    ax2.plot(p1.index, p1['Volume'], color='red')
    ax2.tick_params(axis='y')

    # Set x-ticks and x-tick labels for the second y-axis
    ax2.set_xticks(p1.index[::xlabel_separation])
    ax2.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Display the plot
    plt.title(f'{idx} - from: {start_date} to {end_date} - Lookback: {lookback} days', fontsize=16, fontweight='bold')
    # plt.savefig(f'{plots_folder}/{idx}_close_vol.jpg')
    # plt.show(block=False)
    pdf.savefig()
    plt.close()


##########################################################################
# Plot summed highs and lows of component df
##########################################################################

def highs_and_lows(df_idx, df_eod, t, idx):
    global lookback

    eod_c = df_eod['Adj Close']
    idx_c = df_idx['Adj Close']

    # Calculate rolling max and min for 3 months and 1 month
    rolling_12m_high = eod_c.rolling(window=t[0]).max()
    rolling_12m_low = eod_c.rolling(window=t[0]).min()
    rolling_3m_high = eod_c.rolling(window=t[1]).max()
    rolling_3m_low = eod_c.rolling(window=t[1]).min()
    rolling_1m_high = eod_c.rolling(window=t[2]).max()
    rolling_1m_low = eod_c.rolling(window=t[2]).min()

    # Initialize df2 with 0 values
    df2 = pd.DataFrame(0, index=eod_c.index, columns=pd.MultiIndex.from_product(
        [['ATH', 'ATL', '12MH', '12ML', '3MH', '3ML', '1MH', '1ML'], eod_c.columns]))

    # Populate df2 based on conditions
    for col in eod_c.columns:
        df2[('ATH', col)] = (eod_c[col] >= eod_c[col].expanding().max()).astype(int)
        df2[('ATL', col)] = -(eod_c[col] <= eod_c[col].expanding().min()).astype(int)
        df2[('12MH', col)] = (eod_c[col] >= rolling_12m_high[col]).astype(int)
        df2[('12ML', col)] = -(eod_c[col] <= rolling_12m_low[col]).astype(int)
        df2[('3MH', col)] = (eod_c[col] >= rolling_3m_high[col]).astype(int)
        df2[('3ML', col)] = -(eod_c[col] <= rolling_3m_low[col]).astype(int)
        df2[('1MH', col)] = (eod_c[col] >= rolling_1m_high[col]).astype(int)
        df2[('1ML', col)] = -(eod_c[col] <= rolling_1m_low[col]).astype(int)

    # Group by level 0 and sum within each group
    hl_df = df2.T.groupby(level=0, sort=False).sum().T

    # Creating the High/Low difference DataFrame with the required columns
    hl_diff_df = pd.DataFrame(index=hl_df.index)
    hl_diff_df['ATH-ATL'] = hl_df['ATH'] + hl_df['ATL']
    hl_diff_df['12MH-12ML'] = hl_df['12MH'] + hl_df['12ML']
    hl_diff_df['3MH-3ML'] = hl_df['3MH'] + hl_df['3ML']
    hl_diff_df['1MH-1ML'] = hl_df['1MH'] + hl_df['1ML']

    # Combine hl_df and hl_diff_df into hl&diff_df
    hl_and_diff_df = pd.concat([hl_df, hl_diff_df], axis=1)

    #############################################################################
    # PLOTTING
    #############################################################################
    # Make new df to extract dates for plotting

    p = hl_df.tail(lookback)
    q = hl_diff_df.tail(lookback)

    p1 = p.reset_index().rename(columns={'index': 'Date'})
    q1 = q.reset_index().rename(columns={'index': 'Date'})
    pidx = idx_c.tail(lookback)

    ######################################################
    # Plot Summed Highs and Lows vs Close on separate page
    ######################################################

    fig1, ax1 = plt.subplots(figsize=(17, 12))

    # Define colors for each level 0 label
    colors = {
        'ATH': 'deepskyblue',
        'ATL': 'saddlebrown',
        '12MH': 'forestgreen',
        '12ML': 'red',
        '3MH': 'mediumseagreen',
        '3ML': 'tomato',
        '1MH': 'palegreen',
        '1ML': 'peachpuff'
    }

    for label in reversed(p1.columns[1:]):
        ax1.bar(p1.index, p1[label], label=label, color=colors[label])

    ax1.set_ylabel('Number of Highs and Lows')

    date_labels = p1['Date'].dt.strftime("%d/%m/%y").tolist()
    ax1.set_xticks(p1.index[::xlabel_separation])
    ax1.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Add title and legend
    ax1.set_title(f"{idx} - Highs and Lows")
    ax1.legend(labels=p1.columns, loc='upper left')  # Use column names for legend labels

    # Add a horizontal dotted line to show where bottoms might be
    ax1.axhline(y=0, color='black', linestyle='--')

    # Add index to other axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=2)
    ax1_twin.set_ylabel(idx, color='black')
    ax1_twin.set_xticks(p1.index[::xlabel_separation])
    ax1_twin.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    pdf.savefig(fig1)
    plt.close(fig1)

    #################################################
    # Plot Highs minus Lows vs Close on separate page
    #################################################

    fig2, ax2 = plt.subplots(figsize=(17, 12))

    # Define colors for each level 0 label
    colors = {
        'ATH-ATL': 'darkmagenta',
        '12MH-12ML': 'forestgreen',
        '3MH-3ML': 'crimson',
        '1MH-1ML': 'pink'
    }

    # Plot other columns with thinner lines first
    for col in ['ATH-ATL', '3MH-3ML', '1MH-1ML']:
        ax2.plot(q1[col], label=col, linewidth=0.8, color=colors[col])  # Thinner lines for other columns

    # Plot '12MH-12ML' last to bring it to the front
    ax2.plot(q1['12MH-12ML'], label='12MH-12ML', linewidth=1.6, color='forestgreen')  # Standard line width for
    # 12MH-12ML

    date_labels = q1['Date'].dt.strftime("%d/%m/%y").tolist()
    ax2.set_xticks(q1.index[::xlabel_separation])
    ax2.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Add title and legend
    ax2.set_title(f"{idx} - New Highs minus New Lows")
    # ax2.legend(labels=q1.columns, loc='upper left')  # Use column names for legend labels

    # Add a horizontal dotted line to show where bottoms might be
    ax2.axhline(y=0, color='black', linestyle='--')

    # Add index to other axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(q1.index, pidx, 'black', label=idx, linewidth=2)
    ax2_twin.set_ylabel(idx, color='black')
    ax2_twin.set_xticks(q1.index[::xlabel_separation])
    ax2_twin.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    pdf.savefig(fig2)
    plt.close(fig2)

    """# Combine hl_df and hl_diff_df into hl&diff_df
    # hl_and_diff_df = pd.concat([hl_df, hl_diff_df], axis=1)

    # return hl_and_diff_df
    
    #############################################################################
    # PLOTTING
    #############################################################################
    # Make new df to extract dates for plotting

    p = hl_df.tail(lookback)
    q = hl_diff_df.tail(lookback)

    p1 = p.reset_index().rename(columns={'index': 'Date'})
    q1 = q.reset_index().rename(columns={'index': 'Date'})
    pidx = idx_c.tail(lookback)

    # fig, ax = plt.subplots(figsize=(17, 12))
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(17, 12))

    #######################################
    # Plot Summed Highs and Lows vs Close
    #######################################

    # Define colors for each level 0 label
    colors = {
        'ATH': 'deepskyblue',
        'ATL': 'saddlebrown',
        '12MH': 'forestgreen',
        '12ML': 'red',
        '3MH': 'mediumseagreen',
        '3ML': 'tomato',
        '1MH': 'palegreen',
        '1ML': 'peachpuff'
    }

    for label in reversed(p1.columns[1:]):
        axs[0].bar(p1.index, p1[label], label=label, color=colors[label])

    axs[0].set_ylabel('Number of Highs and Lows')

    date_labels = p1['Date'].dt.strftime("%d/%m/%y").tolist()
    axs[0].set_xticks(p1.index[::xlabel_separation])
    axs[0].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Add title and legend
    axs[0].set_title(f"{idx} - Highs and Lows")
    axs[0].legend(labels=p1.columns, loc='upper left')  # Use column names for legend labels

    # Add a horizontal dotted line to show where bottoms might be
    axs[0].axhline(y=0, color='black', linestyle='--')

    # Add index to other axis
    axs0_twin = axs[0].twinx()
    axs0_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=2)
    axs0_twin.set_ylabel(idx, color='black')
    axs0_twin.set_xticks(p1.index[::xlabel_separation])
    axs0_twin.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    ################################
    # Plot Highs minus Lows vs Close
    ################################

    for col in ['ATH-ATL', '12MH-12ML', '3MH-3ML', '1MH-1ML']:
        axs[1].plot(q1[col], label=col)

    date_labels = q1['Date'].dt.strftime("%d/%m/%y").tolist()
    axs[1].set_xticks(q1.index[::xlabel_separation])
    axs[1].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Add title and legend
    axs[1].set_title(f"{idx} - New Highs minus New Lows")
    axs[1].legend(labels=q1.columns, loc='upper left')  # Use column names for legend labels

    # Add a horizontal dotted line to show where bottoms might be
    axs[1].axhline(y=0, color='black', linestyle='--')

    # Add index to other axis
    axs1_twin = axs[1].twinx()
    axs1_twin.plot(q1.index, pidx, 'black', label=idx, linewidth=2)
    axs1_twin.set_ylabel(idx, color='black')
    axs1_twin.set_xticks(q1.index[::xlabel_separation])
    axs1_twin.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = axs1_twin.get_legend_handles_labels()
    axs1_twin.legend(lines + lines2, labels + labels2, loc='upper left')


    # plt.savefig(f'{plots_folder}/{idx}_highs_lows.jpg')
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    # Combine hl_df and hl_diff_df into hl&diff_df
    hl_and_diff_df = pd.concat([hl_df, hl_diff_df], axis=1)
"""

    return hl_and_diff_df


##########################################################################
# Create df of moving averages
##########################################################################

def calculate_moving_averages(df_close, mas, label):
    ma_list = []

    for ma in mas:
        ma_column_name = f'MA{ma}' if ma != 1 else label

        # Use rolling function to calculate moving averages for all codes simultaneously
        ma_temp = df_close.rolling(window=ma).mean().round(2)

        # Create MultiIndex columns
        ma_temp.columns = pd.MultiIndex.from_product([[ma_column_name], df_close.columns], names=['MA', 'Code'])

        ma_list.append(ma_temp)

    # Concatenate the DataFrames along columns
    ma_df = pd.concat(ma_list, axis=1)

    return ma_df


##########################################################################
# Create df of exp moving averages for Traders Eden
##########################################################################

def calculate_traders_edens(df_close, idx_data, idx):

    # Calculate EMAs
    ma8 = df_close.ewm(span=8, min_periods=0, adjust=False, ignore_na=False).mean().round(2)
    ma80 = df_close.ewm(span=80, min_periods=0, adjust=False, ignore_na=False).mean().round(2)

    # Create MultiIndex columns
    ma8.columns = pd.MultiIndex.from_product([['MA8'], df_close.columns], names=['MA8', 'Code'])
    ma80.columns = pd.MultiIndex.from_product([['MA80'], df_close.columns], names=['MA80', 'Code'])

    # Concatenate the DataFrames along columns
    te_ma_df = pd.concat([ma8, ma80], axis=1)

    # Calculate EMA direction:
    ma8_dir = np.sign(te_ma_df["MA8"].diff())
    ma80_dir = np.sign(te_ma_df["MA80"].diff())

    # Initialize new traders eden df with zeros, same index and columns as ma8_dir and ma80_dir
    df_te = pd.DataFrame(0, index=ma8_dir.index, columns=ma8_dir.columns)

    for col in ma8_dir.columns:
        mask_if_both_1 = (ma8_dir[col] == 1) & (ma80_dir[col] == 1)
        mask_if_both_neg1 = (ma8_dir[col] == -1) & (ma80_dir[col] == -1)

        df_te.loc[mask_if_both_1, col] = 1
        df_te.loc[mask_if_both_neg1, col] = -1

    # Calculate Eden_Up and Eden_Down as %
    num_tickers = df_te.shape[1]
    eden_df = pd.DataFrame(index=df_te.index)
    eden_df['Eden_Up'] = 100 * df_te.eq(1).sum(axis=1) / num_tickers
    eden_df['Eden_Down'] = 100 * -df_te.eq(-1).sum(axis=1) / num_tickers
    # Calculate difference b adding. Note: Eden_Down is already negative
    eden_df['Diff'] = eden_df['Eden_Up'] + eden_df['Eden_Down']
    eden_df[idx] = idx_data.round(0)

    #############################################
    # Plotting
    #############################################
    p1 = eden_df.tail(lookback)
    date_labels = p1.index.strftime("%d/%m/%y").tolist()
    p1 = p1.reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(17, 12))

    ax1.plot(p1.index, p1['Eden_Up'], color='palegreen', label='Eden_Up %')
    ax1.plot(p1.index, p1['Eden_Down'], color='lightcoral', label='Eden_Down %')
    ax1.bar(p1.index, p1['Diff'], color='lightblue', label="Difference (Up-Down)")
    ax1.set_xlabel('Date')
    ax1.set_xticks(p1.index[::xlabel_separation])
    ax1.set_xticklabels(date_labels[::xlabel_separation], rotation=45)
    ax1.set_ylabel('% in Eden')
    ax1.set_title(f"{idx} - % of stocks in Up/Down Traders Eden. (8/80 EMA same direction")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(p1.index, p1[idx], color='black', label=idx)
    ax2.set_ylabel(idx)
    ax2.legend(loc='upper right')

    # Set x-ticks and x-tick labels for the second y-axis

    pdf.savefig()
    plt.close()

    return eden_df


##########################################################################
# Distance from MA to Close
##########################################################################
def difference_close_to_ma(df_close, df_close_idx, ma_df, idx):
    global mas_to_use
    global lookback

    # Calculate t
    # The differences between the original DataFrame and moving averages
    simple_diff_df = df_close.sub(ma_df, level='Code')
    simple_diff_df.drop('Close', axis=1, level=0, inplace=True)

    # Create MultiIndex columns for the differences
    # Get the column names of ma_df
    # ma_column_names = ma_df.columns.get_level_values(0).unique()

    # Rename the columns of diff_df
    mapper = {'MA5': 'DiffMA5', 'MA12': 'DiffMA12', 'MA25': 'DiffMA25', 'MA40': 'DiffMA40', 'MA50': 'DiffMA50',
              'MA100': 'DiffMA100', 'MA200': 'DiffMA200', }
    simple_diff_df = simple_diff_df.rename(columns=mapper, level=0)
    # diff_column_names = diff_df.columns.get_level_values(0).unique()

    # Calculate absolute percentage difference
    abs_pct_diff_df = (simple_diff_df / ma_df['Close']).abs()

    diff_df = abs_pct_diff_df.T.groupby(level=0, sort=False).mean().T.round(2)
    # print(diff_df)

    #############################################################################
    # PLOTTING
    #############################################################################
    # p = diff_df.reset_index().rename(columns={'index': 'Date'})
    # p1 = p.tail(lookback)
    p = diff_df.tail(lookback)
    date_labels = p.index.strftime("%d/%m/%y").tolist()
    p1 = p.reset_index().rename(columns={'index': 'Date'})
    pidx = df_close_idx.tail(lookback)

    # Create subplots
    fig, ax = plt.subplots(figsize=(17, 12))

    #################################################################################

    # Create a stacked bar chart
    bottom = None
    to_plot = p1.columns[1:]  # Ignore 'Date'
    # for col in p1.columns[1:]:
    for col in to_plot:
        ax.bar(p1.index, p1[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = p1[col]
        else:
            bottom += p1[col]

    # Set x-ticks and x-tick labels for first y-axis
    ax.set_xticks(p1.index[::xlabel_separation])
    ax.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Creating the second y-axis on the right
    ax_twin = ax.twinx()
    ax_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=2)

    # Adding labels for both y-axes
    ax.set_ylabel("Avg % between MA and Close")
    ax_twin.set_ylabel(idx, color='black')

    # Combine legends for ax and ax_twin into a single legend
    lines, labels = ax.get_legend_handles_labels()
    lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
    ax.legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    # Add title for plot
    ax.set_title(f"{idx} - Market's Avg % diff. between ticker's moving average (MA) and Close")

    # plt.savefig(f'{plots_folder}/{idx}_close_to_ma.jpg')
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    # return diff_df
    # return abs_pct_diff_df
    return diff_df


#############################################################################
# Sums of close > MAs, plus 2108
#############################################################################
def close_over_mas(df_mas, label, df_close_idx, idx):
    # No need to plot under MA because it's almost a mirror image of over

    global total_stocks
    global lookback

    df_5_25 = df_mas.loc[:, (['Close', 'MA5', 'MA12', 'MA25'], slice(None))]
    df_40 = df_mas.loc[:, (['Close', 'MA40'], slice(None))]
    df_50_200 = df_mas.loc[:, (['Close', 'MA50', 'MA100', 'MA200'], slice(None))]

    # Calculate 1 0 or -1 if close >, = or < MA
    low_mas = np.sign(df_5_25.loc[:, label] - df_5_25).drop(columns=label, level=0)
    mid_ma = np.sign(df_40.loc[:, label] - df_40).drop(columns=label, level=0)
    high_mas = np.sign(df_50_200.loc[:, label] - df_50_200).drop(columns=label, level=0)

    # Rename columns (only level 0) of multiindex
    low_mas.columns = pd.MultiIndex.from_tuples([(f'$vs{aa}', bb) for aa, bb in low_mas.columns],
                                                names=low_mas.columns.names)
    mid_ma.columns = pd.MultiIndex.from_tuples([(f'$vs{aa}', bb) for aa, bb in mid_ma.columns],
                                               names=mid_ma.columns.names)
    high_mas.columns = pd.MultiIndex.from_tuples([(f'$vs{aa}', bb) for aa, bb in high_mas.columns],
                                                 names=high_mas.columns.names)

    # Separate the 1s and -1s into two different dataframes, above and below
    over_low_mas = low_mas[low_mas > 0].fillna(0)
    over_mid_ma = mid_ma[mid_ma > 0].fillna(0)
    over_high_mas = high_mas[high_mas > 0].fillna(0)

    # Make sums of above
    over_low_mas_sum = over_low_mas.T.groupby(level=0, sort=False).sum().T.astype(int)
    over_low_mas_sum_pct = (over_low_mas_sum / total_stocks * 100).round(1)
    over_mid_ma_sum = over_mid_ma.T.groupby(level=0, sort=False).sum().T.astype(int)
    over_mid_ma_sum_pct = (over_mid_ma_sum / total_stocks * 100).round(1)
    over_high_mas_sum = over_high_mas.T.groupby(level=0, sort=False).sum().T.astype(int)
    over_high_mas_sum_pct = (over_high_mas_sum / total_stocks * 100).round(1)

    # Change column names
    over_low_mas_sum_pct.columns = ['$>MA5', '$>MA12', '$>MA25']
    over_mid_ma_sum_pct.columns = ['$>MA40']
    over_high_mas_sum_pct.columns = ['$>MA50', '$>MA100', '$>MA200']

    #############################################################################
    # PLOTTING
    #############################################################################

    p_1 = over_low_mas_sum_pct.tail(lookback)
    date_labels = p_1.index.strftime("%d/%m/%y").tolist()
    p1 = p_1.reset_index(drop=True)

    p_2 = over_mid_ma_sum_pct.tail(lookback)
    p2 = p_2.reset_index(drop=True)

    p_3 = over_high_mas_sum_pct.tail(lookback)
    p3 = p_3.reset_index(drop=True)

    p_idx = df_close_idx.tail(lookback)
    pidx = p_idx.reset_index(drop=True)

    # Create subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(17, 12))

    # Plot > low MAs on top subplot
    #################################################################################

    axs[0].plot(p1.index, p1, linewidth=1, label=p1.columns)
    # idx on twin y-axis on right
    axs0_twin = axs[0].twinx()
    axs0_twin.plot(pidx.index, pidx, linewidth=2, color='black', label=idx)

    # Set x-ticks and x-tick labels for first y-axis
    axs[0].set_xticks(p1.index[::xlabel_separation])
    axs[0].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axes
    axs[0].set_ylabel('% tickers above low MAs')
    axs0_twin.set_ylabel(idx, color='black')

    # Combine legends for both subplots
    lines0, labels0 = axs[0].get_legend_handles_labels()
    lines0_twin, labels0_twin = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper left')

    # Add title for plot
    axs[0].set_title(f"{idx} - % tickers above (low) Moving Averages")

    # Plot T2108 vs Close on middle subplot
    #################################################################################
    axs[1].plot(p2.index, p2, color='b', alpha=0.7, label='T2108')
    axs[1].set_ylabel('T2108-(%>40MA)', color='black')

    # Set x-ticks and x-tick labels for first y-axis
    axs[1].set_xticks(p1.index[::xlabel_separation])
    axs[1].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Add a horizontal dotted lines to show where lows/highs might be
    high = p2['$>MA40'].max()
    low = p2['$>MA40'].min()
    # Calculate 10% below the high and 10% above the low
    high_10_below = high - 0.2 * (high - low)
    low_10_above = low + 0.2 * (high - low)
    # Plot horizontal lines
    axs[1].axhline(y=high_10_below, color='r', linestyle='--')
    axs[1].axhline(y=low_10_above, color='g', linestyle='--')

    # Add index to other axis
    axs1_twin = axs[1].twinx()
    axs1_twin.plot(pidx.index, pidx, 'black', label=idx)
    axs1_twin.set_ylabel(idx, color='black')

    # Combine legends for both subplots
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines1_twin, labels1_twin = axs1_twin.get_legend_handles_labels()
    axs1_twin.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper left')

    # Add title for plot
    axs[1].set_title(f"{idx} - T2108 (% tickers above 40 day Moving Average)")

    # Plot > high MAs on bottom subplot
    #################################################################################
    axs[2].plot(p3.index, p3, linewidth=1, label=p3.columns)
    # idx on twin y-axis on right
    axs2_twin = axs[2].twinx()
    axs2_twin.plot(pidx.index, pidx, linewidth=2, color='black', label=idx)

    # Set x-ticks and x-tick labels for first y-axis
    axs[2].set_xticks(p1.index[::xlabel_separation])
    axs[2].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axes
    axs[2].set_ylabel('% tickers above high MAs')
    axs2_twin.set_ylabel(idx, color='black')

    # Combine legends for both subplots
    lines2, labels2 = axs[2].get_legend_handles_labels()
    lines2_twin, labels2_twin = axs2_twin.get_legend_handles_labels()
    axs2_twin.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc='upper left')

    # Add title for plot
    axs[2].set_title(f"{idx} - % tickers above (high) Moving Averages")

    # plt.savefig(f'{plots_folder}/{idx}_stocks_over_mas.jpg')
    plt.tight_layout(pad=1.0)
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return over_low_mas_sum_pct, over_mid_ma_sum_pct, over_high_mas_sum_pct


#############################################################################
# Advance/decline ratio
#############################################################################
def advance_decline_ratio(df_close, df_close_idx, idx):
    global lookback

    '''
    Advance/Decline Line.
    Indicates the cumulative sum of differences between the number
    of advancing stocks (advances) and declining stocks (declines).

    Advance/Decline Line (Breadth).
    Calculates the ratio of advances to the overall number of stocks.

    Advance/Decline Line (Daily).
    Calculates ratio of difference between the number of advances
    and declines to the overall number of stocks.

    Advance/Decline Ratio.
    Calculates the ratio of advances to declines.

    Advance/Decline Spread (Issues).
    Calculates the difference between advances and declines.

    Absolute Breadth Index.
    Returns the absolute value of Advance/Decline Spread.
    '''

    # Calculate the signs of daily price changes for each ticker
    price_direction = np.sign(df_close.diff())

    #####################################
    # Calculate the Advance/Decline ratio
    #####################################
    advancing_stocks = (price_direction == 1).sum(axis=1)  # Series
    declining_stocks = (price_direction == -1).sum(axis=1)  # Series

    adv_dec_ratio = (advancing_stocks / declining_stocks).round(2)
    # dec_adv_ratio = - (declining_stocks / advancing_stocks).round(2)

    # Rename the columns
    adv_dec_ratio = adv_dec_ratio.rename("adv_dec_ratio")
    # dec_adv_ratio = dec_adv_ratio.rename("dec_adv_ratio")

    ############################################
    # Calculate the Advance/Decline Line Breadth
    # ##########################################
    # adv_dec_breadth = 100 * (advancing_stocks / num_stocks).round(2)

    # Rename the columns
    # adv_dec_breadth = adv_dec_breadth.rename("adv_dec_breadth")

    ##########################################
    # Calculate the Advance/Decline Line Daily
    # ########################################
    # num_stocks = df_close.shape[1]
    # adv_dec_daily = 100 * ((advancing_stocks-declining_stocks) / num_stocks).round(2)

    # Rename the columns
    # adv_dec_daily = adv_dec_daily.rename("adv_dec_daily")

    ############################################
    # Calculate the cumulative sum of difference
    # ##########################################
    adv_dec_diff = (advancing_stocks - declining_stocks)
    #  print(type(adv_dec_diff)) This is a series
    adv_dec_cum_diff = adv_dec_diff.cumsum()
    # Rename the columns
    adv_dec_cum_diff = adv_dec_cum_diff.rename("adv_dec_cum_diff")
    # print(adv_dec_cum_diff.tail(20))

    # Create a DataFrame
    data = {'Advancing': advancing_stocks, 'Declining': declining_stocks, 'A/D_diff': adv_dec_diff}
    df = pd.DataFrame(data)

    # Calculate the 19-day and 39-day EMAs for advancing and declining stocks
    df['Advancing_EMA_19'] = df['Advancing'].ewm(span=19, adjust=False).mean()
    df['Declining_EMA_19'] = df['Declining'].ewm(span=19, adjust=False).mean()
    df['Advancing_EMA_39'] = df['Advancing'].ewm(span=39, adjust=False).mean()
    df['Declining_EMA_39'] = df['Declining'].ewm(span=39, adjust=False).mean()

    df['A/D_diff_EMA_19'] = df['A/D_diff'].ewm(span=19, adjust=False).mean()
    df['A/D_diff_EMA_39'] = df['A/D_diff'].ewm(span=39, adjust=False).mean()

    # Calculate the McClellan Oscillator

    #  df['McClellan_Oscillator'] = df['Advancing_EMA_19'] - df['Declining_EMA_39']
    df['McClellan_Oscillator'] = df['A/D_diff_EMA_19'] - df['A/D_diff_EMA_39']

    # Display the DataFrame with the McClellan Oscillator
    # print(df)

    #############################################################################
    # PLOTTING
    #############################################################################
    p_1 = adv_dec_ratio.tail(lookback)
    date_labels = p_1.index.strftime("%d/%m/%y").tolist()
    p1 = p_1.reset_index(drop=True)
    # p2 = dec_adv_ratio.tail(lookback)
    # p3 = adv_dec_breadth.tail(lookback)
    # p4 = adv_dec_daily.tail(lookback)
    p_5 = adv_dec_cum_diff.tail(lookback)
    p5 = p_5.reset_index(drop=True)

    p_6 = df['McClellan_Oscillator'].tail(lookback)
    p6 = p_6.reset_index(drop=True)

    p_idx = df_close_idx.tail(lookback)
    pidx = p_idx.reset_index(drop=True)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(17, 12))

    ###################
    # Plot A/D ratio vs Close
    ###################
    axs[0].plot(p1.index, p1, color='b', alpha=0.7, label='Advance/Decline Ratio')
    # ax.plot(p1.index, p2, color='lightblue', alpha=0.7, label='Decline/Advance Ratio')
    # ax.plot(p1.index, p3, color='orange', alpha=0.7, label='Advance/Total Breadth')
    # ax.plot(p1.index, p4, color='green', alpha=0.7, label='Decline/Advance Diff Daily')
    # ax.plot(p1.index, p5, color='yellow', alpha=0.7, label='Cumulative Decline/Advance')

    axs[0].set_xticks(p1.index[::xlabel_separation])
    axs[0].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    axs[0].set_ylabel('Advance/Decline Ratio', color='b')
    # Add title for plot
    axs[0].set_title(f"{idx} - Advance/Decline Ratio")

    # Set y-axis limits to the range of accumulated volume, checking for NaN or Inf
    valid_values = p1[~np.isnan(p1) & ~np.isinf(p1)]
    if len(valid_values) > 0:
        axs[0].set_ylim(bottom=min(valid_values), top=max(valid_values))

    # Add a horizontal line at 0
    axs[0].axhline(y=0, color='blue', linestyle='--', linewidth=1.5)
    # Plot index
    axs0_twin = axs[0].twinx()
    axs0_twin.plot(pidx.index, pidx, 'black', label=idx)
    axs0_twin.fill_between(p1.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs0_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs0_twin.set_ylabel(idx, color='black')

    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    # plt.savefig(f'{plots_folder}/{idx}_adv_decl.jpg')
    # plt.show(block=False)

    #####################
    # Plot A/D cumulative
    #####################
    axs[1].plot(p5.index, p5, color='b', alpha=0.7, label='Cumulative (Advances - Declines)')
    axs[1].set_ylabel('Cumulative (Adv - Dec)', color='black')
    axs[1].set_title(f"{idx} - Cumulative (Advances - Declines)")

    # Set y-axis limits to the range of accumulated volume checking for NaN or Inf
    valid_values = p5[~np.isnan(p5) & ~np.isinf(p5)]
    if len(valid_values) > 0:
        axs[1].set_ylim(bottom=min(valid_values), top=max(valid_values))

    # Add a horizontal line at 0
    axs[1].axhline(y=0, color='blue', linestyle='--', linewidth=1.5)

    axs[1].set_xticks(p1.index[::xlabel_separation])
    axs[1].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Plot index
    axs1_twin = axs[1].twinx()
    axs1_twin.plot(pidx.index, pidx, 'black', label=idx)
    axs1_twin.fill_between(p1.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs1_twin.set_ylabel(idx, color='black')
    axs1_twin.set_ylim(bottom=min(pidx), top=max(pidx))

    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = axs1_twin.get_legend_handles_labels()
    axs1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    ###########################
    # Plot McClellan Oscillator
    ###########################

    # Create bars for McClellan Oscillator, with color depending on value
    colors = ['green' if val > 0 else 'red' for val in p6]

    axs[2].bar(p6.index, p6, color=colors, alpha=0.7, label='McClellan Oscillator')
    axs[2].set_ylabel('McClellan Oscillator', color='black')
    axs[2].set_title(f'{idx} - McClellan Oscillator')

    # Add a horizontal line at 0
    axs[2].axhline(y=0, color='blue', linestyle='--', linewidth=1.5)

    axs[2].set_xticks(p6.index[::xlabel_separation])
    axs[2].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Plot index
    axs2_twin = axs[2].twinx()
    axs2_twin.plot(pidx.index, pidx, 'black', label=idx)
    axs2_twin.fill_between(p6.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs2_twin.set_ylabel(idx, color='black')
    axs2_twin.set_ylim(bottom=min(pidx), top=max(pidx))

    lines, labels = axs[2].get_legend_handles_labels()
    lines2, labels2 = axs2_twin.get_legend_handles_labels()
    axs2_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    # Adjust layout to avoid overlapping labels
    plt.tight_layout()

    # plt.savefig(f'{plots_folder}/{idx}cumulative_adv_decl.jpg')
    pdf.savefig()
    plt.close()

    return adv_dec_ratio  # SERIES


#############################################################################
# Accumulated volume
#############################################################################
def accumulated_volume(df_close, df_vol, idx, df_close_idx):  # eod_df['Adj Close'], eod_df['Volume']
    global total_stocks
    global lookback

    # Calculate the signs of daily price changes for each ticker
    signs = np.sign(df_close.diff())
    dir_vol = signs * df_vol
    cum_vol = dir_vol.sum(axis=1).cumsum().rename('CumVol')  # series
    cum_vol_df = pd.DataFrame(cum_vol)  # This is a dataframe

    # Calculate the percent difference between markets entire volume
    # Replace zeros with the previous non-zero value
    """# df_vol.replace(0, method='ffill', inplace=True) # This is the code that gives the future warning
    # df_vol.fillna(method='ffill', inplace=True)  # Also gives errors"""

    # If df_vol is a slice of another DataFrame, make a copy first
    df_vol = df_vol.copy()

    # Forward fill missing values
    df_vol.ffill(inplace=True)

    total_mkt_vol = df_vol.sum(axis=1).rename('MktVol')
    mkt_vol_pct_chg_1 = total_mkt_vol.pct_change().mul(100)  # series
    mkt_vol_pct_chg = pd.DataFrame(mkt_vol_pct_chg_1)
    # print(mkt_vol_pct_chg.columns)

    #############################################################################
    # PLOTTING
    #############################################################################

    # p1 = cum_vol_df.tail(lookback)
    p = cum_vol_df.tail(lookback)
    date_labels_accv = p.index.strftime("%d/%m/%y").tolist()
    p1 = p.reset_index(drop=True)

    p_2 = mkt_vol_pct_chg.tail(lookback)
    date_labels_vpct = p_2.index.strftime("%d/%m/%y").tolist()
    # date_labels_vpct = p_2.index.tolist()
    p2 = p_2.reset_index(drop=True)
    # Backfill NaN values in p2['Vol%chg']
    # p2['MktVol'] = p2['MktVol'].fillna(method='backfill')  # Deprecated approach
    # Recommended approach (future-proof and explicit):
    p2['MktVol'] = p2['MktVol'].bfill()

    p_idx = df_close_idx.tail(lookback)
    pidx = p_idx.reset_index(drop=True)

    # Create subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(17, 12))

    #############################################################################
    # Plot accumulated volume
    #############################################################################

    axs[0].bar(p1.index, p1['CumVol'], width=1, color='goldenrod', alpha=0.7, label='Accumulated Volume')
    axs[0].set_xticks(p1.index[::xlabel_separation])
    axs[0].set_xticklabels(date_labels_accv[::xlabel_separation], rotation=45, ha='right')
    axs[0].set_ylabel('Accumulated Volume', color='black')

    axs[0].set_title("Accumulated Volume")

    # Set y-axis limits to the range of accumulated volume
    axs[0].set_ylim(bottom=min(p1['CumVol']), top=max(p1['CumVol']))

    # Plot index
    axs0_twin = axs[0].twinx()  # ax1.twinx()

    # ax1.plot(pidx.index, pidx, 'black', label=idx)
    axs0_twin.plot(p1.index, pidx, 'black', label=idx)
    axs0_twin.set_ylabel(idx, color='black')

    lines, labels = axs[0].get_legend_handles_labels()  # ax1.get_legend_handles_labels()
    lines2, labels2 = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    #############################################################################
    # Plot difference vol %
    #############################################################################
    axs[1].bar(p2.index, p2['MktVol'], width=1, color='goldenrod', alpha=0.7, label='Vol % change:-'
                                                                                    ' *ffil used on zero values')
    axs[1].set_xticks(p2.index[::xlabel_separation])
    axs[1].set_xticklabels(date_labels_vpct[::xlabel_separation], rotation=45, ha='right')
    axs[1].set_ylabel('Vol % change', color='black')

    axs[1].set_title("Day to day % volume change")

    # Plot index
    axs1_twin = axs[1].twinx()

    # ax1.plot(pidx.index, pidx, 'black', label=idx)
    axs1_twin.plot(p2.index, pidx, 'black', label=idx)
    axs1_twin.set_ylabel(idx, color='black')

    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = axs1_twin.get_legend_handles_labels()
    axs1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    # Adjust layout
    plt.tight_layout()
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return cum_vol


##############################################################################
# % Movers
##############################################################################
def movers(df_close, idx, df_close_idx):
    global lookback

    # print(df_idx.tail(lookback))

    # Fill NaN values in the DataFrame before calculating percentage changes
    # df_filled = df_close.ffill()  # FutureWarning error
    # df_filled = df_close.copy()
    # df_filled.ffill(inplace=True)
    # df_close.ffill().pct_change
    # Define the conditions
    c4plus = df_close.ffill().pct_change() >= 0.04
    c4minus = df_close.ffill().pct_change() <= -0.04
    c25_3plus = df_close.ffill().pct_change(periods=63) >= 0.25
    c25_3minus = df_close.ffill().pct_change(periods=63) <= -0.25
    c25_1plus = df_close.ffill().pct_change(periods=21) >= 0.25
    c25_1minus = df_close.ffill().pct_change(periods=21) <= -0.25
    c50_1plus = df_close.ffill().pct_change(periods=21) >= 0.50
    c50_1minus = df_close.ffill().pct_change(periods=21) <= -0.50
    c13_34plus = df_close.ffill().pct_change(periods=34) >= 0.13
    c13_34minus = df_close.ffill().pct_change(periods=34) <= -0.13

    """# Define the conditions
    c4plus = df_filled.pct_change(fill_method=None) >= 0.04
    c4minus = df_filled.pct_change(fill_method=None) <= -0.04
    c25_3plus = df_filled.pct_change(periods=63) >= 0.25
    c25_3minus = df_filled.pct_change(periods=63) <= -0.25
    c25_1plus = df_filled.pct_change(periods=21) >= 0.25
    c25_1minus = df_filled.pct_change(periods=21) <= -0.25
    c50_1plus = df_filled.pct_change(periods=21) >= 0.50
    c50_1minus = df_filled.pct_change(periods=21) <= -0.50
    c13_34plus = df_filled.pct_change(periods=34) >= 0.13
    c13_34minus = df_filled.pct_change(periods=34) <= -0.13"""

    # Create the new dataframe
    movers_df = pd.concat([c4plus, c4minus,
                           c25_3plus, c25_3minus,
                           c25_1plus, c25_1minus,
                           c50_1plus, c50_1minus,
                           c13_34plus, c13_34minus
                           ],
                          keys=['>4%1d', '<4%1d',
                                '>25%Q', '<25%Q', '>25%M', '<25%M',
                                '>50%M', '<50%M',
                                '>13%34d', '<13%34d'], axis=1
                          )

    movers_df = movers_df.astype(int)

    # Group by the first level of columns and sum along the rows
    breadth_df_summed = movers_df.T.groupby(level=0, sort=False).sum().T.astype(int)
    # For 5 and 10 day ratios
    c4_df = breadth_df_summed[['>4%1d', '<4%1d']]
    c13_df = breadth_df_summed[['>13%34d', '<13%34d']]

    #################################################
    # Short term movers
    #################################################

    b_plus = df_close.ffill().pct_change(periods=2) >= 0.06
    b_minus = df_close.ffill().pct_change(periods=2) <= -0.06
    c_plus = df_close.ffill().pct_change(periods=3) >= 0.07
    c_minus = df_close.ffill().pct_change(periods=3) <= -0.07
    d_plus = df_close.ffill().pct_change(periods=4) >= 0.08
    d_minus = df_close.ffill().pct_change(periods=4) <= -0.08
    e_plus = df_close.ffill().pct_change(periods=5) >= 0.09
    e_minus = df_close.ffill().pct_change(periods=5) <= -0.09
    f_plus = df_close.ffill().pct_change(periods=6) >= 0.1
    f_minus = df_close.ffill().pct_change(periods=6) <= -0.1
    g_plus = df_close.ffill().pct_change(periods=7) >= 0.11
    g_minus = df_close.ffill().pct_change(periods=7) <= -0.11
    h_plus = df_close.ffill().pct_change(periods=8) >= 0.12
    h_minus = df_close.ffill().pct_change(periods=8) <= -0.12

    """b_plus = df_filled.pct_change(periods=2) >= 0.06
    b_minus = df_filled.pct_change(periods=2) <= -0.06
    c_plus = df_filled.pct_change(periods=3) >= 0.07
    c_minus = df_filled.pct_change(periods=3) <= -0.07
    d_plus = df_filled.pct_change(periods=4) >= 0.08
    d_minus = df_filled.pct_change(periods=4) <= -0.08
    e_plus = df_filled.pct_change(periods=5) >= 0.09
    e_minus = df_filled.pct_change(periods=5) <= -0.09
    f_plus = df_filled.pct_change(periods=6) >= 0.1
    f_minus = df_filled.pct_change(periods=6) <= -0.1
    g_plus = df_filled.pct_change(periods=7) >= 0.11
    g_minus = df_filled.pct_change(periods=7) <= -0.11
    h_plus = df_filled.pct_change(periods=8) >= 0.12
    h_minus = df_filled.pct_change(periods=8) <= -0.12"""

    # Create the new dataframe
    st_movers_df = pd.concat([c4plus, c4minus,
                              b_plus, b_minus,
                              c_plus, c_minus,
                              d_plus, d_minus,
                              e_plus, e_minus,
                              f_plus, f_minus,
                              g_plus, g_minus,
                              h_plus, h_minus
                              ],
                             keys=['>4%D', '<4%D',
                                   '>6%2D', '<6%2D',
                                   '>7%3D', '<7%3D',
                                   '>8%4D', '<8%4D',
                                   '>9%5D', '<9%5D',
                                   '>10%6D', '<10%6D',
                                   '>11%7D', '<11%7D',
                                   '>12%8D', '<12%8D'
                                   ], axis=1
                             )

    st_movers_df = st_movers_df.astype(int)

    # Group by the first level of columns and sum along the rows
    st_breadth_df_summed = st_movers_df.T.groupby(level=0, sort=False).sum().T.astype(int)

    #############################################################################
    # PLOTTING
    #############################################################################

    p = breadth_df_summed.tail(lookback)
    st_p = st_breadth_df_summed.tail(lookback)

    p_1 = p[['>4%1d', '>25%Q', '>25%M', '>50%M', '>13%34d']]
    p1 = p_1.reset_index().rename(columns={'index': 'Date'})

    p_2 = p[['<4%1d', '<25%Q', '<25%M', '<50%M', '<13%34d']]
    p2 = p_2.reset_index().rename(columns={'index': 'Date'})

    st_p_1 = st_p[['>4%D', '>6%2D', '>7%3D', '>8%4D', '>9%5D', '>10%6D', '>11%7D', '>12%8D']]
    st_p1 = st_p_1.reset_index().rename(columns={'index': 'Date'})

    st_p_2 = st_p[['<4%D', '<6%2D', '<7%3D', '<8%4D', '<9%5D', '<10%6D', '<11%7D', '<12%8D']]
    st_p2 = st_p_2.reset_index().rename(columns={'index': 'Date'})

    # Setup x labels for the 4 plots. Use p_1 because p1 has reset the index
    # Assuming p_1.index contains date values in some format, convert p_1.index to DatetimeIndex
    labels = p_1.copy()
    labels.index = pd.to_datetime(labels.index)
    # print(f"p1.index type = {type(p1.index)}")
    date_labels = labels.index.strftime("%d/%m/%y").tolist()

    pidx = df_close_idx.tail(lookback)

    # Create subplots
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(17, 24))

    #############################################################################
    # Top subplot Long term POSITIVE MOVERS
    #############################################################################

    # Create a stacked bar chart
    bottom = None
    to_plot = p1.columns[1:]
    # for col in p1.columns[1:]:
    for col in to_plot:
        axs[0].bar(p1.index, p1[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = p1[col]
        else:
            bottom += p1[col]

    axs[0].set_xticks(p1.index[::xlabel_separation])
    axs[0].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axes
    axs[0].set_ylabel('Plus % movers')
    # Add title for plot
    axs[0].set_title(f"{idx} - Positive Movers")

    # Creating the second y-axis on the right
    axs0_twin = axs[0].twinx()
    axs0_twin.fill_between(p1.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs0_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=1, zorder=-1)
    axs0_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs0_twin.set_ylabel(idx, color='black')

    # Combine legends for ax and ax_twin into a single legend
    lines, labels = axs[0].get_legend_handles_labels()
    lines_twin, labels_twin = axs0_twin.get_legend_handles_labels()
    axs[0].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    #############################################################################
    # 2nd subplot LT NEGATIVE MOVERS
    #############################################################################

    # Create a stacked bar chart
    bottom = None
    to_plot = p2.columns[1:]

    for col in to_plot:
        # for col in p2.columns[1:]:
        axs[1].bar(p2.index, -p2[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = -p2[col]
        else:
            bottom += -p2[col]

    # Adding the x-axis with dates every 5
    axs[1].set_xticks(p2.index[::xlabel_separation])
    axs[1].set_xticklabels(date_labels[::xlabel_separation], rotation=45)
    # Adding labels for both y-axes
    axs[1].set_ylabel('Negative % Movers')
    # Add title for plot
    axs[1].set_title(f"{idx} - Negative Movers")

    # Creating the second y-axis on the right
    axs1_twin = axs[1].twinx()
    axs1_twin.fill_between(p1.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs1_twin.plot(p2.index, pidx, 'black', label=idx, linewidth=1, zorder=-1)
    axs1_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs1_twin.set_ylabel(idx, color='black')

    # Combine legends for ax and ax_twin into a single legend
    lines, labels = axs[1].get_legend_handles_labels()
    lines_twin, labels_twin = axs1_twin.get_legend_handles_labels()
    axs[1].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    #############################################################################
    # 2nd subplot Short term POSITIVE MOVERS
    #############################################################################

    # Create a stacked bar chart
    bottom = None
    to_plot = st_p1.columns[1:]
    # for col in p1.columns[1:]:
    for col in to_plot:
        axs[2].bar(st_p1.index, st_p1[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = st_p1[col]
        else:
            bottom += st_p1[col]

    axs[2].set_xticks(st_p1.index[::xlabel_separation])
    axs[2].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axes
    axs[2].set_ylabel('Short Term Plus % movers')
    # Add title for plot
    axs[2].set_title(f"{idx} - Short Term Positive Movers")

    # Creating the second y-axis on the right
    axs2_twin = axs[2].twinx()
    axs2_twin.fill_between(st_p1.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs2_twin.plot(st_p1.index, pidx, 'black', label=idx, linewidth=1, zorder=-1)
    axs2_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs2_twin.set_ylabel(idx, color='black')

    # Combine legends for ax and ax_twin into a single legend
    lines, labels = axs[2].get_legend_handles_labels()
    lines_twin, labels_twin = axs2_twin.get_legend_handles_labels()
    axs[2].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    #############################################################################
    # Bottom subplot Short Term NEGATIVE MOVERS
    #############################################################################

    # Create a stacked bar chart
    bottom = None
    to_plot = st_p2.columns[1:]

    for col in to_plot:
        # for col in p2.columns[1:]:
        axs[3].bar(st_p2.index, -st_p2[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = -st_p2[col]
        else:
            bottom += -st_p2[col]

    # Adding the x-axis with dates every 5
    axs[3].set_xticks(st_p2.index[::xlabel_separation])
    axs[3].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axes
    axs[3].set_ylabel('Short Term Negative % Movers')
    # Add title for plot
    axs[3].set_title(f"{idx} - Short Term Negative Movers")

    # Creating the second y-axis on the right
    axs3_twin = axs[3].twinx()
    axs3_twin.fill_between(st_p1.index, 0, pidx, color='lightgrey', alpha=0.3, label=idx, zorder=-1)  # Light grey area
    axs3_twin.plot(st_p2.index, pidx, 'black', label=idx, linewidth=1, zorder=-1)
    axs3_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs3_twin.set_ylabel(idx, color='black')

    # Combine legends for ax and ax_twin into a single legend
    lines, labels = axs[3].get_legend_handles_labels()
    lines_twin, labels_twin = axs3_twin.get_legend_handles_labels()
    axs[3].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    # plt.savefig(f'{plots_folder}/{idx}_pct_movers.jpg')
    plt.tight_layout()
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return movers_df, st_movers_df, breadth_df_summed, st_breadth_df_summed, c4_df, c13_df


##########################################################################
# 5, 10, 13 day Ratio
##########################################################################
def ratios(df4, df13, idx, df_close_idx):
    global lookback

    fdr = df4.rolling(window=5).sum().dropna()
    ratiofive = fdr.iloc[:, 0] / fdr.iloc[:, 1]
    tdr = df4.rolling(window=10).sum().dropna()
    ratioten = tdr.iloc[:, 0] / tdr.iloc[:, 1]
    tftdr = df13.iloc[:, 0] - df13.iloc[:, 1]

    # Create breadth_ratios DataFrame with new column names
    breadth_ratios = (pd.concat([ratiofive, ratioten, tftdr], axis=1)).round(2)
    breadth_ratios.columns = ['ratio5', 'ratio10', 'tftdr']
    # print(breadth_ratios.tail(20))

    #############################################################################
    # PLOTTING
    #############################################################################

    p = breadth_ratios.reset_index().rename(columns={'index': 'Date'})
    p1 = p.tail(lookback)
    date_labels = p1['Date'].dt.strftime("%d/%m/%y").tolist()
    pidx = df_close_idx.tail(lookback)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(17, 12))

    #############################################################################
    # Plotting 5 and 10 day breadth_ratios on the primary y-axis of upper plot
    #############################################################################

    axs[0].plot(p1.index, p1['ratio5'], label='5-day breadth ratio', color='blue', zorder=0)
    axs[0].plot(p1.index, p1['ratio10'], label='10-day breadth ratio', color='green', zorder=0)

    # Adding the x-axis with dates every 5
    axs[0].set_xticks(p1.index[::xlabel_separation])
    axs[0].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axs[]es
    axs[0].set_ylabel('5 and 10 day ratios')
    # Add title for plot
    axs[0].set_title(f"{idx} - Breadth Ratios (No. 4% daily B-Outs in 5 (or 10) days / number of 4% daily B-Downs in "
                     f"5 (or 10) days")

    # Creating the second y-axs[]is on the right
    axs0_twin = axs[0].twinx()
    axs0_twin.fill_between(p1.index, 0, pidx, color='lightgrey', alpha=0.1, label=idx, zorder=1)  # Light grey area
    axs0_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=1, zorder=1)
    axs0_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs0_twin.set_ylabel(idx, color='black')

    # Combine legends for axs[] and axs[]_twin into a single legend
    lines, labels = axs[0].get_legend_handles_labels()
    lines_twin, labels_twin = axs0_twin.get_legend_handles_labels()
    axs[0].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    #############################################################################
    # Plotting 13/34D ratio on the primary y-axis of lower plot
    #############################################################################

    # Creating the second y-axs[]is on the right
    axs1_twin = axs[1].twinx()
    axs1_twin.fill_between(p1.index, 0, pidx, color='lightgrey', alpha=0.1, label=idx, zorder=1)  # Light grey area
    axs1_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=1, zorder=1)
    axs1_twin.set_ylim(bottom=min(pidx), top=max(pidx))
    axs1_twin.set_ylabel(idx, color='black')

    # Adding the x-axis with dates
    axs[1].plot(p1.index, p1['tftdr'], label='13/34D', color='red', zorder=0)
    axs[1].set_xticks(p1.index[::xlabel_separation])
    axs[1].set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Adding labels for both y-axs[]es
    axs[1].set_ylabel('13/34 ratio')
    # Add title for plot
    axs[1].set_title(f"{idx} - No. tickers up 13% in 34 days minus No. tickers down 13% in 34 days")

    # Add a horizontal line at 0
    axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)

    # Combine legends for axs[] and axs[]_twin into a single legend
    lines, labels = axs[1].get_legend_handles_labels()
    lines_twin, labels_twin = axs1_twin.get_legend_handles_labels()
    axs[1].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    # plt.savefig(f'{plots_folder}/{idx}_breadth_ratios.jpg')
    plt.tight_layout(pad=1.0)
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return breadth_ratios


#########################################################################
# Plotting the yahoo indexes normalised
#########################################################################
def plot_normalized_indexes(mkt_dict, idx):
    global lookback
    global data_folder

    # Extract 'idx_code' values from the filtered dictionary
    idx_code_list = [value['idx_code'] for value in mkt_dict.values()]

    # Initialize combined_df before the loop
    combined_df = pd.DataFrame()

    # Iterate through all_idx_codes
    for code in idx_code_list:
        # Read the CSV file for the current idx_code, only selecting 'Adj Close' column
        csv_file = f"{data_folder}/INDEX_{code}.csv"
        df = pd.read_csv(csv_file, usecols=['Date', 'Adj Close'], index_col='Date', parse_dates=True)

        # Rename the 'Adj Close' column to the idx_code
        adj_close_col = df['Adj Close'].rename(code)

        # Concatenate the 'Adj Close' column to the combined DataFrame
        combined_df = pd.concat([combined_df, adj_close_col], axis=1, sort=True)
        combined_df.index = pd.to_datetime(combined_df.index)
        combined_df.rename(columns={'Adj Close': idx}, inplace=True)

    # print('Combined index df:')
    # print(combined_df.tail(10))

    # Plot the normalized data
    p = combined_df.tail(lookback)
    # Remove nan (bitcoin)
    pp = p.dropna()
    # Rebase p
    p_1 = pp / pp.iloc[0]
    date_labels = p_1.index.strftime("%d/%m/%y").tolist()
    p1 = p_1.reset_index(drop=True)

    # Checks cos of "ValueError: Axis limits cannot be NaN or Inf"
    # print(f'Check {idx_code} for NaN {p1.isnull().sum().sum()}')  # Check for NaN
    # print(f'Check {idx_code} for Inf {np.isinf(p1).sum().sum()}')  # Check for Inf

    # Calculate the maximum and minimum values in the DataFrame columns
    # max_value = p1.max().max()
    # min_value = p1.min().min()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(17, 12))

    # Define a set of distinct colors and linestyles
    lines = ['-', '--', '-.', ':']
    # Create a cycle iterator for linestyles
    line_styles = cycle(lines)

    # Iterate through all columns and plot with distinct colors and linestyles
    for code in p1.columns:
        line = next(line_styles)
        ax1.plot(p1.index, p1[code], label=code, linestyle=line)

    # Plot closing price on the first y-axis
    ax1.set_title(f"Comparing Indices")
    ax1.set_ylabel('Normalised Close', color='black')
    ax1.tick_params(axis='y')
    # Set y-axis limits based on the calculated max and min values

    # ax1.set_ylim(bottom=min_value, top=max_value)
    # print(p1)
    if not p1.empty:
        max_value = p1.max().max()
        min_value = p1.min().min()
        # Set y-axis limits based on the calculated max and min values
        ax1.set_ylim(bottom=min_value, top=max_value)
    else:
        print(f"{idx_code} dataFrame is empty. Skipping setting y-axis limits.")

    # Set x-ticks and x-tick labels for first y-axis
    # ax1.set_xlabel('Date', color='blue')
    ax1.set_xticks(p1.index[::xlabel_separation])
    ax1.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Combine legends for both subplots
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')

    pdf.savefig()
    plt.close()


#########################################################################
# Plotting the yahoo indexes normalised
#########################################################################
def plot_normalized_indexes_minus_btc(mkt_dict, idx):
    global lookback
    global data_folder

    # Extract 'idx_code' values from the filtered dictionary
    idx_code_list = [value['idx_code'] for value in mkt_dict.values() if value['idx_code'] != 'BTC-USD']

    # Initialize combined_df before the loop
    combined_df = pd.DataFrame()

    # Iterate through all_idx_codes
    for code in idx_code_list:
        # Read the CSV file for the current idx_code, only selecting 'Adj Close' column
        csv_file = f"{data_folder}/INDEX_{code}.csv"
        df = pd.read_csv(csv_file, usecols=['Date', 'Adj Close'], index_col='Date', parse_dates=True)

        # Rename the 'Adj Close' column to the idx_code
        adj_close_col = df['Adj Close'].rename(code)

        # Concatenate the 'Adj Close' column to the combined DataFrame
        combined_df = pd.concat([combined_df, adj_close_col], axis=1, sort=True)
        combined_df.index = pd.to_datetime(combined_df.index)
        combined_df.rename(columns={'Adj Close': idx}, inplace=True)

    # print('Combined index df:')
    # print(combined_df.tail(10))

    # Plot the normalized data
    p = combined_df.tail(lookback)
    # Remove nan (bitcoin)
    pp = p.dropna()
    # Rebase p
    p_1 = pp / pp.iloc[0]
    date_labels = p_1.index.strftime("%d/%m/%y").tolist()
    p1 = p_1.reset_index(drop=True)

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(17, 12))

    # Define a set of distinct colors and linestyles
    lines = ['-', '--', '-.', ':']
    # Create a cycle iterator for linestyles
    line_styles = cycle(lines)

    # Iterate through all columns and plot with distinct colors and linestyles
    for code in p1.columns:
        line = next(line_styles)
        ax1.plot(p1.index, p1[code], label=code, linestyle=line)

    # Plot closing price on the first y-axis
    ax1.set_title(f"Comparing Indices (no BTC)")
    ax1.set_ylabel('Normalised Close', color='black')
    ax1.tick_params(axis='y')
    # Set y-axis limits based on the calculated max and min values

    # ax1.set_ylim(bottom=min_value, top=max_value)
    # print(p1)
    if not p1.empty:
        max_value = p1.max().max()
        min_value = p1.min().min()
        # Set y-axis limits based on the calculated max and min values
        ax1.set_ylim(bottom=min_value, top=max_value)
    else:
        print(f"{idx_code} dataFrame is empty. Skipping setting y-axis limits.")

    # Set x-ticks and x-tick labels for first y-axis
    # ax1.set_xlabel('Date', color='blue')
    ax1.set_xticks(p1.index[::xlabel_separation])
    ax1.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Combine legends for both subplots
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')

    pdf.savefig()
    plt.close()


##########################################################################
# Plot a table
##########################################################################
def plot_table_and_score(csv, plot_title, idx):

    global sample_size_for_ranking

    rows_on_page = 54

    df = pd.read_csv(csv, index_col=0, header=0)
    # Round all numbers in the DataFrame to two decimal places
    df = df.round(2)

    # Remove nan and inf
    max_val = df.apply(lambda df_col: df_col[df_col != np.inf].max())

    # Replace inf values with the corresponding max value in each column
    for col in df.columns:
        df[col].replace([np.inf], max_val[col], inplace=True)
    # Replace remaining NaN values with 0
    df.fillna(0, inplace=True)

    # Select the tail size to use to calculate ranking.
    df_for_ranking = df.tail(sample_size_for_ranking)
    # df = df.iloc[::-1]

    # Calculate percentile ranks based on the entire DataFrame
    rank_df = df_for_ranking.rank(pct=True)
    # print('Rank df shape:')
    # print(df.shape)

    # Sum the percentile values for each row
    sum_percentiles = rank_df.sum(axis=1)

    ##############################################
    # Plot table to page size (54 rows)
    ##############################################
    df_to_plot = df_for_ranking.tail(rows_on_page)

    fig, ax = plt.subplots(figsize=(17, 12))
    plt.title(f'{plot_title} Heat map of percentiles', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Use the calculated ranks for the entire DataFrame
    cell_colors = plt.cm.RdYlGn(rank_df.tail(rows_on_page).values.reshape(df_to_plot.shape[0], df_to_plot.shape[1]))

    tbl = ax.table(cellText=df_to_plot.values,
                   colLabels=df_to_plot.columns,
                   rowLabels=df_to_plot.index,
                   loc='center',
                   cellColours=cell_colors)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1)

    pdf.savefig()
    plt.close()

    ##############################################
    # Plot score graph
    ##############################################
    df['Score'] = sum_percentiles
    p1 = df.tail(lookback)

    # Ensure p1.index is a DatetimeIndex before using strftime
    if not isinstance(p1.index, pd.DatetimeIndex):
        # Convert index to DatetimeIndex
        # print(f'df index is: {type(p1.index)}')
        p1.index = pd.to_datetime(p1.index)

    date_labels = p1.index.strftime("%d/%m/%y").tolist()
    # print(f'df index is: {type(p1.index)}')

    p1 = p1.reset_index(drop=True)  # df now has sequential index. Must use DateLabels as x-axis labels

    fig, ax1 = plt.subplots(figsize=(17, 6))

    # Plotting sum_percentiles on the left y-axis
    ax1.plot(p1.index, p1['Score'], label='Percentile Score', color='lightblue', linestyle='-', linewidth=1)
    ax1.set_ylabel('Score (Sum of Percentiles)', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='black')

    # Calculate Exponential Moving Average (EMA) of Score with a span of 5
    ema_5 = p1['Score'].ewm(span=5, adjust=False).mean()

    # Plotting EMA on the left y-axis
    ax1.plot(p1.index, ema_5, label='Score 5-Day EMA', color='blue', linestyle='--', linewidth=1)

    # Adding Adj_Close on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(p1.index, p1['Adj Close'], color='black', linestyle='-', linewidth=1)
    ax2.fill_between(p1.index, 0, p1['Adj Close'], color='lightgrey', alpha=0.1, label=idx, zorder=1)
    ax2.set_ylabel('Index Close', color='black', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(bottom=min(p1['Adj Close']), top=max(p1['Adj Close']))

    # Set x-ticks and x-tick labels for the second y-axis
    ax1.set_xticks(p1.index[::xlabel_separation])
    ax1.set_xticklabels(date_labels[::xlabel_separation], rotation=45)

    # Combine legends for both subplots
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')
    lines_twin, labels_twin = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    # Title for the plot
    plt.suptitle(f'{idx} {plot_title} Score (Sum of percentiles)', fontweight='bold')

    # Saving the graph plot to PDF
    pdf.savefig()
    plt.close()


#######################################################
# Get action, market list to use and lookback period
#######################################################
def get_user_inputs():
    """
    Prompts the user for configuration options.
    Returns: A tuple containing (market_list, update_use_download, lookback_period)
    """
    # Setting default values to avoid: Local variable 'xxx' might be referenced before assignment
    market_list = 1
    update_use_download = 1
    lookback_period = 756

    print('What do you want to do?')
    print('Change inputs: 1, Auto-update: 2')
    while True:  # Will execute until finds a <break>
        try:
            action = int(input('Enter your choice (1 or 2): ') or 2)
            if action not in (1, 2):
                print('Invalid choice. Please enter 1 or 2')
                continue

            if action == 1:
                # Work with all indices or just one?
                market_list = get_market_map(yahoo_idx_components_dictionary)
                # Update, use actual data, or download new data?
                update_use_download = get_user_choice()
                # Define how far to look back on graphs and database start/end
                lookback_period = get_lookback()
            else:
                market_list = yahoo_idx_components_dictionary
                update_use_download = 1
                lookback_period = 756  # 252*3yrs (explanation in docstring)

            break  # Exit loop after valid input

        except ValueError:
            print('Invalid input. Enter 1 (manual) or 2 (auto)')

    return market_list, update_use_download, lookback_period


##########################################################################

# -----------------------------MAIN PROGRAM-------------------------------

##########################################################################

# Get user defined inputs
mkt_list, up_use_dl, lookback = get_user_inputs()

from_date = "2000-01-01"
until_date = download_until()
number_xlabels = 50
xlabel_separation = int(lookback/number_xlabels)

# When downloading new, ask for database starting date (from_date). Ask here or it asks for it on each pass
if up_use_dl == 3:
    # When downloading, must specify start date
    # from_date = input("Enter start date (YYYYMMDD): ") or "20000101"
    from_date_input = input("Enter start date (DDMMYYYY): ") or "01012013"
    # Convert the string to a datetime object
    # date_object = datetime.strptime(from_date, "%Y%m%d")
    date_object = datetime.strptime(from_date_input, "%d%m%Y")
    # Format the datetime object as a string in the desired format
    from_date = date_object.strftime("%Y-%m-%d")
    print(f'Download until: {until_date.strftime("%d-%m-%y")}')
    create_databases(mkt_list, from_date, until_date)

# Use existing data
elif up_use_dl == 1:
    # Perform update
    print(f'Update until: {until_date.strftime("%d-%m-%y")}')
    last_date_in_component_csv, component_df, last_date_in_index_csv, index_df = update_databases(mkt_list)


##########################################################################
# -----------------BREADTH AND INDICATORS------------------------
##########################################################################

for nums in mkt_list:
    market_details = mkt_list[nums]
    # Files to be used
    idx_code = market_details['idx_code']
    market_name = market_details['market']
    tickers = market_details['codes_csv']

    if tickers != 'none':
        # pdf_filename = f'{pdf_folder}/{idx_code}_{datetime.today().strftime("%Y-%m-%d")}.pdf'
        pdf_filename = f'{pdf_folder}/{idx_code}.pdf'
        with (PdfPages(pdf_filename) as pdf):
            #  This is required because we are in a new loop and idx/eod_df are undefined
            idx_df = pd.read_csv(f'{data_folder}/INDEX_{idx_code}.csv', index_col=0, parse_dates=True)
            comp_df = pd.read_csv(f'{data_folder}/EOD_{market_name}.csv', header=[0, 1], index_col=0, parse_dates=True)

            # Count how many zeros are in Volume column (ignore last entry which is often zero)
            total_stocks = comp_df.columns.get_level_values(1).nunique()
            original_zeros_count = (comp_df.iloc[:-1]['Volume'] == 0).sum()  # original_zeros_count is a series
            sum_tickers_with_zero_volume = original_zeros_count.astype(bool).sum()
            total_zero_entries = original_zeros_count.sum()

            print(f'{idx_code}: {total_stocks} tickers. {sum_tickers_with_zero_volume} '
                  f'with >= one zero volume entry.')
            print(f'{idx_code}: {total_stocks * len(comp_df)} volume entries.'
                  f' {total_zero_entries} ({(total_zero_entries/(total_stocks * len(comp_df))*100):.1f}%) '
                  f'with zero volume.')

            # Basic plot to get an idea of correct shape
            plot_close_and_volume(idx_df, idx_code)

            # Dataframe with ATH/L, 12MH/L, 3MH/L and 1MH/L
            # high_low_df = highs_and_lows(idx_df, eod_df, time_periods, idx_code)
            high_low_df = highs_and_lows(idx_df, comp_df, time_periods, idx_code)

            # Multiindex dataframe with all tickers and their MA's: 5, 12, 25, 40, 50, 100, 200
            mov_avgs_df = calculate_moving_averages(comp_df['Adj Close'], mas_to_use, 'Close')

            # Multiindex dataframe with all tickers and their Traders Eden EMA's: 8, 80.
            traders_eden_mas = calculate_traders_edens(comp_df['Adj Close'], idx_df['Close'], idx_code)

            # Dataframe of the sum of average difference between MAs and close for whole market
            ma_c_diff_df = difference_close_to_ma(comp_df['Adj Close'], idx_df['Adj Close'], mov_avgs_df, idx_code)

            # Dataframes of number (percentage) of tickers over MA's: 5, 12, 25, 40, 50, 100, 200
            a, b, c = close_over_mas(mov_avgs_df, 'Close', idx_df['Adj Close'], idx_code)
            over_short_mas_pct, over_40ma_pct, over_long_mas_pct = a, b, c

            # Advance/decline line
            adr = advance_decline_ratio(comp_df['Adj Close'], idx_df['Adj Close'], idx_code)

            # Accumulated volume
            acc_vol = accumulated_volume(comp_df['Adj Close'], comp_df['Volume'], idx_code, idx_df['Adj Close'])

            # Breadth: movers and ratios
            mvrs, st_mvrs, mvrs_sum, st_mvrs_sum, fourpc_df, thirteenpc_df = movers(comp_df['Adj Close'], idx_code, idx_df['Adj Close'])
            br_ratios = ratios(fourpc_df, thirteenpc_df, idx_code, idx_df['Adj Close'])

            # Comparing indices
            plot_normalized_indexes(filtered_index_dict, idx_code)
            plot_normalized_indexes_minus_btc(filtered_index_dict, idx_code)

            ##########################################################################
            # Breadth dataframe
            ##########################################################################
            adr_df = pd.DataFrame(adr, columns=[adr.name])
            acc_vol_df = pd.DataFrame(acc_vol, columns=[acc_vol.name])
            all_dfs_df = pd.concat([high_low_df,
                                    over_short_mas_pct,
                                    over_40ma_pct,
                                    over_long_mas_pct,
                                    mvrs, st_mvrs,
                                    mvrs_sum, st_mvrs_sum,
                                    br_ratios,
                                    adr_df, idx_df, acc_vol_df], axis=1)

            # print(all_dfs_df.columns[all_dfs_df.columns.duplicated()])

            ##############################
            # Breadth Monitor Table
            ##############################
            stockbee_df = all_dfs_df[['ratio5', 'ratio10',
                                      '$>MA40',
                                      '>4%1d', '>13%34d', '>25%M', '>50%M', '>25%Q',
                                      '<4%1d', '<13%34d', '<25%M', '<50%M', '<25%Q',
                                      'Adj Close']]

            # Negate columns so that the percentile colours are inverted
            columns_to_negate = ['<4%1d', '<25%Q', '<25%M', '<50%M', '<13%34d']
            for column in columns_to_negate:
                stockbee_df.loc[:, column] = -stockbee_df[column]

            # print(stockbee_df.tail(20))
            stockbee_df.to_csv('stockbee_df.csv')
            plot_table_and_score('stockbee_df.csv', 'Breadth Monitor', idx_code)

            ##############################
            # Highs and Lows Table
            ##############################
            hilo_df = all_dfs_df[['1MH', '3MH', '12MH', 'ATH',
                                  '1ML', '3ML', '12ML', 'ATL',
                                  'ATH-ATL', '12MH-12ML', '3MH-3ML',
                                  'Adj Close']]
            hilo_df.to_csv('hilo_df.csv')
            plot_table_and_score('hilo_df.csv', 'Highs and Lows', idx_code)

            ##############################
            # Short Term Movers Table
            ##############################
            # Create a copy of the DataFrame to ensure you're working with the original DataFrame
            short_term_movers = all_dfs_df[['adv_dec_ratio',
                                            '>4%D', '>6%2D', '>7%3D', '>8%4D', '>9%5D', '>10%6D', '>11%7D', '>12%8D',
                                            '<4%D', '<6%2D', '<7%3D', '<8%4D', '<9%5D', '<10%6D', '<11%7D', '<12%8D',
                                            'Adj Close'
                                            ]].copy()

            # Negate columns so that the percentile colours are inverted
            st_columns_to_negate = ['<4%D', '<6%2D', '<7%3D', '<8%4D', '<9%5D', '<10%6D', '<11%7D', '<12%8D']
            for column in st_columns_to_negate:
                short_term_movers.loc[:, column] = -short_term_movers[column]

            # Rename the column 'adv_dec_ratio' to 'ADR'
            short_term_movers.rename(columns={'adv_dec_ratio': 'ADR'}, inplace=True)

            short_term_movers.to_csv('short_term_movers.csv')
            plot_table_and_score('short_term_movers.csv', 'Short term movers', idx_code)

# Check if the PDF file exists in the folder
pdf_filename = os.path.join(pdf_folder, '^BVSP.pdf')
if os.path.exists(pdf_filename):
    try:
        os.startfile(pdf_filename)  # Open PDF using the default viewer on Windows
    except AttributeError:
        os.system('open "{}"'.format(pdf_filename))  # Open PDF using the default viewer on macOS or Linux
else:
    print("The PDF file '^BVSP.pdf' does not exist in the specified folder.")
