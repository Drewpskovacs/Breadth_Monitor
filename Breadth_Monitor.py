# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:29:34 2024

@author: Andy
"""
import pandas as pd
from datetime import datetime, timedelta
# import os
import sys
import yfinance as yf
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.font_manager as prop
#  from matplotlib.figure import Figure
#  from matplotlib.table import Table
#  from matplotlib.colors import LinearSegmentedColormap

#####################################
# Variables - Setup
#####################################

hoje = datetime.now()  # .strftime("%d-%m-%Y")
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
    3: {'idx_code': '^FTSE', 'market': 'ftse100', 'codes_csv': 'FTSE100.csv'},
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
            user_choice = int(input('Enter your choice (1: update, 2: use as is, or 3: download new): '))
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
            lookback_period = int(input("Days to look back: (510 default)") or 510)
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

    # Remove historic zero volumes in idx_df and replace with previous days volume
    # idx_df['Volume'] = idx_df['Volume'].replace(0, method='ffill')

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
        start_update_on = df.index[-1]
    else:
        # Last row has volume, start update on day after last row
        start_update_on = df.index[-1] + timedelta(days=1)

    # If last day in df = update until date, no need to update
    if until_date.date() == start_update_on.date():
        print('No need to update')
        #  return updated_idx  # GPT says to make sure to return something meaningful

    elif start_update_on.date() < until_date.date():
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

    # all_markets = [key for key in yahoo_idx_components_dictionary.keys()]

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

        update_index_data(i_c, ind_df)

    # Only update component data from SELECTED index or ALL
    for number in market_list:
        mkt_details = market_list[number]
        # Files to be used
        # i_c = mkt_details['idx_code']
        m_n = mkt_details['market']
        tikrs = mkt_details['codes_csv']

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

            update_components_data(m_n, com_df)

    # Not sure this is required. Variables used inside function
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
    p1 = df_idx.tail(lookback)

    # Create a figure and axis
    graph_name = f'{idx} - Close and Volume'
    fig, ax1 = plt.subplots(figsize=(17, 7))
    # Assuming 'Date' is the index of the DataFrame p1
    date_labels = p1.index.strftime("%d/%m/%y").tolist()

    # Plot closing price on the first y-axis
    ax1.set_ylabel('Close', color='blue')
    ax1.plot(p1.index, p1['Adj Close'])
    ax1.tick_params(axis='y')

    # Set x-ticks and x-tick labels for first y-axis
    # ax1.set_xlabel('Date', color='blue')
    ax1.set_xticks(p1.index[::5])
    ax1.set_xticklabels(date_labels[::5], rotation=45)

    # Add title as text to the plot using data coordinates
    ax1.text(p1.index[0], p1['Adj Close'].max(), graph_name, color='black', ha='left', va='bottom')

    # Create a second y-axis for volume
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volume', color='red')
    ax2.plot(p1.index, p1['Volume'], color='red')
    ax2.tick_params(axis='y')

    # Set x-ticks and x-tick labels for the second y-axis
    ax2.set_xticks(p1.index[::5])
    ax2.set_xticklabels(date_labels[::5], rotation=45)

    # Display the plot
    plt.title(f'{idx} - {hoje} - Lookback: {lookback} days', fontsize=16, fontweight='bold')
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

    # Display the resulting DataFrame df2
    # print(df2)

    # Group by level 0 and sum within each group
    hl_df = df2.T.groupby(level=0, sort=False).sum().T

    #############################################################################
    # PLOTTING
    #############################################################################
    # Make new df to extract dates for plotting

    p = hl_df.tail(lookback)
    p1 = p.reset_index().rename(columns={'index': 'Date'})
    pidx = idx_c.tail(lookback)

    fig, ax = plt.subplots(figsize=(17, 7))

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
        plt.bar(p1.index, p1[label], label=label, color=colors[label])

    ax.set_ylabel('Highs and Lows')

    date_labels = p1['Date'].dt.strftime("%d/%m/%y").tolist()
    ax.set_xticks(p1.index[::5])
    ax.set_xticklabels(date_labels[::5], rotation=45)

    # Add title and legend
    plt.title(f"{idx} - Highs and Lows")
    # plt.legend(labels=p1.columns, loc='upper left')  # Use column names for legend labels

    # Add a horizontal dotted line to show where bottoms might be
    ax.axhline(y=0, color='black', linestyle='--')

    # Add index to other axis
    ax3 = ax.twinx()
    ax3.plot(p1.index, pidx, 'black', label=idx, linewidth=2)
    ax3.set_ylabel(idx, color='black')
    ax3.set_xticks(p1.index[::5])
    ax3.set_xticklabels(date_labels[::5], rotation=45)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')

    # plt.savefig(f'{plots_folder}/{idx}_highs_lows.jpg')
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return hl_df


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
    p1 = diff_df.tail(lookback)
    date_labels = p1.index.strftime("%d/%m/%y").tolist()
    pidx = df_close_idx.tail(lookback)

    # Create subplots
    fig, ax = plt.subplots(figsize=(17, 7))

    #################################################################################

    # Create a stacked bar chart
    bottom = None
    # for col in p1.columns[1:]:
    for col in p1.columns:
        ax.bar(p1.index, p1[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = p1[col]
        else:
            bottom += p1[col]

    # Set x-ticks and x-tick labels for first y-axis
    ax.set_xticks(p1.index[::5])
    ax.set_xticklabels(date_labels[::5], rotation=45)

    # Creating the second y-axis on the right
    ax_twin = ax.twinx()
    ax_twin.plot(p1.index, pidx, 'black', label=idx, linewidth=2)

    # Adding labels for both y-axes
    ax.set_ylabel('Avg % between MA and Close')
    ax_twin.set_ylabel(idx, color='black')

    # Combine legends for ax and ax_twin into a single legend
    lines, labels = ax.get_legend_handles_labels()
    lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
    ax.legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    # Add title for plot
    ax.set_title(f"{idx} - Avg % between MA and Close")

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
    over_low_mas_sum = over_low_mas.T.groupby(level=0, sort=False).sum().T
    over_low_mas_sum_pct = (over_low_mas_sum / total_stocks * 100).round(1)
    over_mid_ma_sum = over_mid_ma.T.groupby(level=0, sort=False).sum().T
    over_mid_ma_sum_pct = (over_mid_ma_sum / total_stocks * 100).round(1)
    over_high_mas_sum = over_high_mas.T.groupby(level=0, sort=False).sum().T
    over_high_mas_sum_pct = (over_high_mas_sum / total_stocks * 100).round(1)

    # Change column names
    over_low_mas_sum_pct.columns = ['$>MA5', '$>MA12', '$>MA25']
    over_mid_ma_sum_pct.columns = ['$>MA40']
    over_high_mas_sum_pct.columns = ['$>MA50', '$>MA100', '$>MA200']

    #############################################################################
    # PLOTTING
    #############################################################################

    p1 = over_low_mas_sum_pct.tail(lookback)
    p2 = over_mid_ma_sum_pct.tail(lookback)
    p3 = over_high_mas_sum_pct.tail(lookback)
    pidx = df_close_idx.tail(lookback)

    # Create subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(17, 9))
    date_labels = p1.index.strftime("%d/%m/%y").tolist()

    # Plot > low MAs on top subplot
    #################################################################################

    axs[0].plot(p1.index, p1, linewidth=1, label=p1.columns)
    # idx on twin y-axis on right
    axs0_twin = axs[0].twinx()
    axs0_twin.plot(pidx.index, pidx, linewidth=2, color='black', label=idx)

    # Set x-ticks and x-tick labels for first y-axis
    axs[0].set_xticks(p1.index[::5])
    axs[0].set_xticklabels(date_labels[::5], rotation=45)

    # Adding labels for both y-axes
    axs[0].set_ylabel('% stocks above low MAs')
    axs0_twin.set_ylabel(idx, color='black')

    # Combine legends for both subplots
    lines0, labels0 = axs[0].get_legend_handles_labels()
    lines0_twin, labels0_twin = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper left')

    # Add title for plot
    axs[0].set_title(f"{idx} - % stocks above low MAs")

    # Plot T2108 vs Close on middle subplot
    #################################################################################
    axs[1].plot(p2.index, p2, color='b', alpha=0.7, label='T2108')
    axs[1].set_ylabel('T2108-(%>40MA)', color='black')

    # Set x-ticks and x-tick labels for first y-axis
    axs[1].set_xticks(p1.index[::5])
    axs[1].set_xticklabels(date_labels[::5], rotation=45)

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
    axs[1].set_title(f"{idx} - T2108 (>MA40)")

    # Plot > high MAs on bottom subplot
    #################################################################################
    axs[2].plot(p3.index, p3, linewidth=1, label=p3.columns)
    # idx on twin y-axis on right
    axs2_twin = axs[2].twinx()
    axs2_twin.plot(pidx.index, pidx, linewidth=2, color='black', label=idx)

    # Set x-ticks and x-tick labels for first y-axis
    axs[2].set_xticks(p1.index[::5])
    axs[2].set_xticklabels(date_labels[::5], rotation=45)

    # Adding labels for both y-axes
    axs[2].set_ylabel('% stocks above high MAs')
    axs2_twin.set_ylabel(idx, color='black')

    # Combine legends for both subplots
    lines2, labels2 = axs[2].get_legend_handles_labels()
    lines2_twin, labels2_twin = axs2_twin.get_legend_handles_labels()
    axs2_twin.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc='upper left')

    # Add title for plot
    axs[2].set_title(f"{idx} - % stocks above high MAs")

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
    adv_dec_cum_diff = adv_dec_diff.cumsum()
    # Rename the columns
    adv_dec_cum_diff = adv_dec_cum_diff.rename("adv_dec_cum_diff")
    # print(adv_dec_cum_diff.tail(20))

    # Create a DataFrame
    data = {'Advancing': advancing_stocks, 'Declining': declining_stocks}
    df = pd.DataFrame(data)

    # Calculate the 19-day and 39-day EMAs for advancing and declining stocks
    df['Advancing_EMA_19'] = df['Advancing'].ewm(span=19, adjust=False).mean()
    df['Declining_EMA_19'] = df['Declining'].ewm(span=19, adjust=False).mean()
    df['Advancing_EMA_39'] = df['Advancing'].ewm(span=39, adjust=False).mean()
    df['Declining_EMA_39'] = df['Declining'].ewm(span=39, adjust=False).mean()

    # Calculate the McClellan Oscillator
    df['McClellan_Oscillator'] = df['Advancing_EMA_19'] - df['Declining_EMA_39']

    # Display the DataFrame with the McClellan Oscillator
    # print(df)

    #############################################################################
    # PLOTTING
    #############################################################################
    p1 = adv_dec_ratio.tail(lookback)
    # p2 = dec_adv_ratio.tail(lookback)
    # p3 = adv_dec_breadth.tail(lookback)
    # p4 = adv_dec_daily.tail(lookback)
    p5 = adv_dec_cum_diff.tail(lookback)
    p6 = df['McClellan_Oscillator'].tail(lookback)

    pidx = df_close_idx.tail(lookback)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(17, 9))
    date_labels = p1.index.strftime("%d/%m/%y").tolist()

    ###################
    # Plot A/D ratio vs Close
    ###################
    axs[0].plot(p1.index, p1, color='b', alpha=0.7, label='Advance/Decline Ratio')
    # ax.plot(p1.index, p2, color='lightblue', alpha=0.7, label='Decline/Advance Ratio')
    # ax.plot(p1.index, p3, color='orange', alpha=0.7, label='Advance/Total Breadth')
    # ax.plot(p1.index, p4, color='green', alpha=0.7, label='Decline/Advance Diff Daily')
    # ax.plot(p1.index, p5, color='yellow', alpha=0.7, label='Cumulative Decline/Advance')

    axs[0].set_xticks(p1.index[::5])
    axs[0].set_xticklabels(date_labels[::5], rotation=45)

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
    axs[1].plot(p5.index, p5, color='b', alpha=0.7, label='Cumulative Advance/Decline')
    axs[1].set_ylabel('Cumulative Adv - Dec', color='black')
    axs[1].set_title(f"{idx} - Cumulative Advances minus Declines")

    # Set y-axis limits to the range of accumulated volume checking for NaN or Inf
    valid_values = p5[~np.isnan(p5) & ~np.isinf(p5)]
    if len(valid_values) > 0:
        axs[1].set_ylim(bottom=min(valid_values), top=max(valid_values))

    # Add a horizontal line at 0
    axs[1].axhline(y=0, color='blue', linestyle='--', linewidth=1.5)

    axs[1].set_xticks(p1.index[::5])
    axs[1].set_xticklabels(date_labels[::5], rotation=45)

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
    axs[2].plot(p6.index, p6, color='b', alpha=0.7, label='McClellan Oscillator')
    axs[2].set_ylabel('McClellan Oscillator', color='black')
    axs[2].set_title(f'{idx} - McClellan Oscillator')

    ''' # Set y-axis limits to the range of accumulated volume checking for NaN or Inf
    valid_values = p5[~np.isnan(p5) & ~np.isinf(p5)]
    if len(valid_values) > 0:
        axs[1].set_ylim(bottom=min(valid_values), top=max(valid_values))'''

    # Add a horizontal line at 0
    axs[2].axhline(y=0, color='blue', linestyle='--', linewidth=1.5)

    axs[2].set_xticks(p6.index[::5])
    axs[2].set_xticklabels(date_labels[::5], rotation=45)

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

    '''print('cum_vol_df')
    print(type(cum_vol_df))
    print(cum_vol_df.columns)
    print(cum_vol_df.index)
    print(cum_vol_df.tail(3))'''

    # Calculate the percent difference between consecutive entries in cum_vol_df
    cum_vol_pct_diff = cum_vol_df['CumVol'].pct_change().mul(100)  # This is a series
    '''print('cum_vol_pct_diff')
    print(type(cum_vol_pct_diff))
    print(cum_vol_pct_diff.tail(3))'''

    # Calculate the percent difference between consecutive entries in df_close
    close_pct_diff = df_close_idx.pct_change().mul(100)  # This is a series
    '''print('close_pct_diff')
    print(type(close_pct_diff))
    print(close_pct_diff.tail(3))'''

    diff = (close_pct_diff - cum_vol_pct_diff)  # This is a series
    '''print('diff')
    print(type(diff))
    print(diff.describe())
    print(diff.tail(3))'''

    #############################################################################
    # PLOTTING
    #############################################################################

    p1 = cum_vol_df.tail(lookback)
    p2 = diff.tail(lookback)
    # print(p1['Date'])
    pidx = df_close_idx.tail(lookback)

    date_labels = p1.index.strftime("%d/%m/%y").tolist()
    # print(date_labels)
    # Create subplots

    # fig, ax = plt.subplots(figsize=(17, 7))
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(17, 9))

    #############################################################################
    # Plot accumulated volume
    #############################################################################

    axs[0].bar(p1.index, p1['CumVol'], width=1, color='goldenrod', alpha=0.7, label='Accumulated Volume')
    axs[0].set_xticks(p1.index[::5])
    axs[0].set_xticklabels(date_labels[::5], rotation=45, ha='right')
    # axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Accumulated Volume', color='black')

    # Set y-axis limits to the range of accumulated volume
    axs[0].set_ylim(bottom=min(p1['CumVol']), top=max(p1['CumVol']))

    # Plot index
    axs0_twin = axs[0].twinx()

    # ax1.plot(pidx.index, pidx, 'black', label=idx)
    axs0_twin.plot(p1.index, pidx, 'black', label=idx)
    axs0_twin.set_ylabel(idx, color='black')

    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    #############################################################################
    # Plot difference close % - accumulated volume %
    #############################################################################
    axs[1].bar(p2.index, p2, width=1, color='goldenrod', alpha=0.7, label='Close % change minus CumVol % change')
    axs[1].set_xticks(p2.index[::5])
    axs[1].set_xticklabels(date_labels[::5], rotation=45, ha='right')
    # axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Close % chg - CumVol % chg', color='black')

    # Set y-axis limits to the range of accumulated volume, checking for NaN or Inf
    bottom_limit = min(p2.replace([np.inf, -np.inf], np.nan).dropna())
    top_limit = max(p2.replace([np.inf, -np.inf], np.nan).dropna())

    axs[1].set_ylim(bottom=bottom_limit, top=top_limit)

    # Plot index
    axs1_twin = axs[1].twinx()

    # ax1.plot(pidx.index, pidx, 'black', label=idx)
    axs1_twin.plot(p2.index, pidx, 'black', label=idx)
    axs1_twin.set_ylabel(idx, color='black')

    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = axs0_twin.get_legend_handles_labels()
    axs0_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    # Adjust layout
    # plt.savefig(f'{plots_folder}/{idx}_cum_vol.jpg')
    plt.tight_layout()
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return cum_vol


##########################################################################
# % movers over time
##########################################################################
def movers(df_close, idx, df_close_idx):
    global lookback

    # print(df_idx.tail(lookback))

    # Fill NaN values in the DataFrame before calculating percentage changes
    df_filled = df_close.ffill()

    # Define the conditions
    c4plus = df_filled.pct_change() >= 0.04
    c4minus = df_filled.pct_change() <= -0.04
    c25_3plus = df_filled.pct_change(periods=63) >= 0.25
    c25_3minus = df_filled.pct_change(periods=63) <= -0.25
    c25_1plus = df_filled.pct_change(periods=21) >= 0.25
    c25_1minus = df_filled.pct_change(periods=21) <= -0.25
    c50_1plus = df_filled.pct_change(periods=21) >= 0.50
    c50_1minus = df_filled.pct_change(periods=21) <= -0.50
    c13_34plus = df_filled.pct_change(periods=34) >= 0.13
    c13_34minus = df_filled.pct_change(periods=34) <= -0.13

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
    breadth_df_summed = movers_df.T.groupby(level=0, sort=False).sum().T
    # For 5 and 10 day ratios
    c4_df = breadth_df_summed[['>4%1d', '<4%1d']]
    c13_df = breadth_df_summed[['>13%34d', '<13%34d']]

    #############################################################################
    # PLOTTING
    #############################################################################

    p = breadth_df_summed.tail(lookback)
    p1 = p[['>4%1d', '>25%Q', '>25%M', '>50%M', '>13%34d']]
    p2 = p[['<4%1d', '<25%Q', '<25%M', '<50%M', '<13%34d']].tail(lookback)
    date_labels = p1.index.strftime("%d/%m/%y").tolist()
    pidx = df_close_idx.tail(lookback)

    '''p = breadth_df_summed.reset_index().rename(columns={'index': 'Date'})
    p1 = p[['Date', '>4%1d', '>25%Q', '>25%M', '>50%M', '>13%34d']].tail(lookback)
    p2 = p[['Date', '<4%1d', '<25%Q', '<25%M', '<50%M', '<13%34d']].tail(lookback)
    date_labels = p1['Date'].dt.strftime("%d/%m/%Y").tolist()
    pidx = df_close_idx.tail(lookback)'''

    # Create subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(17, 9))

    #############################################################################
    # Top subplot POSITIVE MOVERS
    #############################################################################

    # Create a stacked bar chart
    bottom = None
    # for col in p1.columns[1:]:
    for col in p1.columns:
        axs[0].bar(p1.index, p1[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = p1[col]
        else:
            bottom += p1[col]

    axs[0].set_xticks(p1.index[::5])
    axs[0].set_xticklabels(date_labels[::5], rotation=45)

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
    # Bottom subplot NEGATIVE MOVERS
    #############################################################################

    # Create a stacked bar chart
    bottom = None
    for col in p2.columns:
        # for col in p2.columns[1:]:
        axs[1].bar(p2.index, -p2[col], label=col, bottom=bottom)
        if bottom is None:
            bottom = -p2[col]
        else:
            bottom += -p2[col]

    # Adding the x-axis with dates every 5
    axs[1].set_xticks(p1.index[::10])
    axs[1].set_xticklabels(date_labels[::10], rotation=45)
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
    lines_twin, labels_twin = axs0_twin.get_legend_handles_labels()
    axs[1].legend(lines + lines_twin, labels + labels_twin, loc='upper left')

    # plt.savefig(f'{plots_folder}/{idx}_pct_movers.jpg')
    plt.tight_layout()
    # plt.show(block=False)
    pdf.savefig()
    plt.close()

    return movers_df, breadth_df_summed, c4_df, c13_df


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

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(17, 9))

    #############################################################################
    # Plotting 5 and 10 day breadth_ratios on the primary y-axis of upper plot
    #############################################################################

    axs[0].plot(p1.index, p1['ratio5'], label='5-day breadth ratio', color='blue', zorder=0)
    axs[0].plot(p1.index, p1['ratio10'], label='10-day breadth ratio', color='green', zorder=0)

    # Adding the x-axis with dates every 5
    axs[0].set_xticks(p1.index[::5])
    axs[0].set_xticklabels(date_labels[::5], rotation=45)

    # Adding labels for both y-axs[]es
    axs[0].set_ylabel('5 and 10 day ratios')
    # Add title for plot
    axs[0].set_title(f"{idx} - Breadth Ratios")

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
    axs[1].set_xticks(p1.index[::5])
    axs[1].set_xticklabels(date_labels[::5], rotation=45)

    # Adding labels for both y-axs[]es
    axs[1].set_ylabel('13/34 ratio')
    # Add title for plot
    axs[1].set_title(f"{idx} - 13/34D Ratio")

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

    """# Specify the reference ('^BVSP') CSV file
    ref_csv_file = f"{data_folder}/INDEX_{idx}.csv"

    # Read the reference CSV file to get the 'Adj Close' column
    combined_df = pd.read_csv(ref_csv_file, usecols=['Date', 'Adj Close'], index_col='Date', parse_dates=True)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df.rename(columns={'Adj Close': idx}, inplace=True)"""

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

    # print(f'Combined index df:')
    # print(combined_df.tail(10))

    # Plot the normalized data
    p = combined_df.tail(lookback)
    # Remove nan (bitcoin)
    pp = p.dropna()
    # Rebase p
    p1 = pp / pp.iloc[0]

    # Checks cos of "ValueError: Axis limits cannot be NaN or Inf"
    # print(f'Check {idx_code} for NaN {p1.isnull().sum().sum()}')  # Check for NaN
    # print(f'Check {idx_code} for Inf {np.isinf(p1).sum().sum()}')  # Check for Inf

    # Calculate the maximum and minimum values in the DataFrame columns
    # max_value = p1.max().max()
    # min_value = p1.min().min()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(17, 7))
    date_labels = p1.index.strftime("%d/%m/%y").tolist()

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
    ax1.set_xticks(p1.index[::5])
    ax1.set_xticklabels(date_labels[::5], rotation=45)

    # Combine legends for both subplots
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')

    pdf.savefig()
    plt.close()


##########################################################################
# Function to apply color to cells of dataframe
##########################################################################
"""def get_colors(s):

    s = s.astype(float).sort_values()
    return (pd.merge_asof(s.reset_index(), s.quantile(q=percentiles).reset_index(), direction='backward')
            .set_index('Date')['index']
            .map(lambda x: f'background-color: {percentile_color.get(x, "")}')
            )"""

def get_colors(series, percentiles, percentile_color):

    colors= []
    for value in series.values:
        for percentile, color in sorted(zip(percentiles, percentile_color.items()), reverse=True):
            if value <= percentile:
                colors.append(color)
                break
    return colors


##########################################################################
# Draw Stock Bee Primary Breadth Indicators
##########################################################################
def stock_b_table(full_df):

    df = full_df[['>4%1d', '<4%1d',
                  'ratio5', 'ratio10',
                  '>25%Q', '<25%Q',
                  '>25%M', '<25%M',
                  '>50%M', '<50%M',
                  '>13%34d', '<13%34d',
                  '$>MA40',
                  'Adj Close']]

    # Format 'Adj Close' column to 2 decimal places with thousands separators
    df = df.copy()
    df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    df.loc[:, 'Close'] = df['Close'].apply(lambda x: '{:,.2f}'.format(x))

    # Create a new column for formatted date
    df['Date'] = df.index.strftime('%d-%m-%y')

    # Rearrange the order of columns with 'Date' as the first column
    df = df[['Date'] + [col for col in df.columns if col != 'Date']]

    # print(df.tail(10))
    # print(type(df.index))
    # df.to_csv('use_for_percentiles.csv')

    t1 = df.tail(42)

    # Convert the DataFrame to a matplotlib table
    # table_fig = plt.figure(figsize=(17, 12))  # Adjust figure size as needed
    fig, ax = plt.subplots(figsize=(17, 12))  # Adjust figure size as needed
    # ax = fig.add_subplot(111, frameon=False)  # Remove frame
    # Remove unnecessary axes details
    ax.axis('off')

    table_col_labels = list(t1.columns)

    # Create the table
    table = ax.table(cellText=t1.values, colLabels=table_col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Make column headers bold
    for (i, key) in enumerate(table_col_labels):
        cell = table[0, i]
        cell.set_fontsize(12)  # Increase font size for column headers
        cell.set_text_props(weight='bold')  # Make column headers bold
        cell.set_facecolor('lightgray')  # Set cell background color
        cell.set_height(0.05)

    table.scale(1, 1)  # Adjust table size as needed
    # Add title
    ax.set_title("Stock Bee Breadth Indicators", fontsize=12, y=1.02)

    # Add the table to the PDF
    pdf.savefig(fig)
    plt.close(fig)


##########################################################################
# Draw highs and lows table
##########################################################################
def hi_lo_table(full_df):  # get_colors_func, p, p_clr):

    df = full_df[['ATH', 'ATL', '12MH', '12ML', '3MH', '3ML', '1MH', '1ML', 'Adj Close']]
    df = df.copy()
    # Create a new column for formatted date
    df.loc[:, 'Date'] = df.index.strftime('%d/%m/%y')

    # Rearrange the order of columns with 'Date' as the first column
    df = df[['Date'] + [col for col in df.columns if col != 'Date']]

    # Format 'Adj Close' column to 2 decimal places with thousands separators
    df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    df.loc[:, 'Close'] = df['Close'].apply(lambda x: '{:,.2f}'.format(x))

    # print(df.tail(10))
    # print(type(df.index))
    # df.to_csv('use_for_percentiles.csv')

    t1 = df.tail(42)

    # Convert the DataFrame to a matplotlib table
    fig, ax = plt.subplots(figsize=(17, 12))  # Adjust figure size as needed
    # Remove unnecessary axes details
    ax.axis('off')

    # Create the table
    table_col_labels = list(t1.columns)
    table = ax.table(cellText=t1.values, colLabels=table_col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Make column headers bold
    for (i, key) in enumerate(table_col_labels):
        cell = table[0, i]
        cell.set_fontsize(12)  # Increase font size for column headers
        cell.set_text_props(weight='bold')  # Make column headers bold
        cell.set_facecolor('lightgray')  # Set cell background color
        cell.set_height(0.05)

    table.scale(1, 1)  # Adjust table size as needed

    # Add title
    ax.set_title("Highs and Lows", fontsize=12, y=1.02)

    # Add the table to the PDF
    pdf.savefig(fig)
    plt.close(fig)


##########################################################################
# Plot a table test
##########################################################################
def plot_table(df_in):

    df = df_in[['>4%1d', '<4%1d',
                'ratio5', 'ratio10',
                '>25%Q', '<25%Q',
                '>25%M', '<25%M',
                '>50%M', '<50%M',
                '>13%34d', '<13%34d',
                '$>MA40',
                'Adj Close']]

    df.style.apply(get_colors)

    fig, ax = plt.subplots(figsize=(17, 7))
    ax.axis('off')  # Turn off axis labels and ticks

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#f5f5f5'] * len(df.columns)
                     )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    pdf.savefig(fig)
    plt.close(fig)


##########################################################################

# -----------------------------MAIN PROGRAM-------------------------------

##########################################################################

# Work with all indices or just one?
mkt_list = get_market_map(yahoo_idx_components_dictionary)

# What do you want to do? Update, use or download new data?
up_use_dl = get_user_choice()

# Define how far to look back on graphs and database start/end
lookback = get_lookback()
from_date = "2000-01-01"
until_date = download_until()

# When downloading new, ask for database starting date (from_date). Ask here or it asks for it on each pass
if up_use_dl == 3:
    # When downloading, must specify start date
    from_date = input("Enter start date (YYYYMMDD): ") or "20000101"
    # Convert the string to a datetime object
    date_object = datetime.strptime(from_date, "%Y%m%d")
    # Format the datetime object as a string in the desired format
    from_date = date_object.strftime("%Y-%m-%d")
    print(f'Download until: {until_date}')
    create_databases(mkt_list, from_date, until_date)

# Use existing data
elif up_use_dl == 1:
    # Perform update
    print(f'Update until: {until_date}')
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
        pdf_filename = f'{pdf_folder}/{idx_code}_{datetime.today().strftime("%Y-%m-%d")}.pdf'
        with PdfPages(pdf_filename) as pdf:
            #  This is required because we are in a new loop and idx/eod_df are undefined
            idx_df = pd.read_csv(f'{data_folder}/INDEX_{idx_code}.csv', index_col=0, parse_dates=True)
            comp_df = pd.read_csv(f'{data_folder}/EOD_{market_name}.csv', header=[0, 1], index_col=0, parse_dates=True)

            # print(f'First and last 3 rows of components df ({market_name}):')
            sample_mkt = pd.concat([comp_df.head(1), comp_df.tail(1)])
            # print(sample_mkt)
            # print(f'First and last 3 rows of index df ({idx_code}):')
            sample_idx = pd.concat([idx_df.head(1), idx_df.tail(1)])
            # print(sample_idx)
            total_stocks = comp_df.columns.get_level_values(1).nunique()

            # Which market will be used for calculations
            # print(f'Using {idx_code} and {market_name}')

            # with PdfPages(pdf_filename) as pdf:

            # Basic plot to get an idea of correct shape
            plot_close_and_volume(idx_df, idx_code)

            # Dataframe with ATH/L, 12MH/L, 3MH/L and 1MH/L
            # high_low_df = highs_and_lows(idx_df, eod_df, time_periods, idx_code)
            high_low_df = highs_and_lows(idx_df, comp_df, time_periods, idx_code)

            # Multiindex dataframe with all tickers and their MA's: 5, 12, 25, 40, 50, 100, 200
            mov_avgs_df = calculate_moving_averages(comp_df['Adj Close'], mas_to_use, 'Close')
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
            mvrs, mvrs_sum, fourpc_df, thirteenpc_df = movers(comp_df['Adj Close'], idx_code, idx_df['Adj Close'])
            br_ratios = ratios(fourpc_df, thirteenpc_df, idx_code, idx_df['Adj Close'])

            # Comparing indices
            plot_normalized_indexes(filtered_index_dict, idx_code)

            ##########################################################################
            # Breadth dataframe
            ##########################################################################
            adr_df = pd.DataFrame(adr, columns=[adr.name])
            acc_vol_df = pd.DataFrame(acc_vol, columns=[acc_vol.name])
            all_dfs_df = pd.concat([high_low_df, over_short_mas_pct, over_40ma_pct,
                                    over_long_mas_pct, mvrs_sum, br_ratios,
                                    adr_df, idx_df, acc_vol_df], axis=1)
            # print(breadth_df.tail(10))
            # print(breadth_df.columns)
            # Select the last 10 rows
            # nan_check = all_dfs_df.tail(10).isna()
            # if nan_check.any().any():
            #     print("NaN in last 10 rows")
            # else:
            #     print("No NaN in last 10 rows")
            # Display the result
            # print(nan_check)

            #plot_table(all_dfs_df)

            # hi_lo_table(all_dfs_df.copy(), percentiles, percentile_color)

            stock_b_table(all_dfs_df)
            hi_lo_table(all_dfs_df)

