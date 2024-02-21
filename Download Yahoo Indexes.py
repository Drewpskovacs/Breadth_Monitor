import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from_date = '2000-01-01'
until_date = datetime.now()

# Where files and database are stored (same directory as MAIN PROGRAM)
data_folder = 'Data/Data_files'
'''codes_folder = 'Data/Codes_to_download'
plots_folder = 'Data/Plots'
pdf_folder = 'Data/PDF'
market_monitor_folder = 'Data/Mkt_Monitor'''


extra_indices_yahoo_dictionary = {

    1: {'idx_code': '^BVSP', 'market': 'bovespa'},
    2: {'idx_code': '^IXIC', 'market': 'nasdaq'},
    3: {'idx_code': '^FTSE', 'market': 'ftse100'},
    4: {'idx_code': '^FTMC', 'market': 'ftse250'},
    5: {'idx_code': '^GSPC', 'market': 'sp500'},
    6: {'idx_code': 'IFIX', 'market': 'fii'},
    7: {'idx_code': 'GC=F', 'market': 'gold'},
    8: {'idx_code': 'BTC-USD', 'market': 'bitcoin'},
    9: {'idx_code': 'BRL=X', 'market': 'dollar'},
    10: {'idx_code': 'CL=F', 'market': 'crudeoil'},
    11: {'idx_code': 'ZN=F', 'market': '10yrTnote'},
    12: {'idx_code': 'ZT=F', 'market': '2yrTnote'},
    13: {'idx_code': '^VIX', 'market': 'vix'},
    14: {'idx_code': 'IDIV.SA', 'market': 'dividends'},
    15: {'idx_code': '^IBX50', 'market': 'ibx50'},
    16: {'idx_code': '^IBX50', 'market': 'test'}

}

def download_and_save_extras(tikr, mk, first, last):

    global data_folder

    # Download historical data
    data = yf.download(tikr, start=first, end=last)
    data.index = pd.to_datetime(data.index)

    # Save to CSV file
    file_name = f"{data_folder}/{mk}.csv"
    data.to_csv(file_name)
    print(f"Data for {tikr} downloaded and saved to {file_name}")
    #plot_close_and_volume(tikr, mk)

def download_and_save_data_extra_yahoo_indexes(yahoo, first, last):
    for key, value in yahoo.items():
        ticker = value['idx_code']
        mkt = value['market']
        download_and_save_extras(ticker, mkt, first, last)

# Download the extra indices from Yahoo
download_and_save_data_extra_yahoo_indexes(extra_indices_yahoo_dictionary, from_date, until_date)