import pandas as pd

# Which CSV file to modify?
csv_path = 'EOD_IFIX.csv'  # This is the component file
#csv_path = 'EOD_fii.csv'  # This is the index file

if csv_path == 'EOD_IFIX.csv':
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Assuming the first column is named 'Date' and contains dates in the format '26/12/2023'
    # Convert the 'Date' column to the desired format 'YYYY-MM-DD'
    # Read the CSV file with a date index
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True, dayfirst=True, decimal=',')

    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)

    # Convert the index to the desired format '%Y-%m-%d'
    df.index = df.index.strftime('%Y-%m-%d')

    # Remove rows where all values are NaN and count the number of deleted rows
    deleted_rows_count = len(df) - len(df.dropna(how='all'))

    # Print the number of deleted rows
    print(f'Number of deleted rows: {deleted_rows_count}')
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')

    # Sort the DataFrame based on the 'Date' column in ascending order
    df = df.sort_values(by='Date')

    print(df.head(5))
    # Overwrite the original file with the updated and sorted DataFrame
    df.to_csv(csv_path)
    print('File saved')
    
elif csv_path == 'EOD_fii.csv':
    # Read the CSV file
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True, infer_datetime_format=True)

    # Print the first few rows to inspect column names and structure
    print(df.head())

    # Remove rows where all values are NaN and count the number of deleted rows
    deleted_rows_count = len(df) - len(df.dropna(how='all'))

    # Print the number of deleted rows
    print(f'Number of deleted rows: {deleted_rows_count}')
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')

    # Sort the DataFrame based on the index (Date) in ascending order
    df = df.sort_index()

    # Convert the 'Date' index back to the 'YYYY-MM-DD' format
    df.index = df.index.strftime('%Y-%m-%d')
    print(df.head(5))

    # Overwrite the original file with the updated, sorted, and cleaned DataFrame
    df.to_csv(csv_path)
    print('File saved')
else:
    print('Filename problem')