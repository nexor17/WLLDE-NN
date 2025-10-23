import math
import os
from re import X
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def process_csv(input_csv):
    print(f"Processing {input_csv}...")
    
    # Read CSV file manually to handle empty lines and trailing periods
    datetimes = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    volumes = []
    
    with open(input_csv, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[1:]):  # Skip header
            line = line.strip()
            # Skip empty lines, lines with just a period, or lines that don't have enough data
            if not line or line == '.' or line.count(',') < 5:
                continue
            
            try:
                parts = line.split(',')
                if len(parts) >= 6 and parts[0]:  # Ensure we have all columns and datetime is not empty
                    datetimes.append(parts[0])
                    open_prices.append(float(parts[1]))
                    high_prices.append(float(parts[2]))
                    low_prices.append(float(parts[3]))
                    close_prices.append(float(parts[4]))
                    volumes.append(float(parts[5]))
            except (ValueError, IndexError) as e:
                # Skip malformed rows
                continue
    
    if len(datetimes) == 0:
        raise ValueError("No valid data rows found in CSV")
    
    # Convert to numpy arrays
    open_prices = np.array(open_prices)
    high_prices = np.array(high_prices)
    low_prices = np.array(low_prices)
    y = np.array(close_prices)
    volumes = np.array(volumes)
    
    # Extract day of year and hour of day from datetime
    day_of_year = []
    hour_of_day = []
    for dt_str in datetimes:
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            day_of_year.append(dt.timetuple().tm_yday)
            hour_of_day.append(dt.hour)
        except ValueError:
            # Skip invalid datetime formats
            continue
    
    day_of_year = np.array(day_of_year)
    hour_of_day = np.array(hour_of_day)
    
    x = np.arange(len(y))

    xMat = np.matrix(x).T
    m = xMat.shape[0]
    
    xCon = np.concatenate(([np.ones((m, 1)), xMat]),1)
    
    def lwr(x, y, tau):
        yPred = np.zeros(m)
        for i in range(m):
            yPred[i] = (x[i] * localWeight(x[i], x, y, tau)).item()
        return yPred
    
    def localWeight(x1, x, y, tau):
        wt = funct(x1, x, tau)
        w = np.linalg.inv(x.T * (wt * x)) * (x.T * (wt * y))
        return w
    
    def funct(x1, x, tau):
        diff = x1 - x
        sq_diff = np.multiply(diff, diff)
        distances = np.sum(sq_diff, axis=1)
        weights_diag = np.exp(distances / (-2.0 * tau**2))
        return np.diag(np.ravel(weights_diag))
    
    yMat = np.matrix(y)
    tua = 0.9
    yPred = lwr(xCon, yMat.T, tua)
    
    # Stack all data including day_of_year and hour_of_day
    output_data = np.vstack([day_of_year, hour_of_day, open_prices, high_prices, low_prices, y, volumes, yPred]).T
    
    output_dir = 'optidata'
    base_filename = os.path.splitext(os.path.basename(input_csv))[0]
    output_filename = f'{base_filename}_opti.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    header = 'dayoftheyear,houroftheday,open,high,low,close,volume,WLLDE'
    np.savetxt(output_path, output_data, delimiter=',', header=header, comments='', fmt=['%d', '%d', '%.2f', '%.2f', '%.2f', '%.2f', '%d', '%.2f'])
    
    print(f"Saved to {output_path}")
    
    #plt.scatter(x, y)
    #plt.plot(x, yPred)
    #plt.show()

# Process all CSV files in the data directory
if __name__ == "__main__":
    data_dir = 'data'
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
    else:
        print(f"Found {len(csv_files)} CSV files to process")
        for csv_file in csv_files:
            try:
                process_csv(csv_file)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")


    
    

