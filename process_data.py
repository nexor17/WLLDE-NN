import pandas as pd
import os

# Directory containing the CSV files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Function to process each CSV file
def process_csv(file_path):
    """Reads a CSV, replaces Datetime with DayOfYear, and saves it back."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if 'Datetime' column exists
        if 'Datetime' not in df.columns:
            print(f"'Datetime' column not found in {os.path.basename(file_path)}. Skipping.")
            return

        # Convert 'Datetime' column to datetime objects
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

        # Drop rows where 'Datetime' could not be parsed
        df.dropna(subset=['Datetime'], inplace=True)

        # Calculate the day of the year and create a new column
        df['DayOfYear'] = df['Datetime'].dt.dayofyear

        # Drop the original 'Datetime' column
        df.drop(columns=['Datetime'], inplace=True)
        
        # Reorder columns to have 'DayOfYear' first
        cols = df.columns.tolist()
        if 'DayOfYear' in cols:
            cols.insert(0, cols.pop(cols.index('DayOfYear')))
            df = df[cols]

        # Save the modified DataFrame back to the same file
        df.to_csv(file_path, index=False)
        print(f"Processed and updated {os.path.basename(file_path)}")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

# Main execution block
if __name__ == "__main__":
    # Check if the data directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
    else:
        # Loop through all files in the data directory
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                file_path = os.path.join(DATA_DIR, filename)
                process_csv(file_path)
