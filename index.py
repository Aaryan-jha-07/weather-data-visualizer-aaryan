import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# --- TASK 1: LOAD DATA ---
def load_data(filename):
   
    try:
        print(f"--- Loading Data from {filename} ---")
        df = pd.read_csv(filename)

        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nStatistical Description:")
        print(df.describe())
        
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please check the filename.")
        return None


def clean_data(df):
   
    print("\n--- Cleaning Data ---")
    
   
    if 'Date' not in df.columns:
        print("Note: 'Date' column missing. Generating synthetic dates for analysis.")
        df['Date'] = pd.date_range(start='2023-01-01', periods=len(df))
    else:
        df['Date'] = pd.to_datetime(df['Date'])
 
    if 'Temperature' not in df.columns and 'MinTemp' in df.columns and 'MaxTemp' in df.columns:
        df['Temperature'] = (df['MinTemp'] + df['MaxTemp']) / 2
        print("Created 'Temperature' column from MinTemp and MaxTemp.")

    if 'Humidity' not in df.columns and 'Humidity9am' in df.columns and 'Humidity3pm' in df.columns:
        df['Humidity'] = (df['Humidity9am'] + df['Humidity3pm']) / 2
        print("Created 'Humidity' column from Humidity9am and Humidity3pm.")

  
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    print("Data cleaned. Missing values handled.")
    return df


def compute_statistics(df):
    """
    Computes basic statistics using NumPy.
    """
    print("\n--- Computing Statistics (using NumPy) ---")
    
    if 'Temperature' in df.columns:
        temps = df['Temperature'].to_numpy()
        
        mean_temp = np.mean(temps)
        max_temp = np.max(temps)
        min_temp = np.min(temps)
        std_dev = np.std(temps)
        
        print(f"Temperature Statistics:")
        print(f" - Mean: {mean_temp:.2f} C")
        print(f" - Max:  {max_temp:.2f} C")
        print(f" - Min:  {min_temp:.2f} C")
        print(f" - Std Dev: {std_dev:.2f}")
    
    if 'Rainfall' in df.columns:
        rain = df['Rainfall'].to_numpy()
        print(f"Total Rainfall recorded: {np.sum(rain):.2f} mm")


def analyze_grouping(df):
    """
    Groups data by Month to find monthly average temperatures.
    """
    print("\n--- Grouping Analysis ---")
   
    df['Month'] = df['Date'].dt.month_name()
    
   
    monthly_stats = df.groupby('Month')['Temperature'].mean().sort_values()
    print("Average Temperature by Month:")
    print(monthly_stats)
    return monthly_stats


def visualize_data(df):
    """
    Generates and saves plots using Matplotlib.
    """
    print("\n--- Generating Plots ---")
    
    df_sorted = df.sort_values('Date')
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['Date'], df_sorted['Temperature'], label='Avg Temp', color='orange')
    if 'MaxTemp' in df.columns:
        plt.plot(df_sorted['Date'], df_sorted['MaxTemp'], label='Max Temp', linestyle='--', alpha=0.5)
    plt.title('Temperature Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.grid(True)
    plt.savefig('temperature_trend.png')
    print("Saved 'temperature_trend.png'")

   
    plt.figure(figsize=(8, 5))
    plt.hist(df['Humidity'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Humidity')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Frequency')
    plt.savefig('humidity_dist.png')
    print("Saved 'humidity_dist.png'")

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df_sorted['Date'], df_sorted['Temperature'], 'r-')
    ax1.set_title('Temperature Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temp (C)')
    ax1.tick_params(axis='x', rotation=45)
    
   
    ax2.plot(df_sorted['Date'], df_sorted['Humidity'], 'b-')
    ax2.set_title('Humidity Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Humidity (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('combined_subplots.png')
    print("Saved 'combined_subplots.png'")


def export_data(df):
    """
    Exports the cleaned data to a CSV file.
    """
    print("\n--- Exporting Cleaned Data ---")
    output_filename = 'cleaned_weather_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"Cleaned dataset saved as '{output_filename}'.")


if __name__ == "__main__":

    filename = 'weather.csv'
   
    weather_df = load_data(filename)
    
    if weather_df is not None:
        weather_df_clean = clean_data(weather_df)
        compute_statistics(weather_df_clean)
        analyze_grouping(weather_df_clean)
        visualize_data(weather_df_clean)
        export_data(weather_df_clean)
        
        print("\nAnalysis Complete. Check the output PNG files and CSV.")