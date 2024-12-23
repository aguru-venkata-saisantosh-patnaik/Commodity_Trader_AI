#!/usr/bin/env python
# coding: utf-8

# # Data Fetching, Cleaning and Preprocessing
# 

# In[1]:


pip install yfinance


# In[2]:


import yfinance as yf
import os
from datetime import datetime

# Directory to save the data files
data_dir = "commodity_data"
os.makedirs(data_dir, exist_ok=True)

# Define the commodities and their Yahoo Finance ticker symbols
commodities = {
    "crude_oil": "CL=F",    # Crude Oil Futures
    "coffee": "KC=F",       # Coffee Futures
    "natural_gas": "NG=F",  # Natural Gas Futures
    "gold": "GC=F",         # Gold Futures
    "wheat": "ZW=F",        # Wheat Futures
    "cotton": "CT=F",       # Cotton Futures
    "corn": "ZC=F",         # Corn Futures
    "sugar": "SB=F",        # Sugar Futures
    "silver": "SI=F",       # Silver Futures
    "copper": "HG=F",       # Copper Futures
}

# Define the date range for historical data
start_date = "2014-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# Function to fetch and save data for each commodity
def fetch_and_save_commodity_data(ticker, commodity_name):
    print(f"Fetching data for {commodity_name}...")
    try:
        # Fetch the historical data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Check if data was retrieved
        if not data.empty:
            # Save to CSV file
            file_path = os.path.join(data_dir, f"{commodity_name}.csv")
            data.to_csv(file_path)
            print(f"Data for {commodity_name} saved to {file_path}")
        else:
            print(f"No data found for {commodity_name}.")

    except Exception as e:
        print(f"An error occurred while fetching data for {commodity_name}: {e}")

# Fetch data for each commodity in the list
for commodity_name, ticker in commodities.items():
    fetch_and_save_commodity_data(ticker, commodity_name)

print("Data collection complete.")


# Data cleaning and preprocessing
# 

# In[3]:


import pandas as pd
import os

# Directory containing the saved data files
data_dir = "commodity_data"

# List of commodities, matching file names created in the previous code
commodities = [
    "crude_oil", "coffee", "natural_gas", "gold", "wheat",
    "cotton", "corn", "sugar", "silver", "copper"
]

# Function to load and display the data for each commodity
def display_commodity_data(commodity_name):
    file_path = os.path.join(data_dir, f"{commodity_name}.csv")

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the data
        df = pd.read_csv(file_path)

        # Display the first few rows of the data
        print(f"\nData for {commodity_name.capitalize()}:\n")
        print(df.head(), "\n")  # Display the first 5 rows
    else:
        print(f"No data file found for {commodity_name}.")

# Display data for each commodity
for commodity_name in commodities:
    display_commodity_data(commodity_name)


# In[4]:


import pandas as pd
import os

# Directory containing the saved data files
data_dir = "commodity_data"
commodity_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "_cleaned" not in f and "_features" not in f]  # Skip files with '_cleaned.csv'

def clean_commodity_data(file_path, commodity_name):
    print(f"Cleaning data for {commodity_name} from {file_path}...")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, skiprows=2)  # Skip first 2 rows with unnecessary data

    # Ensure correct column headers
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df.columns = expected_columns

    # Drop rows where 'Date' is NaN (noise rows)
    df = df.dropna(subset=["Date"])

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=["Date"])  # Drop rows where Date conversion failed

    # Ensure numeric values in price and volume columns, coercing errors to NaN and then dropping
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with any remaining NaN values
    df = df.dropna()

    # Remove duplicates by 'Date' and keep the first instance
    df = df.drop_duplicates(subset="Date", keep="first")

    # Save the cleaned data back to the CSV file
    cleaned_file_path = os.path.join(data_dir, f"{commodity_name}_cleaned.csv")
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved for {commodity_name} at {cleaned_file_path}.")

# Process each file in the directory, skipping already cleaned files
for file_name in commodity_files:
    commodity_name = file_name.replace(".csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    clean_commodity_data(file_path, commodity_name)

print("\nData cleaning complete for all commodities.")


# In[5]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"

# List of cleaned files for each commodity, assuming the suffix "_cleaned.csv" for cleaned data
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

def display_cleaned_data(file_path, commodity_name):
    # Load the cleaned CSV file
    df = pd.read_csv(file_path)

    # Display the first few rows to verify the data structure
    print(f"\nCleaned Data for {commodity_name.capitalize()}:\n")
    print(df.head(), "\n")  # Display the first 5 rows

# Loop through each cleaned file and display data
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    display_cleaned_data(file_path, commodity_name)

print("Data display complete for all commodities.")


# In[6]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"

# List of cleaned files for each commodity
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

def modify_columns(file_path, commodity_name):
    print(f"Modifying columns for {commodity_name} data...")

    # Load the cleaned CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Drop the 'Close' column
    if 'Close' in df.columns:
        df = df.drop(columns=['Close'])

    # Rename 'Adj Close' to 'Close'
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})

    # Save the modified data back to the CSV file
    df.to_csv(file_path, index=False)
    print(f"Columns modified and data saved for {commodity_name}.")

# Process each cleaned file
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    modify_columns(file_path, commodity_name)

print("Column modifications complete for all commodities.")


# In[7]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"

# List of cleaned files for each commodity, assuming the suffix "_cleaned.csv" for cleaned data
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

def display_final_cleaned_data(file_path, commodity_name):
    # Load the cleaned CSV file
    df = pd.read_csv(file_path)

    # Display the first few rows to verify the data structure
    print(f"\nFinal Cleaned Data for {commodity_name.capitalize()}:\n")
    print(df.head(), "\n")  # Display the first 5 rows

# Loop through each cleaned file and display data
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    display_final_cleaned_data(file_path, commodity_name)

print("Final cleaned data display complete for all commodities.")


# In[8]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"

# List of cleaned files for each commodity
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

def format_date_column(file_path, commodity_name):
    print(f"Formatting date column for {commodity_name} data...")

    # Load the cleaned CSV file
    df = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime, then remove the time part by formatting
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Keep only the date part

    # Save the modified data back to the CSV file
    df.to_csv(file_path, index=False)
    print(f"Date column formatted and data saved for {commodity_name}.")

# Process each cleaned file
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    format_date_column(file_path, commodity_name)

print("Date formatting complete for all commodities.")


# In[9]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"

# List of cleaned files for each commodity, assuming the suffix "_cleaned.csv" for cleaned data
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

def display_final_cleaned_data(file_path, commodity_name):
    # Load the cleaned CSV file
    df = pd.read_csv(file_path)

    # Display the first few rows to verify the data structure
    print(f"\nFinal Cleaned Data for {commodity_name.capitalize()}:\n")
    print(df.head(), "\n")  # Display the first 5 rows

# Loop through each cleaned file and display data
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    display_final_cleaned_data(file_path, commodity_name)

print("Final cleaned data display complete for all commodities.")


# # Exploratory Data Analysis (EDA)

# In[10]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"

# List of cleaned files for each commodity
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

# Function to load and display summary statistics for each commodity
def summary_statistics(file_path, commodity_name):
    df = pd.read_csv(file_path)

    # Calculate summary statistics
    stats = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()

    print(f"\nSummary Statistics for {commodity_name.capitalize()}:\n")
    print(stats)

# Display summary statistics for each commodity
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    summary_statistics(file_path, commodity_name)

print("Summary statistics complete for all commodities.")


# In[11]:


import matplotlib.pyplot as plt

def plot_time_series(file_path, commodity_name):
    df = pd.read_csv(file_path, parse_dates=['Date'])

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title(f"Price Trend for {commodity_name.capitalize()}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Plot time series for each commodity
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")
    file_path = os.path.join(data_dir, file_name)
    plot_time_series(file_path, commodity_name)


# In[12]:


def plot_rolling_volatility(file_path, commodity_name, window=30):
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Calculate rolling 30-day standard deviation of the Close price
    df['Rolling Volatility'] = df['Close'].rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Rolling Volatility'], label=f'{window}-Day Rolling Volatility', color='orange')
    plt.title(f"Rolling Volatility for {commodity_name.capitalize()} (Window={window} Days)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

# Plot rolling volatility for each commodity
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")
    file_path = os.path.join(data_dir, file_name)
    plot_rolling_volatility(file_path, commodity_name)


# In[13]:


import pandas as pd

# Load all commodities into a single DataFrame with Date and Close columns
commodity_data = {}

for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path, parse_dates=['Date'])
    commodity_data[commodity_name] = df.set_index('Date')['Close']

# Combine all Close price series into a single DataFrame
combined_df = pd.DataFrame(commodity_data)

# Calculate and display the correlation matrix
correlation_matrix = combined_df.corr()
print("\nCorrelation Matrix of Commodity Prices:\n")
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Commodity Prices")
plt.show()


# # Sentiment Analysis

# In[14]:


get_ipython().system('pip install vaderSentiment')



# In[15]:


import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time
from datetime import datetime

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Your News API key
api_key = '7afc72ab80ad4f2ba7dc0f4b13fd193d'  # Ensure you have a valid API key

# Commodities list
commodities = ["crude oil", "coffee", "natural gas", "gold", "wheat", "cotton", "corn", "sugar", "silver", "copper"]

# Sentiment thresholds
thresholds = {
    "positive": 0.1,    # Adjust these values to your preferences
    "negative": -0.1
}

# Function to fetch and analyze sentiment for each commodity
def fetch_and_analyze_sentiment(commodity):
    url = f'https://newsapi.org/v2/everything?q={commodity}&language=en&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])

    # Print response to debug
    print(f"Fetching news for {commodity}...")
    print(f"API Response: {response.json()}")  # Print the full response for debugging

    # List to store sentiment data
    sentiment_data = []

    for article in articles:
        title = article['title']
        description = article['description'] or ""
        content = article['content'] or ""

        # Aggregate the text for sentiment analysis
        text = f"{title}. {description}. {content}"

        # Get sentiment score
        sentiment = analyzer.polarity_scores(text)
        compound_score = sentiment['compound']

        # Classify sentiment based on thresholds
        if compound_score >= thresholds['positive']:
            sentiment_label = "Positive"
        elif compound_score <= thresholds['negative']:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Add data to list
        sentiment_data.append({
            'Commodity': commodity,
            'Title': title,
            'Description': description,
            'Content': content,
            'Published At': article['publishedAt'],
            'Sentiment Score': compound_score,
            'Sentiment Label': sentiment_label
        })

    return pd.DataFrame(sentiment_data)

# Gather and combine sentiment data for all commodities
def gather_commodity_sentiment(commodities):
    all_sentiment_data = pd.DataFrame()

    for commodity in commodities:
        sentiment_df = fetch_and_analyze_sentiment(commodity)
        all_sentiment_data = pd.concat([all_sentiment_data, sentiment_df], ignore_index=True)
        time.sleep(1)  # to avoid hitting rate limits

    # Debug print
    print("Combined Sentiment Data:")
    print(all_sentiment_data.head())  # Show the first few rows of combined sentiment data

    return all_sentiment_data

# Dynamically adjust sentiment for commodities
def adjust_sentiment_analysis(all_sentiment_data):
    analysis_results = []

    # Ensure 'Commodity' column exists
    if 'Commodity' not in all_sentiment_data.columns:
        raise KeyError("Column 'Commodity' not found in the sentiment data.")

    for commodity in commodities:
        df_commodity = all_sentiment_data[all_sentiment_data['Commodity'] == commodity]

        # Calculate positive, neutral, negative sentiment counts
        positive_count = df_commodity[df_commodity['Sentiment Label'] == "Positive"].shape[0]
        neutral_count = df_commodity[df_commodity['Sentiment Label'] == "Neutral"].shape[0]
        negative_count = df_commodity[df_commodity['Sentiment Label'] == "Negative"].shape[0]

        # Determine overall sentiment trend based on majority
        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = "Positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Calculate sentiment score averages and trends
        avg_sentiment_score = df_commodity['Sentiment Score'].mean()
        latest_timestamp = df_commodity['Published At'].max()

        analysis_results.append({
            'Commodity': commodity,
            'Overall Sentiment': overall_sentiment,
            'Average Sentiment Score': avg_sentiment_score,
            'Positive Count': positive_count,
            'Neutral Count': neutral_count,
            'Negative Count': negative_count,
            'Last Update': latest_timestamp
        })

    return pd.DataFrame(analysis_results)

# Run the analysis and adjust sentiment dynamically
all_sentiment_data = gather_commodity_sentiment(commodities)
analysis_results = adjust_sentiment_analysis(all_sentiment_data)

# Display the final analysis for each commodity
print("\nSentiment Analysis Results:")
print(analysis_results)

# Optionally, save results for further analysis
all_sentiment_data.to_csv("dynamic_commodity_sentiment_data.csv", index=False)
analysis_results.to_csv("commodity_sentiment_summary.csv", index=False)


# In[16]:


import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time
from datetime import datetime

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Your News API key
api_key = '7afc72ab80ad4f2ba7dc0f4b13fd193d'

# Commodities list
commodities = ["crude oil", "coffee", "natural gas", "gold", "wheat", "cotton", "corn", "sugar", "silver", "copper"]

# Sentiment thresholds
thresholds = {
    "positive": 0.1,    # Adjust these values to your preferences
    "negative": -0.1
}

# Function to fetch and analyze sentiment for each commodity
def fetch_and_analyze_sentiment(commodity):
    url = f'https://newsapi.org/v2/everything?q={commodity}&language=en&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])

    # List to store sentiment data
    sentiment_data = []

    for article in articles:
        title = article['title']
        description = article['description'] or ""
        content = article['content'] or ""

        # Aggregate the text for sentiment analysis
        text = f"{title}. {description}. {content}"

        # Get sentiment score
        sentiment = analyzer.polarity_scores(text)
        compound_score = sentiment['compound']

        # Classify sentiment based on thresholds
        if compound_score >= thresholds['positive']:
            sentiment_label = "Positive"
        elif compound_score <= thresholds['negative']:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Add data to list
        sentiment_data.append({
            'Commodity': commodity,
            'Title': title,
            'Description': description,
            'Content': content,
            'Published At': article['publishedAt'],
            'Sentiment Score': compound_score,
            'Sentiment Label': sentiment_label
        })

    return pd.DataFrame(sentiment_data)

# Gather and combine sentiment data for all commodities
def gather_commodity_sentiment(commodities):
    all_sentiment_data = pd.DataFrame()

    for commodity in commodities:
        print(f"Fetching news for {commodity}...")
        sentiment_df = fetch_and_analyze_sentiment(commodity)
        all_sentiment_data = pd.concat([all_sentiment_data, sentiment_df], ignore_index=True)
        time.sleep(1)  # to avoid hitting rate limits

    return all_sentiment_data

# Dynamically adjust sentiment for commodities
def adjust_sentiment_analysis(all_sentiment_data):
    analysis_results = []

    for commodity in commodities:
        df_commodity = all_sentiment_data[all_sentiment_data['Commodity'] == commodity]

        # Calculate positive, neutral, negative sentiment counts
        positive_count = df_commodity[df_commodity['Sentiment Label'] == "Positive"].shape[0]
        neutral_count = df_commodity[df_commodity['Sentiment Label'] == "Neutral"].shape[0]
        negative_count = df_commodity[df_commodity['Sentiment Label'] == "Negative"].shape[0]

        # Determine overall sentiment trend based on majority
        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = "Positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Calculate sentiment score averages and trends
        avg_sentiment_score = df_commodity['Sentiment Score'].mean()
        latest_timestamp = df_commodity['Published At'].max()

        analysis_results.append({
            'Commodity': commodity,
            'Overall Sentiment': overall_sentiment,
            'Average Sentiment Score': avg_sentiment_score,
            'Positive Count': positive_count,
            'Neutral Count': neutral_count,
            'Negative Count': negative_count,
            'Last Update': latest_timestamp
        })

    return pd.DataFrame(analysis_results)

# Run the analysis and adjust sentiment dynamically
all_sentiment_data = gather_commodity_sentiment(commodities)
analysis_results = adjust_sentiment_analysis(all_sentiment_data)

# Display the final analysis for each commodity
print("\nSentiment Analysis Results:")
print(analysis_results)

# Optionally, save results for further analysis
all_sentiment_data.to_csv("dynamic_commodity_sentiment_data.csv", index=False)
analysis_results.to_csv("commodity_sentiment_summary.csv", index=False)


# In[17]:


import os

# Define the directory to store the cleaned sentiment data
sentiment_data_dir = "sentiment_data"
os.makedirs(sentiment_data_dir, exist_ok=True)

# Define file paths for storing sentiment data
cleaned_sentiment_file_path = os.path.join(sentiment_data_dir, "cleaned_sentiment_data.csv")
summary_sentiment_file_path = os.path.join(sentiment_data_dir, "commodity_sentiment_summary.csv")

# Clean and format the analysis results DataFrame
def clean_sentiment_data(analysis_results):
    # Drop any duplicates if present
    analysis_results = analysis_results.drop_duplicates()

    # Ensure no missing values in critical columns
    analysis_results = analysis_results.dropna(subset=["Commodity", "Overall Sentiment", "Average Sentiment Score"])

    # Sort by Commodity for easy reference
    analysis_results = analysis_results.sort_values(by="Commodity").reset_index(drop=True)

    # Save the cleaned sentiment data
    analysis_results.to_csv(cleaned_sentiment_file_path, index=False)
    print(f"Cleaned sentiment data saved to {cleaned_sentiment_file_path}")

# Run the cleaning function
clean_sentiment_data(analysis_results)

# Optionally, save all raw sentiment data if required for comparison or auditing
all_sentiment_data.to_csv(summary_sentiment_file_path, index=False)
print(f"Complete sentiment summary saved to {summary_sentiment_file_path}")


# In[ ]:





# # Feature Engineering for using enhance predictive analysis and improve trading strategy signals
# 

# Daily Returns

# In[18]:


import pandas as pd
import os

# Directory containing the cleaned data files
data_dir = "commodity_data"
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_cleaned.csv")]

def add_daily_returns(df):
    df['Daily_Return'] = df['Close'].pct_change()
    return df


# Moving Averages (10-day, 30-day 50-day)

# In[19]:


def add_moving_averages(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    return df


# Exponential Moving Average (EMA)

# In[20]:


def add_ema(df, span=10):
    df['EMA_10'] = df['Close'].ewm(span=span, adjust=False).mean()
    return df


# Bollinger Bands

# In[21]:


def add_bollinger_bands(df, window=20):
    # Middle Band (20-day Moving Average)
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    # Upper Band and Lower Band with standard deviation multiplier
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=window).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=window).std()
    return df


# Relative Strength Index (RSI)

# In[22]:


def add_rsi(df, window=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


# Moving Average Convergence Divergence (MACD)

# In[23]:


def add_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


# Applying the Features to Each Commodity

# In[24]:


def engineer_features(file_path, commodity_name):
    print(f"Engineering features for {commodity_name}...")

    # Load the cleaned CSV file
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Apply feature engineering functions
    df = add_daily_returns(df)
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_macd(df)

    # Save the enhanced data back to the same file with a new suffix
    feature_file_path = os.path.join(data_dir, f"{commodity_name}_features.csv")
    df.to_csv(feature_file_path, index=False)
    print(f"Features added and data saved for {commodity_name}.")

# Process each file in the directory
for file_name in commodity_files:
    commodity_name = file_name.replace("_cleaned.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    engineer_features(file_path, commodity_name)

print("Feature engineering complete for all commodities.")


# Visualising Features

# In[25]:


import pandas as pd
import os
import matplotlib.pyplot as plt

# Directory containing the feature-enhanced data files
data_dir = "commodity_data"
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_features.csv")]

def plot_features(file_path, commodity_name):
    # Load the data
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Plot Close Price with Moving Averages
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['MA_10'], label='10-Day MA', color='orange')
    plt.plot(df['Date'], df['MA_50'], label='50-Day MA', color='green')
    plt.title(f"{commodity_name.capitalize()} - Close Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Bollinger Bands
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['BB_Upper'], label='Upper Bollinger Band', linestyle='--', color='orange')
    plt.plot(df['Date'], df['BB_Lower'], label='Lower Bollinger Band', linestyle='--', color='orange')
    plt.fill_between(df['Date'], df['BB_Lower'], df['BB_Upper'], color='orange', alpha=0.1)
    plt.title(f"{commodity_name.capitalize()} - Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot RSI
    plt.figure(figsize=(14, 4))
    plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
    plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
    plt.title(f"{commodity_name.capitalize()} - RSI")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot MACD and Signal Line
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['Date'], df['Signal_Line'], label='Signal Line', color='red')
    plt.bar(df['Date'], df['MACD'] - df['Signal_Line'], label='MACD Histogram', color='gray', alpha=0.5)
    plt.title(f"{commodity_name.capitalize()} - MACD and Signal Line")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Loop through each feature-enhanced file and visualize features
for file_name in commodity_files:
    commodity_name = file_name.replace("_features.csv", "")  # Extract commodity name from file name
    file_path = os.path.join(data_dir, file_name)
    plot_features(file_path, commodity_name)

print("Feature plots generated for all commodities.")


# # Backtesting Strategies and finding the best strategy for each commodity

# In[26]:


import pandas as pd

# Momentum Strategy using EMA_10
def momentum_strategy(df, window=10):
    df['Momentum'] = df['Close'].diff(window)
    df['Position'] = 0
    df.loc[(df['Momentum'] > 0) & (df['Close'] > df['EMA_10']), 'Position'] = 1
    df.loc[(df['Momentum'] < 0) & (df['Close'] < df['EMA_10']), 'Position'] = -1
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df



# In[27]:


import pandas as pd

# Range Trading Strategy using Bollinger Bands
def range_trading_strategy(df, support_level=None, resistance_level=None):
    if support_level is None:
        support_level = df['BB_Lower'].mean()
    if resistance_level is None:
        resistance_level = df['BB_Upper'].mean()

    df['Position'] = 0
    df.loc[df['Close'] <= support_level, 'Position'] = 1  # Buy near lower Bollinger Band
    df.loc[df['Close'] >= resistance_level, 'Position'] = -1  # Sell near upper Bollinger Band
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df


# In[28]:


import pandas as pd

# Breakout Strategy using Moving Average
def breakout_strategy(df, breakout_window=20):
    df['Highest_High'] = df['Close'].rolling(window=breakout_window).max()
    df['Lowest_Low'] = df['Close'].rolling(window=breakout_window).min()
    df['Position'] = 0
    df.loc[(df['Close'] > df['Highest_High'].shift(1)) & (df['Close'] > df['MA_10']), 'Position'] = 1
    df.loc[(df['Close'] < df['Lowest_Low'].shift(1)) & (df['Close'] < df['MA_10']), 'Position'] = -1
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df


# In[29]:


import pandas as pd

# Mean Reversion Strategy using RSI
def mean_reversion_strategy(df, window=20, threshold=0.05):
    df['Moving_Avg'] = df['Close'].rolling(window=window).mean()
    df['Deviation'] = (df['Close'] - df['Moving_Avg']) / df['Moving_Avg']
    df['Position'] = 0
    df.loc[(df['Deviation'] < -threshold) & (df['RSI'] < 30), 'Position'] = 1  # Buy below threshold and RSI < 30
    df.loc[(df['Deviation'] > threshold) & (df['RSI'] > 70), 'Position'] = -1  # Sell above threshold and RSI > 70
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df


# In[30]:


import pandas as pd

# News-Based Trading Strategy using RSI
def news_based_trading_strategy(df, sentiment_data):
    df['Position'] = 0
    df.loc[(sentiment_data['Sentiment'] == 'Positive') & (df['RSI'] < 70), 'Position'] = 1
    df.loc[(sentiment_data['Sentiment'] == 'Negative') & (df['RSI'] > 30), 'Position'] = -1
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df


# In[31]:


import pandas as pd

# Seasonal Strategy using EMA_10
def seasonal_trading_strategy(df, buy_months=[1, 2, 11, 12], sell_months=[5, 6, 7, 8]):
    df['Position'] = 0
    df['Month'] = df['Date'].dt.month
    df.loc[(df['Month'].isin(buy_months)) & (df['Close'] > df['EMA_10']), 'Position'] = 1
    df.loc[(df['Month'].isin(sell_months)) & (df['Close'] < df['EMA_10']), 'Position'] = -1
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df


# In[32]:


# Backtest all strategies
def backtest_all_strategies(df):
    strategies = {
        "Momentum Strategy": momentum_strategy(df.copy()),
        "Range Trading Strategy": range_trading_strategy(df.copy()),
        "Breakout Strategy": breakout_strategy(df.copy()),
        "Mean Reversion Strategy": mean_reversion_strategy(df.copy()),
        "Seasonal Strategy": seasonal_trading_strategy(df.copy()),
    }
    results = {name: (strategy['Cumulative_Return'].iloc[-1] - 1) * 100 for name, strategy in strategies.items()}
    return results

# Backtest, calculate CAGR, and identify best strategies
all_results = []
for file_name in commodity_files:
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    results = backtest_all_strategies(df)
    for strategy, return_pct in results.items():
        all_results.append({
            'Commodity': file_name.replace("_features.csv", ""),
            'Strategy': strategy,
            'Cumulative Return (%)': return_pct
        })

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
results_df['Cumulative_Return_Factor'] = (results_df['Cumulative Return (%)'] / 100) + 1
data_period_years = 10
results_df['CAGR (%)'] = ((results_df['Cumulative_Return_Factor'] ** (1 / data_period_years)) - 1) * 100

# Group results by Commodity and Strategy
final_cagr_df = results_df.groupby(['Commodity', 'Strategy'], as_index=False)['CAGR (%)'].mean()
best_strategies = final_cagr_df.loc[final_cagr_df.groupby("Commodity")['CAGR (%)'].idxmax()]

# Display results
print("\n--- Cumulative Returns for Each Strategy by Commodity ---")
print(results_df[['Commodity', 'Strategy', 'Cumulative Return (%)']])
print("\n--- CAGR for Each Strategy by Commodity ---")
print(final_cagr_df)
print("\n--- Best Trading Strategy for Each Commodity Based on CAGR ---")
print(best_strategies)

# Save results to CSV files
results_df.to_csv("commodity_strategy_cumulative_returns.csv", index=False)
best_strategies.to_csv("best_trading_strategy_per_commodity.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# 

# 

# In[ ]:





# # XGBoost

# In[33]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np


# In[34]:


'''
def train_xgboost_model(df, commodity_name):
    # Ensure necessary features exist
    if 'Future_Close' not in df.columns:
        df['Future_Close'] = df['Close'].shift(-1)  # Target is next day's Close price

    # Drop rows with missing values
    df = df.dropna()

    # Define features (X) and target (y)
    X = df.drop(columns=['Future_Close', 'Date'], errors='ignore')
    y = df['Future_Close']

    # Train-test split (80/20 without shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train XGBoost model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print results
    print(f"{commodity_name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    # Save predictions
    predictions_df = X_test.copy()
    predictions_df['Actual_Close'] = y_test
    predictions_df['Predicted_Close'] = y_pred
    predictions_df.to_csv(f"xgboost_predictions_{commodity_name}.csv", index=False)

    return {"Commodity": commodity_name, "MAE": mae, "RMSE": rmse}
'''


# In[35]:


def train_xgboost_model(df, commodity_name):
    """
    Train XGBoost model on the first 90% of data.
    Use 70% for training and 20% for testing.
    Leave the last 10% completely hidden for backtesting.
    """
    # Ensure necessary features exist
    if 'Future_Close' not in df.columns:
        df['Future_Close'] = df['Close'].shift(-1)  # Target is next day's Close price

    # Drop rows with missing values
    df = df.dropna()

    # Split into 90% usable data and 10% hidden data
    cutoff = int(len(df) * 0.9)
    usable_df = df.iloc[:cutoff]      # First 90%
    hidden_df = df.iloc[cutoff:]      # Last 10% (hidden data)

    # Define features (X) and target (y) for usable data
    X = usable_df.drop(columns=['Future_Close', 'Date'], errors='ignore')
    y = usable_df['Future_Close']

    # Split usable data: 70% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.222, shuffle=False)

    # Initialize and train XGBoost model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_test_pred = model.predict(X_test)

    # Evaluate on the test set
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"{commodity_name} Results:")
    print(f"Test Set -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save test set predictions
    predictions_test_df = X_test.copy()
    predictions_test_df['Actual_Close'] = y_test
    predictions_test_df['Predicted_Close'] = y_test_pred
    predictions_test_df.to_csv(f"xgboost_predictions_test_{commodity_name}.csv", index=False)

    # Save hidden data (for backtesting without predictions)
    hidden_df.to_csv(f"xgboost_hidden_data_{commodity_name}.csv", index=False)

    return {
        "Commodity": commodity_name,
        "Test MAE": mae,
        "Test RMSE": rmse
    }


# In[36]:


# Directory containing the feature-enhanced data files
data_dir = "commodity_data"
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_features.csv")]

# Storage for results
results = []

# Loop through all commodities
for file_name in commodity_files:
    commodity_name = file_name.replace("_features.csv", "")
    file_path = os.path.join(data_dir, file_name)

    # Load the data
    if os.path.exists(file_path):
        print(f"Training XGBoost for {commodity_name}...")
        df = pd.read_csv(file_path)

        # Ensure date column is parsed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Train and evaluate XGBoost
        result = train_xgboost_model(df, commodity_name)
        results.append(result)
    else:
        print(f"File not found: {file_path}")

# Combine results into a DataFrame
results_df = pd.DataFrame(results)

# Display results
print("\n--- XGBoost Results for All Commodities ---")
print(results_df)

# Save final results
results_df.to_csv("xgboost_results_summary.csv", index=False)


# #Fine Tuning

# In[37]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np


# In[38]:


from sklearn.model_selection import TimeSeriesSplit

def finetune_xgboost(X_train, y_train):
    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.70, 0.80],
        'colsample_bytree': [0.7, 0.8],
        'gamma': [0, 0.1, 0.5]
    }

    # TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize XGBoost model
    model = XGBRegressor(random_state=42)

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=-1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


# In[39]:


# Directory containing the feature-enhanced data files
data_dir = "commodity_data"
commodity_files = [f for f in os.listdir(data_dir) if f.endswith("_features.csv")]

# Storage for results
finetune_results = []

# Loop through all commodities
for file_name in commodity_files:
    commodity_name = file_name.replace("_features.csv", "")
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        print(f"Fine-tuning XGBoost for {commodity_name}...")

        # Load data
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        df['Future_Close'] = df['Close'].shift(-1)
        df = df.dropna()

        # Define features and target
        X = df.drop(columns=['Future_Close', 'Date'], errors='ignore')
        y = df['Future_Close']

        # Train-test split (80/20 without shuffle)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Fine-tune XGBoost
        best_model = finetune_xgboost(X_train, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{commodity_name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

        # Store results
        finetune_results.append({
            "Commodity": commodity_name,
            "MAE": mae,
            "RMSE": rmse
        })

        # Save predictions
        predictions_df = X_test.copy()
        predictions_df['Actual_Close'] = y_test
        predictions_df['Predicted_Close'] = y_pred
        predictions_df.to_csv(f"finetuned_xgboost_predictions_{commodity_name}.csv", index=False)
    else:
        print(f"File not found: {file_path}")

# Combine results into a DataFrame
results_df = pd.DataFrame(finetune_results)

# Display results
print("\n--- Fine-Tuned XGBoost Results for All Commodities ---")
print(results_df)

# Save final results
results_df.to_csv("finetuned_xgboost_results_summary.csv", index=False)


# # SIGNAL

# In[40]:


import pandas as pd
import os
import numpy as np


# In[41]:


def generate_signals(df):
    """
    Generate trading signals based on line intersection logic.
    Buy (+1), Sell (-1), Hold (0).
    """
    signals = []

    # Loop through the data from the 2nd row onwards
    for i in range(1, len(df)):
        # Hypothetically join previous day's closing and predicted prices to today's prices
        prev_actual = df['Actual_Close'].iloc[i-1]
        prev_predicted = df['Predicted_Close'].iloc[i-1]
        today_actual = df['Actual_Close'].iloc[i]
        today_predicted = df['Predicted_Close'].iloc[i]

        # Check for intersection
        if (prev_actual < prev_predicted and today_actual > today_predicted) or \
           (prev_actual > prev_predicted and today_actual < today_predicted):
            # Intersection occurred
            if today_actual < today_predicted:
                signals.append(-1)  # Sell signal
            elif today_actual > today_predicted:
                signals.append(1)   # Buy signal
        else:
            signals.append(0)  # Hold signal

    # The first row has no signal as we need previous day's data
    signals.insert(0, 0)  # Append 0 for the first row

    # Add signals to the DataFrame
    df['Signal'] = signals
    return df


# In[42]:


# Directory containing the XGBoost prediction results
output_dir = "./"  # Adjust as needed
prediction_files = [
    f"finetuned_xgboost_predictions_coffee.csv",
    f"finetuned_xgboost_predictions_wheat.csv",
    f"finetuned_xgboost_predictions_natural_gas.csv",
    f"finetuned_xgboost_predictions_sugar.csv",
    f"finetuned_xgboost_predictions_gold.csv",
    f"finetuned_xgboost_predictions_silver.csv",
    f"finetuned_xgboost_predictions_cotton.csv",
    f"finetuned_xgboost_predictions_copper.csv",
    f"finetuned_xgboost_predictions_corn.csv",
    f"finetuned_xgboost_predictions_crude_oil.csv"
]

# Storage for results
all_signals = []

# Loop through each prediction file
for file_name in prediction_files:
    commodity_name = file_name.replace("finetuned_xgboost_predictions_", "").replace(".csv", "")
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        print(f"Generating signals for {commodity_name}...")

        # Load the prediction data
        df = pd.read_csv(file_path)

        # Ensure the data has required columns
        if 'Actual_Close' in df.columns and 'Predicted_Close' in df.columns:
            # Generate signals
            df_with_signals = generate_signals(df)

            # Save the results to a new CSV
            output_file = f"signals_{commodity_name}.csv"
            df_with_signals.to_csv(output_file, index=False)

            # Store in results list for summary
            all_signals.append({"Commodity": commodity_name, "Output File": output_file})
        else:
            print(f"Missing required columns in {file_name}")
    else:
        print(f"File not found: {file_path}")

# Display summary
signals_summary = pd.DataFrame(all_signals)
print("\n--- Signal Generation Summary ---")
print(signals_summary)


# # Trading and Backtesting
# 

# In[43]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


# In[44]:


# Directory for hidden data
hidden_data_dir = "/Users/avssp/Desktop/commodity_data"  # Adjust if needed
hidden_files = [
    "xgboost_hidden_data_coffee.csv",
    "xgboost_hidden_data_wheat.csv",
    "xgboost_hidden_data_natural_gas.csv",
    "xgboost_hidden_data_sugar.csv",
    "xgboost_hidden_data_gold.csv",
    "xgboost_hidden_data_silver.csv",
    "xgboost_hidden_data_cotton.csv",
    "xgboost_hidden_data_copper.csv",
    "xgboost_hidden_data_corn.csv",
    "xgboost_hidden_data_crude_oil.csv"
]


# In[45]:


def trading_execution(hidden_df, model, commodity_name):
    print(f"Executing trades for {commodity_name}...")

    # Ensure target column exists
    hidden_df['Future_Close'] = hidden_df['Close'].shift(-1)
    hidden_df = hidden_df.dropna()

    # Define features (X) and ensure consistency with training features
    X_hidden = hidden_df.drop(columns=['Future_Close', 'Actual_Close', 'Predicted_Close'], errors='ignore')
    X_hidden = X_hidden.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Ensure feature names match
    missing_cols = set(model.feature_names_in_) - set(X_hidden.columns)
    if missing_cols:
        for col in missing_cols:
            X_hidden[col] = 0  # Add missing columns with default values

    # Generate predictions
    hidden_df['Predicted_Close'] = model.predict(X_hidden[model.feature_names_in_])

    # Generate signals based on predictions
    hidden_df['Signal'] = 0
    hidden_df.loc[hidden_df['Predicted_Close'] > hidden_df['Close'], 'Signal'] = 1  # Buy signal
    hidden_df.loc[hidden_df['Predicted_Close'] < hidden_df['Close'], 'Signal'] = -1  # Sell signal

    # Calculate returns based on signals
    hidden_df['Daily_Return'] = hidden_df['Close'].pct_change()
    hidden_df['Strategy_Return'] = hidden_df['Signal'].shift(1) * hidden_df['Daily_Return']
    hidden_df['Cumulative_PnL'] = (1 + hidden_df['Strategy_Return']).cumprod()

    # Save signals and PnL data
    signal_file = os.path.join(hidden_data_dir, f"signals_with_pnl_{commodity_name}.csv")
    hidden_df.to_csv(signal_file, index=False)
    print(f"Signals and PnL data saved for {commodity_name} in {signal_file}")

    return hidden_df


# In[46]:


def plot_pnl(hidden_df, commodity_name):
    plt.figure(figsize=(14, 6))
    plt.plot(hidden_df.index, hidden_df['Cumulative_PnL'], label='Cumulative PnL', color='blue')
    plt.title(f"Cumulative PnL for {commodity_name.capitalize()} (2023-2024)")
    plt.xlabel("Index")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(hidden_data_dir, f"pnl_chart_{commodity_name}.png"))
    plt.show()


# In[49]:


final_results=[]

for hidden_file in hidden_files:
    commodity_name = hidden_file.replace("xgboost_hidden_data_", "").replace(".csv", "")
    file_path = os.path.join(hidden_data_dir, hidden_file)

    if os.path.exists(file_path):
        print(f"Processing {commodity_name}...")

        # Load the hidden data
        hidden_df = pd.read_csv(file_path)

        # Create 'Future_Close' column if it doesn't exist
        if 'Future_Close' not in hidden_df.columns:
            hidden_df['Future_Close'] = hidden_df['Close'].shift(-1)  # Shift the 'Close' column

        # Drop rows with missing 'Future_Close' values (last row in the dataset)
        hidden_df = hidden_df.dropna(subset=['Future_Close'])

        # Ensure the fine-tuned model is loaded
        model_file = f"finetuned_xgboost_predictions_{commodity_name}.csv"
        model_path = os.path.join(hidden_data_dir, model_file)
        if os.path.exists(model_path):
            # Train the model on usable data
            df = pd.read_csv(model_path)

            # Create 'Future_Close' for the model training dataset if not present
            if 'Future_Close' not in df.columns:
                df['Future_Close'] = df['Close'].shift(-1)
            df = df.dropna(subset=['Future_Close'])

            # Select only numeric columns and exclude non-feature columns
            X = df.drop(columns=['Future_Close', 'Actual_Close', 'Predicted_Close'], errors='ignore')
            X = X.select_dtypes(include=[np.number])  # Keep only numeric columns
            y = df['Future_Close']

            # Retrain the model with fine-tuned parameters
            model = XGBRegressor(**{
                'n_estimators': 300,
                'max_depth': 7,
                'learning_rate': 0.03,
                'subsample': 0.75,
                'colsample_bytree': 0.8,
                'gamma': 0.1
            })
            model.fit(X, y)


            # Execute trading logic
            hidden_df = trading_execution(hidden_df, model, commodity_name)

            # Plot and save PnL chart
            plot_pnl(hidden_df, commodity_name)

            # Append final cumulative return for summary
            final_pnl = hidden_df['Cumulative_PnL'].iloc[-1] - 1
            final_results.append({"Commodity": commodity_name, "Final_PnL (%)": final_pnl})
        else:
            print(f"Model not found for {commodity_name}")
    else:
        print(f"File not found: {file_path}")


# In[50]:


# Display final results
final_results_df = pd.DataFrame(final_results)
print("\n--- Final Trading Results for All Commodities ---")
print(final_results_df)

# Save final results to CSV
final_results_df.to_csv(os.path.join(hidden_data_dir, "final_trading_results.csv"), index=False)


# In[ ]:




