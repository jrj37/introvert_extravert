import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Load data from a CSV file into a Polars DataFrame.
        Args:
            file_path (str): Path to the CSV file.
        Returns:
            pl.DataFrame: A Polars DataFrame containing the loaded data.
        """
    return pl.read_csv(file_path)

def preprocess_data(df) -> pl.DataFrame:
    """
    Preprocess the DataFrame by handling missing values and scaling features.
        Args:
            df (pl.DataFrame): Input DataFrame to preprocess.
        Returns:
            pl.DataFrame: Preprocessed DataFrame with scaled features.
        """
    # Handle missing values
   # df = df.fill_null(strategy="mean")
    # Convert to numpy array for scaling
    label_encoder = LabelEncoder()
    df = df.drop_nulls()
    #check personality column % of classes
    personality_counts = df["Personality"].value_counts()
    print("Personality class distribution:")
    print(personality_counts)
    # Encode categorical variables
    x = df.drop("Personality")
    y = df["Personality"].to_numpy()
    for column in x.columns:
        print(f"Processing column: {column}")
        if x[column].dtype == pl.String:
            x = x.with_columns([
            pl.Series(column, label_encoder.fit_transform(x[column].to_numpy()))
            ])

    data_array = x.to_numpy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)

    return pl.DataFrame(scaled_data, schema=x.schema), pl.Series("Personality", y)