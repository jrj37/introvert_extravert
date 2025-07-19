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

def encode_categorical(df: pl.DataFrame) -> pl.DataFrame:
    """
    Encode categorical variables in the DataFrame.
        Args:
            df (pl.DataFrame): Input DataFrame with categorical variables.
        Returns:
            pl.DataFrame: DataFrame with encoded categorical variables.
        """
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == pl.String:
            df = df.with_columns([
                pl.Series(column, label_encoder.fit_transform(df[column].to_numpy()))
            ])
    return df

def preprocess_data(df: pl.DataFrame, train: bool = True) -> pl.DataFrame:
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
    id = df.get_column("id")
    df = df.drop("id")
    if train:
        # Encode categorical variables
        x = df.drop("Personality")
        y = df["Personality"].to_numpy()
        x = encode_categorical(x)
    else:
        x = encode_categorical(df)

    data_array = x.to_numpy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)
    if train:
        return pl.DataFrame(scaled_data, schema=x.schema), pl.Series("Personality", y)
    else:
        # For test data, we don't have labels
        return pl.DataFrame(scaled_data, schema=x.schema), id