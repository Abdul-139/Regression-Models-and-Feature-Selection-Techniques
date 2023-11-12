# -*- coding: utf-8 -*-
"""Data Preprocessing Template"""

# Importing the relevant libraries
import pandas as pd
import seaborn as sns

def load_dataset(url):
    """
    Load a dataset from a given URL.

    Parameters:
    - url (str): The URL of the dataset.

    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(url)
    return data

def clean_data(data, columns_to_drop=None, dropna_threshold=0.05):
    """
    Clean the dataset by handling missing values and providing descriptive statistics.

    Parameters:
    - data (pd.DataFrame): The dataset to be cleaned.
    - columns_to_drop (list): List of columns to be dropped. Default is None.
    - dropna_threshold (float): Threshold for dropping rows with missing values. Default is 0.05.

    Returns:
    - pd.DataFrame: The cleaned dataset.
    """
    # Display missing values
    print("Missing values:")
    print(data.isnull().sum())

    # Drop specified columns if provided
    if columns_to_drop:
        data = data.drop(columns_to_drop, axis=1)

    # Drop rows with missing values if less than threshold
    if dropna_threshold > 0:
        data = data.dropna(axis=0, thresh=int(len(data.columns) * (1 - dropna_threshold)))

    # Display missing values after cleaning
    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    # Display descriptive statistics
    print("\nDescriptive statistics:")
    print(data.describe(include='all'))

    # Display dataset information
    print("\nDataset information:")
    print(data.info())

    # Display column names
    print("\nColumn names:")
    print(data.columns)

    return data

def visualize_data(data, columns):
    """
    Visualize the data using a pair plot and correlation matrix.

    Parameters:
    - data (pd.DataFrame): The dataset to be visualized.
    - columns (list): List of columns for visualization.

    Returns:
    - None
    """
    # Pair plot
    sns.pairplot(data[columns])

    # Correlation matrix
    print("\nCorrelation matrix:")
    print(data.corr())

# Example usage
if __name__ == "__main__":
    # Example URL
    dataset_url = 'https://raw.githubusercontent.com/AjStephan/curcumin/main/PubChem_compound_list.csv'

    # Load dataset
    loaded_data = load_dataset(dataset_url)

    # Columns to drop during cleaning
    columns_to_drop = ['cmpdname', 'isosmiles', 'mw', 'exactmass', 'monoisotopicmass']

    # Clean the data
    cleaned_data = clean_data(loaded_data, columns_to_drop)

    # Columns for visualization
    visualization_columns = ['xlogp', 'polararea', 'heavycnt', 'hbonddonor', 'hbondacc', 'rotbonds']

    # Visualize the data
    visualize_data(cleaned_data, visualization_columns)
