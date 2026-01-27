from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import pickle




def drop_missing_data(df, threshold=15, axis=0):
    """
    Drops rows or columns from a DataFrame based on a missing data threshold.
    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        threshold (float): The threshold for missing data percentage to drop.
        axis (int): Axis along which to operate (0 for rows, 1 for columns).
    Returns:
        pd.DataFrame: The DataFrame after dropping rows/columns exceeding the missing data threshold.
    """
    # Calculate the percentage of missing data
    if axis == 0:  # Rows
        missing_percentage = df.isnull().mean(axis=1) * 100
    elif axis == 1:  # Columns
        missing_percentage = df.isnull().mean(axis=0) * 100
    else:
        raise ValueError("Invalid axis. Use 0 for rows or 1 for columns.")

    # Identify indices where the missing percentage is greater than the threshold
    indices_to_drop = missing_percentage[missing_percentage > threshold].index

    # Drop rows/columns exceeding the missing data threshold
    df_dropped = df.drop(index=indices_to_drop) if axis == 0 else df.drop(columns=indices_to_drop)

    return df_dropped



def impute_missing_values(df, numerical_cols, categorical_cols):
    """
    Imputes missing values in a DataFrame for numerical and categorical columns.

    Parameters:
        df (pd.DataFrame): The DataFrame with missing values to impute.
        numerical_cols (list of str): List of column names for numerical data.
        categorical_cols (list of str): List of column names for categorical data.

    Returns:
        pd.DataFrame: The DataFrame with imputed values.
    """
    # Imputing numerical data
    if numerical_cols:  # Ensure there are numerical columns to process
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols].apply(pd.to_numeric, errors='coerce'))

    # Imputing categorical data
    if categorical_cols:  # Ensure there are categorical columns to process
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df



def process_and_encode_categorical(df, categorical_cols):
    """
    Handles categorical data that may contain numerical (floating-point) values by converting
    them to 'unknown', and then applies ordinal encoding to the categorical columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        categorical_cols (list of str): List of categorical column names.

    Returns:
        pd.DataFrame: The DataFrame with processed and encoded categorical columns.
    """
    for col in categorical_cols:
        # Check if the column has any floating-point numbers or NaNs
        if df[col].dtype == float:
            # Replace NaN or any float with 'unknown'
            df[col] = df[col].apply(lambda x: 'unknown' if pd.isna(x) or isinstance(x, float) else x)
        else:
            # Convert all entries to string to ensure consistency for encoding
            df[col] = df[col].astype(str)

    # Encoding categorical variables
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    return df


def identify_column_types(df, exclude_numerical=None):
    """
    Identifies categorical and numerical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        exclude_numerical (list of str, optional): List of column names to exclude from numerical columns.

    Returns:
        tuple: Returns two lists, one for categorical columns and one for numerical columns.
    """
    if exclude_numerical is None:
        exclude_numerical = []

    # Identify columns by data type
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude specified columns from numerical columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_numerical]

    return categorical_cols, numerical_cols





def calculate_missing_data(df):
    """
    Calculates and prints the percentage of missing data per row and per column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: Returns two pandas Series, one for row-wise and one for column-wise missing data percentages.
    """
    # Calculate the percentage of missing data per row
    row_missing_percentage = df.isnull().mean(axis=1) * 100
    # Calculate the percentage of missing data per column
    col_missing_percentage = df.isnull().mean() * 100

    # Optionally print the results
    print("Percentage of Missing Data per Row:")
    print(row_missing_percentage)
    print("\nPercentage of Missing Data per Column:")
    print(col_missing_percentage)

    return row_missing_percentage, col_missing_percentage


def remove_low_variance_features(df, threshold=0.2):
    """
    Removes features with variance below a specified threshold.

    Parameters:
        df (pd.DataFrame): The DataFrame from which to remove low-variance features.
        threshold (float): The threshold for variance below which features are removed.

    Returns:
        pd.DataFrame: A DataFrame with only the features that meet the variance threshold.
    """
    # Create the VarianceThreshold object with the specified threshold
    selector = VarianceThreshold(threshold=threshold)

    # Fit the selector to the data
    df_reduced = selector.fit_transform(df)

    # Get the columns that were kept by the selector
    cols = df.columns[selector.get_support()]

    # Create a DataFrame with the retained features
    df_reduced = pd.DataFrame(df_reduced, columns=cols, index=df.index)

    # Print information about the reduction
    print(f"Reduced from {df.shape[1]} to {df_reduced.shape[1]} features.")

    return df_reduced




def remove_highly_correlated_features(df, correlation_threshold=0.95):
    """
    Removes features that are highly correlated with other features in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame from which to remove highly correlated features.
        correlation_threshold (float): Threshold above which features are considered highly correlated.

    Returns:
        pd.DataFrame: A DataFrame with reduced features based on the correlation threshold.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns that have correlation greater than the specified threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    # Drop the highly correlated features
    df_reduced = df.drop(to_drop, axis=1)

    # Print information about the number of dropped features
    print(f"Dropped {len(to_drop)} highly correlated features.")

    return df_reduced

# Function to apply transformation based on skewness
def transform_skewed_data(df, columns):
    # Define skewness threshold
    skewness_threshold = 1.0

    for col in columns:
        skewness = df[col].skew()
        print(f"Skewness for {col}: {skewness}")

        # Check if the column is skewed based on the threshold
        if skewness > skewness_threshold:
            # Apply log transformation (you can choose different transformations)
            df[col + '_log'] = np.log(df[col] + 1)  # Adding 1 to handle zero values safely
            print(f"Applied log transformation to {col}")
        elif skewness < -skewness_threshold:
            # Apply other transformations as needed, e.g., for negative skewness
            df[col + '_reciprocal'] = 1 / (df[col] + 1)  # Example for reciprocal transformation
            print(f"Applied reciprocal transformation to {col}")

    return df


def reduce_features_via_clustering(df, distance_threshold=0.5):
    # Remove or fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove constant columns to avoid NaN in correlation matrix
    df = df.loc[:, (df != df.iloc[0]).any()]

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Handle potential NaN after correlation calculation
    corr_matrix.fillna(0, inplace=True)

    # Perform hierarchical clustering
    cluster = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=distance_threshold)
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    cluster.fit(corr_matrix.where(upper_triangle))

    # Map cluster labels to feature names
    feature_labels = pd.Series(cluster.labels_, index=corr_matrix.columns)
    representative_features = feature_labels.groupby(feature_labels).apply(lambda x: x.index[0])

    # Select representative features from the original DataFrame
    reduced_df = df[representative_features.values]

    print(f"Original number of features: {df.shape[1]}")
    print(f"Reduced number of features: {reduced_df.shape[1]}")

    return reduced_df


def report_missing_data(df):
    """
    Reports the percentage of missing data in each column and row of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: Returns two pandas Series, one for column-wise and one for row-wise missing data percentages.
    """
    # Calculate the percentage of missing data per column
    col_missing_percentage = df.isnull().mean() * 100
    col_missing_percentage = col_missing_percentage[col_missing_percentage > 0].sort_values(ascending=False)

    # Calculate the percentage of missing data per row
    row_missing_percentage = df.isnull().mean(axis=1) * 100
    row_missing_percentage = row_missing_percentage[row_missing_percentage > 0].sort_values(ascending=False)

    # Print the results
    if not col_missing_percentage.empty:
        print("Percentage of Missing Data per Column:")
        print(col_missing_percentage)
    else:
        print("No missing data in any column.")

    if not row_missing_percentage.empty:
        print("\nPercentage of Missing Data per Row:")
        print(row_missing_percentage)
    else:
        print("No missing data in any row.")

    return col_missing_percentage, row_missing_percentage



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def standardize_text(data):
    """Converts text to lower case and removes non-alphanumeric characters."""
    data = data.lower()
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    return data.strip()

def normalize_text(data):
    """Lemmatizes text and removes stopwords."""
    normalized_data = []
    for sentence in data:
        tokens = nltk.word_tokenize(sentence)
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
        normalized_data.append(' '.join(lemmatized))
    return normalized_data





def cluster_phrases_dbscan(data, eps=0.5, min_samples=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_

    # Create a dictionary to collect phrases for each cluster
    clustered_phrases = {}
    for label, phrase in zip(labels, data):
        if label not in clustered_phrases:
            clustered_phrases[label] = []
        clustered_phrases[label].append(phrase)

    return clustered_phrases

def process_categorical_data(df):
    """Processes categorical data by standardizing, normalizing, and clustering."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].astype(str).apply(standardize_text)
        df[column] = normalize_text(df[column].tolist())
        phrase_to_cluster = cluster_phrases_dbscan(df[column].tolist())
        df[column] = df[column].map(phrase_to_cluster)
    return df

