
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def curate_data():
    """
    This function loads the raw data, performs cleaning and preprocessing, and saves
    the curated data (train and test sets) as pickle files.
    """
    # Load the dataset
    df = pd.read_csv('dataset.csv', sep='|')

    # --- Exploratory Data Analysis (EDA) ---
    print("Performing EDA...")

    # Create a directory for EDA plots
    eda_path = 'eda_plots'
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    # 1. Target variable distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Response', data=df)
    plt.title('Distribution of Response Variable')
    plt.savefig(os.path.join(eda_path, 'response_distribution.png'))
    plt.close()

    # 2. Numerical feature distributions
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features.remove('Response')

    for col in numerical_features:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(eda_path, f'{col}_distribution.png'))
        plt.close()

    # 3. Correlation matrix of numerical features
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numerical_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(os.path.join(eda_path, 'correlation_matrix.png'))
    plt.close()

    print(f"EDA plots saved in '{eda_path}' directory.")
    # --- End of EDA ---

    # --- Data Cleaning and Preprocessing ---
    print("Cleaning and preprocessing data...")

    # Handle 'Feature_ps_3'
    # 999 often means 'not previously contacted'. Let's treat it as a separate category.
    df['Feature_ps_3'] = df['Feature_ps_3'].apply(lambda x: 'not_contacted' if x == 999 else 'contacted')

    # Separate features and target
    X = df.drop('Response', axis=1)
    y = df['Response']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_features]), 
                                   columns=encoder.get_feature_names_out(categorical_features),
                                   index=X_train.index)
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_features]), 
                                  columns=encoder.get_feature_names_out(categorical_features),
                                  index=X_test.index)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_features]), 
                                  columns=numerical_features, 
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_features]), 
                                 columns=numerical_features, 
                                 index=X_test.index)

    # Combine processed features
    X_train_processed = pd.concat([X_train_scaled, X_train_encoded], axis=1)
    X_test_processed = pd.concat([X_test_scaled, X_test_encoded], axis=1)

    # Save the curated data
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/train.pkl', 'wb') as f:
        pickle.dump({'X': X_train_processed, 'y': y_train}, f)

    with open('data/test.pkl', 'wb') as f:
        pickle.dump({'X': X_test_processed, 'y': y_test}, f)
        
    # Save the scaler and encoder
    with open('data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('data/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    print("Data curation complete. Train and test sets saved in 'data/' directory.")

if __name__ == '__main__':
    curate_data()
