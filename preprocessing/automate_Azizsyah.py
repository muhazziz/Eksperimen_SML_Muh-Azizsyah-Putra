import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(url):
    return pd.read_csv(url)

def preprocess_data(df):
    df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df_clean = df_clean.dropna()

    le_gender = LabelEncoder()
    df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])
    df_clean = pd.get_dummies(df_clean, columns=['Geography'], drop_first=True)

    X = df_clean.drop('Exited', axis=1)
    y = df_clean['Exited']
    return X, y

def main():
    # 1. Load Data
    url = "https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv"
    df = load_data(url)
    
    # 2. Preprocess Data
    X, y = preprocess_data(df)

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Scaling
    scaler = StandardScaler()
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    output_dir = "churn_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(f"{output_dir}/train.csv", index=False)
    test_data.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"Preprocessing selesai. Dataset tersimpan di folder {output_dir}/")

if __name__ == "__main__":
    main()