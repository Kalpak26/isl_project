# src/model_trainer.py
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

class SignLanguageModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def load_data(self):
        raw_data_path = os.path.join(self.project_root, 'data', 'raw')
        all_data = []
        
        # Load all CSV files from each sign folder
        for sign_folder in os.listdir(raw_data_path):
            sign_path = os.path.join(raw_data_path, sign_folder)
            if os.path.isdir(sign_path):
                for file in os.listdir(sign_path):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(sign_path, file))
                        all_data.append(df)
        
        # Combine all data
        data = pd.concat(all_data, ignore_index=True)
        
        # Separate features and labels
        X = data.drop('sign', axis=1)
        y = data['sign']
        
        return X, y

    def train(self):
        # Load and split data
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save model
        model_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to: {model_path}")

def main():
    trainer = SignLanguageModel()
    trainer.train()

if __name__ == "__main__":
    main()