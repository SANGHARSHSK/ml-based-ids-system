import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os

class DataPipeline:
    def __init__(self, filepath,sample_size=100000):
        self.filepath     =filepath
        self.sample_size  =sample_size  # limit for low ram 
        self.scaler       =StandardScaler()
        self.features     =None

    def load_data(self):
        print("Loading data...")
        df = pd.read_csv(self.filelpath,nrows=self.sample_size)
        df.columns = df.columns.str.strip()

        # memory optimization
        for col in df.select_dtypes('float64').columns:
            df[col] = df[col].astypes('float32')
        print(f"Loaded shape: {df.shape}")
        return df
    def clean_data(self,df):
        print("Cleaning data...")

        # Drop non-features columns
        drop_cols = ['Flow ID','Source IP','Destinatinon IP','Source Port','TImeStamp']
        df.drop(drop_cols, axis =1, errors='ignore',inplace=True)
        # Handles inf & null vlaues
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.drop_duplicates(inplace=True)

        print(f"clean shape: {df.shape}")
        return df
    def encode_lables(self,df):
        print("Encoding lables...")

        #binary encoding (0=Normal,1=attack)
        df['Label'] = df['Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)

        #Also save multiclass labels as for future scope
        attack_map={}
        for label in df['Label'].unique():
            attack_map[label] = label
        print(f"Class distribution:\n{df['Label'].value_counts()}")
        return df
    
    def balance_data(self,X,y):
        print("Balanlce classes with SMOTE...")
        print(f"Before SMOTE: {pd.series(y).value_counts().to_dict()}")

        # Light smote for low RAM
        smote = SMOTE(random_state=42,
                      k_neighbors=3,
                      sampling_strategy=0.5)
        X_res,y_res =smote.fit_resample(X,y)

        print(f"After SMOTE: {pd.series(y_res).value_counts().to_dict()}")
        return X_res,y_res
    def scale_features(self,X_train,X_test):
        print("scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled,X_test_scaled 

    def run(self):
        #full pipeline
        df = self.load_data()
        df = self.clean_data()
        df = self.encode_lables(df)

        X = df.drop('Label',axis=1)
        y = df['Label']  

        self.features = X.columns.tolist() 
        X_train, X_test, y_trian,y_test =train_test_split(
            X,y,
            test_size = 0.2,
            random_state = 42,
            stratify = y
        )  
        # Balance only training data

        X_train, y_train = self.balance_data(
            X_train, y_train)
        
        # scale
        X_train, X_test =self.scale_features(
            X_train,X_test
        )

        # save everything 
        os.makedirs('model',exist_ok=True)
        os.makedirs('data/processed',exist_ok=True)

        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_test.npy',  X_test)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_test.npy',  y_test)

        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('models/features.pkl', 'wb') as f:
            pickle.dump(self.features, f)

        print("✅ Pipeline complete! Data saved.")
        return X_train, X_test, y_train, y_test
# Run pipeline
if __name__ == "__main__":
    pipeline = DataPipeline(
        "data/network_traffic.csv",
        sample_size=100000
    )
    pipeline.run()

