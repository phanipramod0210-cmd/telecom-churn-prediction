#!/usr/bin/env python3
import argparse, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna({'TotalCharges': 0})
    le = LabelEncoder()
    for c in ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','Contract','PaperlessBilling','PaymentMethod']:
        if c in df.columns:
            df[c] = le.fit_transform(df[c].astype(str))
    if 'Churn' in df.columns:
        df['Churn'] = le.fit_transform(df['Churn'].astype(str))
    return df

def main(args):
    df = load_data(args.data)
    y = df['Churn'] if 'Churn' in df.columns else pd.Series([0]*len(df))
    X = df.drop(columns=['Churn']) if 'Churn' in df.columns else df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)
    print("Saved model to", args.out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='model/churn_model.joblib')
    args = p.parse_args()
    main(args)
