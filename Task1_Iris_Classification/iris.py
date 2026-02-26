# ==============================
# Iris Flower Classification (Using Given CSV Dataset)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


def main():

    # 1️⃣ Load Dataset (Given CSV File)
    df = pd.read_csv("iris.csv")

    print("First 5 Rows:\n", df.head())
    print("\nDataset Shape:", df.shape)
    print("-" * 50)

    # 2️⃣ Features and Target
    X = df[['SepalLengthCm', 'SepalWidthCm', 
            'PetalLengthCm', 'PetalWidthCm']]

    y = df['Species']

    # 3️⃣ Convert Species (text) to numbers
    le = LabelEncoder()
    y = le.fit_transform(y)

    print("Classes:", le.classes_)
    print("-" * 50)

    # 4️⃣ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5️⃣ Model Training
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 6️⃣ Prediction
    y_pred = model.predict(X_test)

    # 7️⃣ Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 8️⃣ Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()