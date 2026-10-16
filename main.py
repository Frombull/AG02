# AG2 - Engenharia de Software (GES)
# Aluno: Marco Di Toro - matricula 150
# Classificacao do canal de vendas (HoReCa x Retail) com Decision Tree

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option("future.no_silent_downcasting", True)

CSV_PATH = "wholesale.csv"
CHANNELS = {"HoReCa": 0, "Retail": 1}
REGIONS = {"Lisbon": 0, "Oporto": 1, "Other": 2}
FEATURES = ["Region", "Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["Channel"] = df["Channel"].replace(CHANNELS).astype("int64")
    df["Region"] = df["Region"].replace(REGIONS).astype("int64")
    return df.reindex(columns=FEATURES + ["Channel"])


def train(df):
    X = df[FEATURES]
    y = df["Channel"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), target_names=["HoReCa", "Retail"]))
    return model


def main():
    df = load_data()
    train(df)


if __name__ == "__main__":
    main()
