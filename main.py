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


def main():
    df = load_data()
    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()
