# AG02, Marco Renzo Di Toro, 150, GES

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

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


def read_sample():
    region = ""
    while region not in REGIONS:
        region = input("Region (Lisbon, Oporto ou Other): ").strip().capitalize()
    values = [REGIONS[region]]
    for name in FEATURES[1:]:
        values.append(float(input(f"{name} (u.m.): ")))
    return pd.DataFrame([values], columns=FEATURES)


def main():
    df = load_data()
    model = train(df)
    names = {v: k for k, v in CHANNELS.items()}
    while input("Classificar uma amostra? (s/n): ").strip().lower() == "s":
        sample = read_sample()
        print("Canal previsto:", names[int(model.predict(sample)[0])])


if __name__ == "__main__":
    main()


# Retail   ---    HoReCa 
# Other    ---    Other
# 12669    ---    13265
# 9656     ---    1196
# 7561     ---    4221
# 214      ---    6404
# 2674     ---    507
# 1338     ---    1788
