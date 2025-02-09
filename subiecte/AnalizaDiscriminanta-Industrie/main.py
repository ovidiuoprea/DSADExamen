import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from seaborn import scatterplot


def codificare(t, variabile_calitative):
    for v in variabile_calitative:
        if not is_numeric_dtype(t[v]):
            t[v] = pd.Categorical(t[v]).codes + 1

def nan_replace_df(t: pd.DataFrame):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if is_numeric_dtype(t[v]):
            t.fillna({v: t[v].mean()}, inplace=True)
        else:
            t.fillna({v: t[v].mode()[0]}, inplace=True)

def grafic_scoruri(t: pd.DataFrame, y_test, clase):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot scoruri in primele 2 axe")
    ax.set_xlabel("C1")
    ax.set_ylabel("C2")
    scatterplot(x=t[:, 0], y=t[:, 1], hue=y_test, hue_order=clase, legend=True)

def show():
    plt.show()

# industrie = pd.read_csv("data_in/Industrie.csv", index_col=0)
# Cerinta A e foarte similara cu ce am facut la Canonica (cred)
# TODO: later

set_date = pd.read_csv("data_in/ProiectB.csv", index_col=0, dtype={"Mason": str, "Struct": str})
nan_replace_df(set_date)
codificare(set_date, ["Mason","Struct"])

predictori = list(set_date)[:-1]
tinta = list(set_date)[-1]


x_train, x_test, y_train, y_test = train_test_split(set_date[predictori], set_date[tinta], test_size=0.3)



model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)
clase = model_lda.classes_

q = len(clase)

etichete_clase = ["Componenta" + str(i+1) for i in range(q - 1) ]

scoruri = model_lda.transform(x_test)
df_s = pd.DataFrame(data=scoruri, index=x_test.index, columns=etichete_clase)

grafic_scoruri(scoruri, y_test, clase)
# show()

y_test_prezis = model_lda.predict(x_test)

print([y_test, y_test.values])
df_ytest = pd.DataFrame(y_test_prezis, y_test.index, ["LDA"])
df_ytest["Y_test"] = y_test.values

df_ytest.to_csv("data_out/predict_test.csv")

set_aplicare = pd.read_csv("data_in/ProiectB_apply.csv", index_col=0, dtype={"Mason": str, "Struct": str})
nan_replace_df(set_aplicare)
codificare(set_aplicare, ["Mason","Struct"])

y_aplicare = model_lda.predict(set_aplicare[predictori])
print(y_aplicare)

df_yaplicare = pd.DataFrame(y_aplicare, set_aplicare.index, ["LDA"])
df_yaplicare.name = "Set aplicare"

df_yaplicare.to_csv("data_out/predict_apply.csv")