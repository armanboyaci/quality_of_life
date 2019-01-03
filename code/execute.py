import pandas as pd
from model import preprocess, train, predict

df = pd.read_csv("../data/train.csv", sep=";", decimal=",", index_col=[1,2,3])
df_train = preprocess(df)
m = train(df_train)

df = pd.read_csv("../data/test.csv", sep=";", decimal=",", index_col=[1,2,3])
df_test = preprocess(df, train=False)
df_test["predicted"] = predict(m, df_test)
print(df_test)