import pandas as pd
import statsmodels.api as sm

def preprocess(df, train=True):
    columns = [ 
    "CPI_AGRWTH",
    "GDP_USD_CAP",

    "TER_Enrolment_rate",
    "EDU_TERTIARY_PC_WKGPOP",
   
    "Access to Clean Fuels and Technologies for cooking (% of total population) [2 1_ACCESS CFT TOT]",
    "Access to electricity (% of total population) [1 1_ACCESS ELECTRICITY TOT]",
    "%_Sanitation_Access",
       ]
    if train:
        columns = ["Quality_of_life_measure"] + columns
                  
    df = df[columns].dropna()
    df_filled = df.groupby("LOCATION").transform(lambda x: x.fillna(method='ffill')).reset_index().dropna()
    return df_filled
    
def train(df):
    y = pd.DataFrame(df["Quality_of_life_measure"])
    X = df.drop(["Quality_of_life_measure", "LOCATION"], axis=1)
    x_with_dummies = pd.get_dummies(X)
    x = sm.add_constant(x_with_dummies)
    model = sm.OLS(y, x).fit()
    return model

def predict(model, df):
    df = df.drop("LOCATION", axis=1)
    x_with_dummies = pd.get_dummies(df)
    x = sm.add_constant(x_with_dummies, has_constant='add')
    return model.predict(x)