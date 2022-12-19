import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def prep_sentinel(df, columns, columns_subset, dataset_subset = 10000, train_val_split=0.8, scaled=True, shuffle=False):

    if columns_subset:
        columns = columns[:columns_subset]
    dataRaw = df[columns].dropna()

    if dataset_subset:
        dataRaw = dataRaw.iloc[:dataset_subset, :]

    df = dataRaw.copy()
    x = df.values

    if scaled:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dfNorm = pd.DataFrame(x_scaled, columns=df.columns)
    else:
        dfNorm = pd.DataFrame(x, columns=df.columns)

    X_train, X_test, y_train, y_test = train_test_split(dfNorm, dfNorm, train_size=train_val_split,
                                                        shuffle=shuffle)
    df_train = pd.DataFrame(X_train, columns=dfNorm.columns)
    df_test = pd.DataFrame(X_test, columns=dfNorm.columns)

    return df_train, df_test