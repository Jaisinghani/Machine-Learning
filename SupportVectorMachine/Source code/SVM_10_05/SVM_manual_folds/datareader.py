import pandas as pd
def readGlassDataSet(fileName):
    glass_data = pd.read_csv(fileName)
    print(glass_data.shape)
    print(glass_data.head())
    print(glass_data.describe())
    return glass_data