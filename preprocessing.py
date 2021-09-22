import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)

# reading in cleaned, complete dataset before one-hot encoding for visualizing
data = pd.read_csv('agaricuslepiota.data', names=["Class", "capShape",
                                                  "capSurface", "capColor",
                                                  "bruises?", "odor",
                                                  "gillAttachment", "gillSpacing",
                                                  "gillSize", "gillColor",
                                                  "stalkShape", "stalkRoot",
                                                  "stalkSurfaceAboveRing", "stalkSurfaceBelowRing",
                                                  "stalkColorAboveRing", "stalkColorBelowRing",
                                                  "veilType", "veilColor", "ringNumber,ringType",
                                                  "sporePrintColor", "population", "habitat"], index_col=False)

# creating One hot encoding for all columns
onehotdata = pd.get_dummies(data)

# splitting dataset function
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

# using the split function for our data
train, validate, test = train_validate_test_split(onehotdata)
#print(train.head())
#print(validate.head())
#print(test.head())
#print(len(train.columns))

# storing the 3 data into 3 files
np.savetxt(r'training.txt', train.values, fmt='%d')
np.savetxt(r'val.txt', validate.values, fmt='%d')
np.savetxt(r'testing.txt', test.values, fmt='%d')