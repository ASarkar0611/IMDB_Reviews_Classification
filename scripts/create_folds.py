# create folds on the IMDB dataset

from sklearn import model_selection as ms

def create_kfold(df):

    # Map positive to 1 and negative to 0
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == 'positive' else 0)

    # Create new column kfold and assign value as -1
    df["kfold"] = -1

    # Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df["sentiment"].values

    # Initiate kfold class from model_selection
    numFolds = 5
    kf = ms.StratifiedKFold(n_splits=numFolds)

    # Fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    return df, numFolds

