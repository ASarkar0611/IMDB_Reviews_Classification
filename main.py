# importing files
from scripts import create_folds # Importing file to use function
from common import fileSplit
from scripts import runModel
from common.utils import ClscommonUtils #Importing class

# importing packages
import nltk
#nltk.download('punkt')
import warnings

def load_data(path):
    return ClscommonUtils.load_data(path)

def create_kfolds(df):
    return create_folds.create_kfold(df)

def trainTestplit(df_fold, fold_):
    return fileSplit.trainTestplit(df_fold, fold_)

def RunModel(train_df, test_df):
    return runModel.run_Model(train_df, test_df)

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    df = load_data('data/IMDB_Reviews.csv')
    df = df.sample(1000)
    df_fold, numFolds = create_kfolds(df)
    #train_df, test_df = trainTestplit(df_fold, numFolds)

    for fold_ in range(numFolds):
        train_df, test_df = trainTestplit(df_fold, fold_)
        print(f"Fold: {fold_}")
        RunModel(train_df, test_df)


