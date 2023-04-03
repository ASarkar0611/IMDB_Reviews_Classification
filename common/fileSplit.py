# Split the dataframe in training and test df

def trainTestplit(df_fold, foldVal):

    train_df = df_fold[df_fold.kfold != foldVal].reset_index(drop=True)
    test_df = df_fold[df_fold.kfold == foldVal].reset_index(drop=True)

    return train_df, test_df