# count vectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from models import dispatcher
from scripts import predict

MODELS = dispatcher.MODELS

def run_Model(train_df, test_df, modelNum):
    #df_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
    df_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

    # fit
    xtrain = df_vec.fit_transform(train_df['review'])
    # transform
    xtest = df_vec.transform(test_df['review'])

    for key, value in MODELS.items():

        if key == modelNum:
            print(f"Model {value[0]}:")
            predict.predictVal(value[1],xtrain,xtest,train_df,test_df)
