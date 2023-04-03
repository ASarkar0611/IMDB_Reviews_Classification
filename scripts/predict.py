from sklearn import metrics

def predictVal(model,xtrain,xtest,train_df,test_df):
    model.fit(xtrain, train_df['sentiment'])
    preds = model.predict(xtest)

    acc = metrics.accuracy_score(test_df['sentiment'], preds)
    return acc