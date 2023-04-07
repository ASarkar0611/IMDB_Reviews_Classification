from sklearn import metrics

def predictVal(model,xtrain,xtest,train_df,test_df):
    model.fit(xtrain, train_df['sentiment'])
    preds = model.predict(xtest)

    # Evaluate the model
    acc = metrics.accuracy_score(test_df['sentiment'], preds)
    cm = metrics.confusion_matrix(test_df['sentiment'], preds)
    cr = metrics.classification_report(test_df['sentiment'], preds)

    print(f"Accuracy - {acc}")
    print(f"Confusion Matrix -\n {cm}")
    print(f"Classification Report -\n {cr}")
