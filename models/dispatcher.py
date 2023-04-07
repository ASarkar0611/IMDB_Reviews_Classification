from sklearn import linear_model as lm
from sklearn import naive_bayes as nb
from sklearn.svm import SVC

MODELS = {'Logistic Regression': lm.LogisticRegression(),
          'Multinomial NB': nb.MultinomialNB(),
          'SVM': SVC(kernel='linear')}