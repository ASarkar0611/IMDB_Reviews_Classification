from sklearn import linear_model as lm
from sklearn import naive_bayes as nb

MODELS = {'Logistic Regression': lm.LogisticRegression(),
          'Multinomial NB': nb.MultinomialNB()}