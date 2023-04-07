from sklearn import linear_model as lm
from sklearn import naive_bayes as nb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

MODELS = {1: ['Logistic Regression', lm.LogisticRegression()],
          2: ['Multinomial NB', nb.MultinomialNB()],
          3: ['SVM', SVC(kernel='linear')],
          4: ['Random Forest', RandomForestClassifier(max_depth=3, random_state=0)]}