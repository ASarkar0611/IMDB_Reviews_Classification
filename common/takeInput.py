# Ask user which machine learning model to run

def giveInput():
    print("IMDB review Classification")
    print("Let us know which model to run:")
    print("Enter the numeric value corresponding to the ML models as below:")
    print("1. Logistic Regression\t2. Multinomial Naive Bayes\t3. Support Vector Model(SVM)")
    print("4. Random Forest\n")
    inpOpt = int(input('Enter the option:'))

    return inpOpt