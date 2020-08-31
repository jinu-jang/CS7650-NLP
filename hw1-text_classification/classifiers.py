import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn
from math import log
import pandas as pd

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set

        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model

        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions

        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!

        # Hard coding the assumption that there are only 2 classes
        self.logprior = [0, 0]


    def fit(self, X, Y):
        # Add your code here!
        df_X = pd.DataFrame(X)
        df_Y = pd.DataFrame(Y)

        self.classes = [0, 1]

        num_doc = X.shape[0]
        self.V = X.shape[1]

        self.loglikelihood = np.zeros((len(self.classes), self.V))

        num_pos = sum(Y)
        num_class = [len(Y) - num_pos, num_pos]

        for c in self.classes:
            self.logprior[c] = log(num_class[c] / num_doc)

            count_w_given_c = df_X[(df_Y == c).label].sum() + 1
            total_count_smooth = count_w_given_c.sum()

            for w in range(self.V):
                self.loglikelihood[c, w] = log(count_w_given_c[w] / total_count_smooth)



    def predict(self, X):
        # Add your code here!
        answer = []

        for d in range(X.shape[0]):
            sums = (self.loglikelihood * X[d]).sum(axis=1) + self.logprior
            if sums[0] > sums[1]:
                answer.append(0)
            else:
                answer.append(1)
        return answer


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")


    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")


    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


class PerceptronClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")

    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")

# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayes):
class BonusClassifier(PerceptronClassifier):
    def __init__(self):
        super().__init__()
