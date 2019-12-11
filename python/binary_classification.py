# Import libraries

# Mainstays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Models
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, precision_recall_curve


class BinaryClassifier:
    '''Abstract class to process data, perform binary classification, and run general model diagnostics'''

    def __init__(self, train_data,
                       valid_data,
                       test_data,
                       predictors,
                       outcome,
                       threshold = 0.5,
                       random_state = 0):
        '''
        @param train_data       DataFrame with training set data
        @param valid_data       DataFrame with validation set data
        @param test_data        Dataframe with test set data
        @param predictors       List of strings of column names of predictors
        @param outcome          String with outcome column name
        @param threshold        Optional: Float, positive prediction cutoff
        @param random_state     Optional: int
        '''

        # Store inputs
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.predictors = predictors
        self.outcome = outcome
        self.threshold = threshold
        self.random_state = random_state

    def train(self, keep = True):
        '''
        Dummy method for model training. Must be defined in child class.

        @param keep, Optional: Bool, if true save model, else return model
        '''
        raise NotImplementedError("Must override train")

    def set_threshold(self, threshold):
        '''
        Method to set positive prediction cutoff

        @param threshold, Float
        '''
        self.threshold = threshold

    def set_model(self, model):
        '''
        Method to set model

        @param model
        '''
        self.model = model

    def compute_prob(self, prob_set = "Test",
                           predict = True,
                           keep = True):
        '''
        Method to compute probability scores and make predictions

        @param prob_set, Optional: either "Valid" or "Test"
        @param predict,  Optional: Bool, if true store predictions made at self.threshold
        @param keep,     Optional: Bool, if true, store probabilities and predicitons, else return
        '''

        # Predict probabilities
        if prob_set == "Test":
            prob = self.model.predict_proba(self.test_data[self.predictors])
        else:
            prob = self.model.predict_proba(self.valid_data[self.predictors])

        # Convert to dataframe
        prob = pd.DataFrame(prob)
        prob = prob.rename(columns = {0: 'prob_0', 1: 'prob_1'})

        # Create predictions
        if predict:
            pred = [1 if (x >= self.threshold) else 0 for x in prob['prob_1']]

        # Store results or return them
        if keep:
            if prob_set == "Test":
                self.prob_test = prob
                self.pred_test = pred
            else:
                self.prob_valid = prob
                self.pred_valid = pred
        else:
            return prob, pred

    def performance_metric(self, prob_set = "Test",
                                 measure = "Accuracy"):
        '''
        Method to compute model performance metrics

        @param measure, Optional: String with metric to compute
                        Options: ["AUC", "Accuracy", "F1", "Precision", "Recall"]
        '''

        # Extract precomputed predictions
        if prob_set == "Test":
            prob = self.prob_test
            pred = self.pred_test
            y = list(self.test_data[self.outcome])
        else:
            prob = self.prob_valid
            pred = self.pred_valid
            y = list(self.valid_data[self.outcome])

        # Compute performance metric
        if measure == "AUC":
            fpr, tpr, threshold = roc_curve(y, list(prob['prob_1']))
            return auc(fpr, tpr)
        elif measure == "Accuracy":
            return accuracy_score(y, pred)
        elif measure == "F1":
            return f1_score(y, pred)
        elif measure == "Precision":
            return precision_score(y, pred)
        elif measure == "Recall":
            return recall_score(y, pred)
        else:
            raise ValueError("Unrecognized model performance metric {}".format(measure))

    def test(self, prob_set = "Test",
                   p = True):
        '''
        Method to test model performance and return diagnostics:
        - Precision (AKA: PPV)
            = (true positive) / (true positive + false positive)
        - Accuracy
        - Recall (AKA: Senstivity)
            = (true positive) / (true positive + false negative)
        - F1
            = harmonic mean of precision and recall

        Generate plots for:
            - ROC
            - Precision-Recall curce

        @param prob_set, Optional: either "Valid" or "Test"
        @param p,        Optional: Bool, if true, print diagnostics
        '''

        # Compute probabilities and predictions
        self.compute_prob(prob_set = prob_set, predict = True)

        # Initialize plot
        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10,10))

        # Extract precomputed predictions
        if prob_set == "Test":
            prob = list(self.prob_test['prob_1'])
            y = list(self.test_data[self.outcome])
        else:
            prob = list(self.prob_valid['prob_1'])
            y = list(self.valid_data[self.outcome])

        # ROC Curve
        fpr, tpr, t = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        axs[0].set_title("{} Reciever Operating Characteristic".format(prob_set))
        axs[0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        axs[0].legend(loc = 'lower right')
        axs[0].plot([0,1], [0,1], 'r--')
        axs[0].set_xlim([0,1])
        axs[0].set_ylim([0,1])
        axs[0].set_ylabel("True Positive Rate (TPR)")
        axs[0].set_xlabel("False Positive Rate (FPR)")

        # Precision-Recall Curve
        precision, recall, t = precision_recall_curve(y, prob)
        pr_auc = auc(recall, precision)
        axs[1].set_title("{} Precision Recall Curve".format(prob_set))
        axs[1].plot(recall, precision, 'b', label = 'AUC = %0.2f' % pr_auc)
        axs[1].legend(loc = 'lower right')
        axs[1].set_xlim([0,1])
        axs[1].set_ylim([0,1])
        axs[1].set_xlabel("Recall (Sensitivity)")
        axs[1].set_ylabel("Precision (PPV)")

        # Compute diagnostics
        self.accuracy = self.performance_metric(measure = "Accuracy", prob_set = prob_set) * 100.0
        self.f1 = self.performance_metric(measure = "F1", prob_set = prob_set) * 100.0
        self.precision = self.performance_metric(measure = "Precision", prob_set = prob_set) * 100.0
        self.recall = self.performance_metric(measure = "Recall", prob_set = prob_set) * 100.0

        if p:
            print("Accuracy:             %.2f%%" % self.accuracy)
            print("F1:                   %.2f%%" % self.f1)
            print("Precision (PPV):      %.2f%%" % self.precision)
            print("Recall (Sensitivity): %.2f%%" % self.recall)
            print("\n")
            plt.show()

        else:
            return plt


class logreg(BinaryClassifier):
    '''
    Child of class BinaryClassifier
    Creates logistic regression binary classification model
    '''

    def train(self, penalty = 'l2',
                    cv = 2,
                    solver = 'liblinear',
                    max_iter = 100,
                    keep = True):
        '''
        Train logistic regression binary classifier

        @param penalty,  Optional: String, type of regularization
                         Default: 'l2'
        @param cv,       Optional, int, number of cross validation folds
                         Default: 2
        @param solver,   Optional: String, algorithm used for optimization
                         Default: 'saga', faster for large datasets
        @param max_iter, Optional: int, maximum iterations for solver to converge
        @param keep,     Optional: Bool, if true save model, else return model
        '''

        # Start timer
        start_time = time.time()

        # Initialize non-elasticnet model
        if penalty != 'elasticnet':
            model = LogisticRegressionCV(penalty = penalty,
                                         random_state = self.random_state,
                                         solver = solver,
                                         max_iter = max_iter,
                                         cv = cv)

        # Train model
        model.fit(self.train_data[self.predictors],
                  self.train_data[self.outcome])

        # Print training time
        end_time = time.time()
        print("Time to train: {:.2f}".format(end_time - start_time))

        # Keep or return model
        if keep:
            self.model = model
        else:
            return model


class DecisionTree(BinaryClassifier):
    '''
    Child of class BinaryClassifier
    Creates Decision Tree binary classification model
    '''

    def train(self, criterion = 'gini',
                    max_depth = 4,
                    keep = True):
        '''
        Train DecisionTreeClassifier

        @param criterion,    Optional: str, impurity measure (options: ['gini', 'entropy'])
        @param max_depth,    Optional: int, max depth of tree
        @param keep,         Optional: bool, if true, set model, if false, return model
        '''

        # Start timer
        start = time.time()

        # Initialize model
        model = DecisionTreeClassifier(random_state = self.random_state,
                                       criterion = criterion,
                                       max_depth = max_depth)

        # Train model
        model.fit(self.train_data[self.predictors],
                  self.train_data[self.outcome])

        # Print training time
        end = time.time()
        print("Time to train: {:.2f}".format(end - start))


        # Keep or return model
        if keep:
            self.model = model
        else:
            return model


class RandomForest(BinaryClassifier):
    '''
    Child of class BinaryClassifier
    Creates Random Forest binary classification model
    '''

    def train(self, n_estimators = 10,
                    criterion = 'gini',
                    max_depth = 4,
                    keep = True):
        '''
        Train RandomForestClassifier

        @param n_estimators, Optional: int, max number of trees in forest
        @param criterion,    Optional: str, impurity measure (options: ['gini', 'entropy'])
        @param max_depth,    Optional: int, max depth of each tree
        @param keep,         Optional: bool, if true, set model, if false, return model
        '''

        # Start timer
        start = time.time()

        # Initialize model
        model = RandomForestClassifier(n_estimators=n_estimators,
                                        criterion = criterion,
                                        max_depth = max_depth,
                                        random_state = self.random_state)

        # Train model
        model.fit(self.train_data[self.predictors],
                  self.train_data[self.outcome])

        # Print training time
        end = time.time()
        print("Time to train: {:.2f}".format(end - start))

        # Keep or return model
        if keep:
            self.model = model
        else:
            return model


class AdaBoost(BinaryClassifier):
    '''
    Child of class BinaryClassifier
    Creates AdaBoost binary classification model
    '''

    def train(self, n_estimators = 50,
                    base_depth = 1,
                    learning_rate = 1.0,
                    keep = True):
        '''
        Train AdaBoostClassifier

        @param n_estimators,  Optional: int, max number of estimators where boosting is stopped
        @param base_depth,    Optional: int, depth of DecisionTreeClassifier base classifier
        @param learning_rate, Optional: float, shrinks the contribution of each classifier
        @param keep,         Optional: bool, if true, set model, if false, return model
        '''

        # Start timer
        start = time.time()

        # Initialize model
        model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = base_depth),
                                    n_estimators = n_estimators,
                                    learning_rate = learning_rate,
                                    random_state = self.random_state)

        # Train model
        model.fit(self.train_data[self.predictors],
                  self.train_data[self.outcome])

        # Print training time
        end = time.time()
        print("Time to train: {:.2f}".format(end - start))

        # Keep or return model
        if keep:
            self.model = model
        else:
            return model

class GDA(BinaryClassifier):
    '''
    Child of class BinaryClassifier
    Creates GDA (Gaussian discriminant analysis) binary classification model
    '''

    def train(self, keep = True):
        '''
        Train GDA

        @param keep,         Optional: bool, if true, set model, if false, return model
        '''

        # Start timer
        start = time.time()

        # Initialize model
        model = LinearDiscriminantAnalysis(priors=None) # Might want to change priors?

        # Train model
        model.fit(self.train_data[self.predictors],
                  self.train_data[self.outcome])

        # Print training time
        end = time.time()
        print("Time to train: {:.2f}".format(end - start))

        # Keep or return model
        if keep:
            self.model = model
        else:
            return model
