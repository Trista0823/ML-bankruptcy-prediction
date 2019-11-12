import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import os
from sklearn.neighbors import KernelDensity
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
random_state = 4


def outlier_cut(df, features, cut, side='right'):

    """
    It will replace outliers by a specified quantile value. Outliers are defined by quantile value as well.

    :param df: pd.DataFrame, the whole dataframe
    :param features: list, contains the column names of features which need to cut outlier in the dataframe
    :param cut: float, 0 <= cut <= 1, the quantile used to define and replace outlier.
    :param side: string. 'right': replace the right side outliers;
                         'left': replace the left side outliers;
                         'both': replace both right and left side outliers.
    :return: pd.DataFrame, df whose outliers have been replaced by a specified quantile value.
    """

    # iterate over features(columns)
    for col in features:
        Q1 = df[col].quantile(cut)
        Q2 = df[col].quantile(1-cut)
        if side == 'right':
            df.loc[df[col] > Q2, col] = Q2
        elif side == 'left':
            df.loc[df[col] < Q1, col] = Q1
        else:
            df.loc[df[col] < Q1, col] = Q1
            df.loc[df[col] > Q2, col] = Q2
    return df


def feature_importance_analysis(X_train, Y_train, n, random_state):

    """
    Use RandomForestClassifier to conduct a feature importance analysis. Print out n most important features and show
    bar chart with the mean value and the standard deviation of feature importance for n most important features.

    :param X_train: pd.DataFrame, contains features in train dataset
    :param Y_train: pd.DataFrame, contains targets in test dataset
    :param n: int, number of features to be printed and showed in the graph
    :param random_state: int, set the random_state for the model used in feature importance analysis
    :return: None.
    """

    # Feature importance analysis with random forest
    forest = RandomForestClassifier(random_state=random_state).fit(X_train, Y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n):
        print("%d. %s (%f)" % (f + 1, list(X_train.columns.values)[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    _ = plt.figure(figsize=(18, 8))
    _ = plt.title("Feature importances")
    _ = plt.bar(range(n), importances[indices][:n],
                color="r", yerr=std[indices][:n], align="center")

    # plt.xticks(range(X_train.shape[1]), indices)
    _ = plt.xticks(range(n), [list(X_train.columns.values)[i] for i in indices[:n]])
    _ = plt.xlim([-1, n])
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scorer='accuracy', n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    """
    Generate a simple plot of the training and validation learning curve.

    :param estimator: object type that implements the "fit" and "predict" methods.
                      An object of that type which is cloned for each validation.
    :param title: string. Title for the chart.
    :param X: array-like, shape (n_samples, n_features).Training vector, where n_samples is the number of samples and
              n_features is the number of features.
    :param y: array-like, shape (n_samples) or (n_samples, n_features), optional.
              Target relative to X for classification or regression;
              None for unsupervised learning.
    :param ylim: tuple, shape (ymin, ymax), optional. Defines minimum and maximum yvalues plotted.
    :param cv: int, cross-validation generator or an iterable, optional.
               Determines the cross-validation splitting strategy.
    :param scorer: string or callable object that returns a scalar score.
    :param n_jobs: int or None, optional (default=None). Number of jobs to run in parallel.
    :param train_sizes: array-like, shape (n_ticks,), dtype float or int.
                        Relative or absolute numbers of training examples that will be used
                        to generate the learning curve.
    :return: plt, a simple plot of the test and training learning curve
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def binary_kde_plot(train_data, col_name, target):

    """
    Generate a simple plot of kernal density functions of one feature in a binary classification task.
    Two kernal density functions for two target class will be drawn in one chart.

    :param train_data: pd.DataFrame, contains features and targets
    :param col_name: string, column name of a feature whose kernal density function would be drawn
    :param target: string, column name of target in train_data
    :return:None
    """
    train = train_data[train_data[target] == 1][col_name].values.reshape(-1, 1)
    train_not = train_data[train_data[target] == 0][col_name].values.reshape(-1, 1)

    x_plot = np.linspace(train_data[col_name].quantile(0.01), train_data[col_name].quantile(0.99), 5000)[:, np.newaxis]

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train)
    log_dens = kde.score_samples(x_plot)
    plt.plot(x_plot[:, 0], np.exp(log_dens), 'g-', label=target.lower())

    kde_not = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_not)
    log_dens_not = kde_not.score_samples(x_plot)
    plt.plot(x_plot[:, 0], np.exp(log_dens_not), 'r-', label='not'+target.lower())

    plt.legend()
    plt.title("Distribution of {n:s} vs Bankrupt".format(n=col_name))
    plt.show()


def linear_regression_imputation(train_data, test_data, missing_columns, target):

    """
    Use linear regression model to impute missing values in several columns, and other intact columns will be used as
    features. Target in original data won't be used as a feature in linear regression imputation. The linear regression
    model learned from train dataset will also be applied to test dataset.

    :param train_data: pd.DataFrame, train data contains features with some missing values and target
    :param test_data: pd.DataFrame, test data contains features with some missing values and target
    :param missing_columns: list, names of columns with missing values
    :param target: list of string, name of target
    :return: pd.DataFrame, pd.DataFrame; train and test data whose missing value have been imputed with
             linear regression model
    """

    imputed_train_data = pd.DataFrame(columns=[name for name in missing_columns])
    imputed_test_data = pd.DataFrame(columns=[name for name in missing_columns])

    for feature in missing_columns:
        imputed_train_data[feature] = train_data[feature]
        imputed_test_data[feature] = test_data[feature]
        parameters = list(set(train_data.columns) - set(missing_columns) - set(target))

        # Create a Linear Regression model to estimate the missing data
        model = LinearRegression()
        model.fit(X=train_data.dropna()[parameters], y=train_data.dropna()[feature])

        # observe that I preserve the index of the missing data from the original dataframe
        imputed_train_data.loc[train_data[feature].isnull(), feature] = model.predict(train_data[parameters])[
            train_data[feature].isnull()]
        imputed_test_data.loc[test_data[feature].isnull(), feature] = model.predict(test_data[parameters])[
            test_data[feature].isnull()]

        # replace train and test dataset with imputed dataset
        train_data[feature] = imputed_train_data[feature]
        test_data[feature] = imputed_test_data[feature]

    return train_data, test_data


def simple_modeling(X_train, Y_train, kfold, random_state, scorer='fbeta', beta=np.sqrt(5)):

    """
    Compare 10 popular classifiers and evaluate the mean score of each of them by cross validation procedure.
    Classifiers: SVC, DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier,
                 GradientBoostingClassifier, MLPClassifier, KNeighborsClassifier, LogisticRegression,
                 LinearDiscriminantAnalysis

    :param X_train: pd.DataFrame, features in train dataset
    :param Y_train: array-like, target in train dataset
    :param kfold: int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy
    :param random_state: int, RandomState instance or None
    :param scorer: string. 'fbeta': use fbeta scorer for evaluation in cross validation and test
                           'precision': use precision scorer for evaluation in cross validation and test
                           'accuracy': use accuracy scorer for evaluation in cross validation and test
                           'recall': use recall scorer for evaluation in cross validation and test
    :param beta: float, beta parameter used in fbeta scorer.
    :return: pd.DataFrame, cross validation scores, standard deviation and test scores for 10 classifier
             in a decreasing order of test score
    """

    # Modeling step Test different algorithms
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                          learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    cv_results = {'cross_val_score': [], 'test_score': []}

    if scorer == 'precision':
        for classifier in classifiers:
            cv_results['cross_val_score'].append(
                cross_val_score(classifier, X_train, y=Y_train, scoring='precision', cv=kfold, n_jobs=4))

    elif scorer == 'accuracy':
        for classifier in classifiers:
            cv_results['cross_val_score'].append(
                cross_val_score(classifier, X_train, y=Y_train, scoring='accuracy', cv=kfold, n_jobs=4))

    elif scorer == 'recall':
        for classifier in classifiers:
            cv_results['cross_val_score'].append(
                cross_val_score(classifier, X_train, y=Y_train, scoring='recall', cv=kfold, n_jobs=4))
            
    else:
        f_scorer = make_scorer(fbeta_score, beta=beta, average='micro')
        for classifier in classifiers:
            cv_results['cross_val_score'].append(
                cross_val_score(classifier, X_train, y=Y_train, scoring=f_scorer, cv=kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results['cross_val_score']:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,
                           "Algorithm": ["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees",
                                         "GradientBoosting", "MultipleLayerPerceptron", "KNeighboors",
                                         "LogisticRegression", "LinearDiscriminantAnalysis"]})

    cv_res = cv_res.set_index(['Algorithm'])
    return cv_res.sort_values(by='CrossValMeans', ascending=False)


def gridSearch(models, X_train, Y_train, scorer, kfold, beta=np.sqrt(5)):

    """
    Conduct grid search of hyper parameter tuning for a series of models
    :param models: dictionary. key: estimator object;  values: dict or list of dictionaries
                    Dictionary with parameters names (string) as keys and lists of parameter settings to try as values,
                    or a list of such dictionaries, in which case the grids spanned by each dictionary
                    in the list are explored.
    :param X_train: pd.DataFrame, features in train dataset
    :param Y_train: pd.DataFrame, target in train dataset
    :param scorer: string. 'fbeta': use fbeta scorer for evaluation in GridSearchCV
                           'precision': use precision scorer for evaluation in GridSearchCV
                           'accuracy': use accuracy scorer for evaluation in GridSearchCV
                           'recall': use recall scorer for evaluation in GridSearchCV
    :param kfold: int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy
    :param beta: float, beta parameter used in fbeta scorer.
    :return: list,list. Contains best estimators for each type of estimators and their best score
    """

    best_model, best_score = [], []
    f_scorer = make_scorer(fbeta_score, beta=beta, average='micro')

    for model, parameters in models.items():
        if scorer != 'fbeta':
            gs = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scorer, n_jobs=4, verbose=1)
        else:
            gs = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=f_scorer, n_jobs=4, verbose=1)

        gs.fit(X_train, Y_train)
        best_model.append(gs.best_estimator_)
        best_score.append(gs.best_score_)

    return best_model, best_score


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    """
    Print and plot the confusion matrix.
    :param cm: array, shape = [n_classes, n_classes]. confusion matrix
    :param classes: list, contains the classes of target
    :param title: string, chart title
    :param cmap: Matplotlib's built-in colormaps, set the color style of the chart
    :return: None
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices corresponding to the observations
    containing more than n outliers according to the Tukey method.

    :param df: pd.DataFrame, the whole dataframe
    :param n: int, the criterion number of outlier features of defining an outlier sample
    :param features: list, contains the column names of features which need to detect outlier in the dataframe
    :return: list, contains the index of outlier samples
    """

    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def myModel(fileName, model):
    """
    take file name of test dataset as argument, perform predictions on each example in the test set
    If the test dataset contains the target column, it will return predictions with correspondence actural target;
    otherwise, it will return prediction with test dataset

    :param fileName: string, the name of a CSV file containing the test set. the file should be in ./data
    :param model: estimator object selected from previous work
    :return: pd.DataFrame, contains predictions
    """
    # Get Data
    DATA_PATH = "./data"
    train_data = pd.read_csv(os.path.join(DATA_PATH, "bankruptcy_data.csv"), index_col=0)
    test_data = pd.read_csv(os.path.join(DATA_PATH, fileName), index_col=0)

    # Missing Value Imputation
    train_data[train_data == '?'] = np.nan
    test_data[test_data == '?'] = np.nan

    train_data = train_data.drop(index=[4884, 1783, 5880])

    feature_set = set(test_data.columns.values[test_data.isnull().any()]) \
                      .union(set(train_data.columns.values[train_data.isnull().any()])) - set(['X37'])
    for i in feature_set:
        train_data[i] = train_data[i].fillna(train_data[i].median())
        test_data[i] = test_data[i].fillna(train_data[i].median())

    train_data, test_data = linear_regression_imputation(train_data, test_data, ['X37'], ['Bankrupt'])

    train_data = train_data.astype('float64')
    test_data = test_data.astype('float64')

    # Create New features
    datasets = [train_data, test_data]
    for dataset in datasets:
        dataset['X65'] = dataset['X55'] / dataset['X3']
        dataset['X66'] = dataset['X1'] * dataset['X65']
        dataset['X67'] = dataset['X2'] * dataset['X65']
        dataset['X68'] = dataset['X7'] * dataset['X65']
        dataset['X69'] = dataset['X8'] * dataset['X67']
        dataset['X70'] = dataset['X9'] * dataset['X65']
        dataset['X71'] = dataset['X10'] * dataset['X65']
        dataset['X72'] = dataset['X18'] * dataset['X65']
        dataset['X73'] = dataset['X13'] * dataset['X70'] - dataset['X72']
        dataset['X74'] = dataset['X14'] * dataset['X70'] - dataset['X72']
        dataset['X75'] = dataset['X22'] * dataset['X65']
        dataset['X76'] = dataset['X55'] / dataset['X28']
        dataset['X77'] = dataset['X34'] * dataset['X67']
        dataset['X78'] = dataset['X77'] / dataset['X33']
        dataset['X79'] = dataset['X35'] * dataset['X65']
        dataset['X80'] = dataset['X20'] * dataset['X70'] / 365
        dataset['X81'] = dataset['X48'] * dataset['X65']
        dataset['X82'] = dataset['X70'] / dataset['X61']
        dataset['X83'] = dataset['X54'] * dataset['X76']
        dataset['X84'] = dataset['X59'] * dataset['X71']
        dataset['X85'] = dataset['X46'] * dataset['X78'] + dataset['X80']
        dataset['X86'] = dataset['X70'] - dataset['X56'] * dataset['X70']

        dataset['X87'] = dataset['X67'] / dataset['X71']
        dataset['X88'] = (dataset['X85'] - dataset['X80']) / dataset['X71']
        dataset['X89'] = dataset['X66'] / dataset['X71']
        dataset['X90'] = dataset['X65'] / dataset['X70']

    feature_set = list(test_data.columns.values[test_data.isnull().any()])
    for i in feature_set:
        test_data[i] = test_data[i].fillna(train_data[i].median())

    # Modeling
    features = set(train_data.columns.values) - set(['Bankrupt'])
    X_test = test_data[features]
    Y_test_predict = model.predict(X_test)

    try:
        Y_test = test_data['Bankrupt']
        result = pd.DataFrame({"predict": Y_test_predict, "actual": Y_test})
        return result
    except:
        test_data['Predict'] = Y_test_predict
        return test_data


if __name__ == '__main__':
    print('done')