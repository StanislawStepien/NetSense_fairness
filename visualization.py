import os
import pandas as pd
import pydotplus
from six import StringIO
from IPython.display import Image
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz, plot_tree


def visualize_tree(model: DecisionTreeClassifier or ExtraTreeClassifier, features: list[str], save_directory: str,
                   img_filename: str):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=features, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(save_directory + img_filename)
    Image(graph.create_png())


def visualize_random_forest(model: RandomForestClassifier, features: list[str], save_directory: str, img_filename: str):
    fn = features
    cn = model.classes_
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
    for index in range(len(model.estimators_)):
        plot_tree(model.estimators_[index],
                  feature_names=fn,
                  class_names=cn,
                  filled=True,
                  ax=axes[index])
        axes[index].set_title('Estimator: ' + str(index), fontsize=11)
    fig.savefig(save_directory + img_filename)


def visualize_logistic_regression(model, features: list[str], save_directory: str, filename: str,
                                  x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    model_fi = permutation_importance(model, x_train, y_train, random_state=1)
    out = f"Importance of features for {str(model.__class__)}:\n"
    for index, feature in enumerate(features):
        out += f"{feature}: {model_fi['importances_mean'][index]}\n"
    if os.path.exists(save_directory + filename):
        with open(save_directory + filename, 'w') as file:
            file.write(out)
    else:
        with open(save_directory + filename, 'x') as file:
            file.write(out)


def visualize_model(
        model: DecisionTreeClassifier or ExtraTreeClassifier or RandomForestClassifier or LogisticRegression or GaussianNB or KNeighborsClassifier,
        save_directory: str, img_filename: str, features: list[str], x_train: pd.DataFrame,
        y_train: pd.DataFrame) -> None:
    if model.__class__ in [DecisionTreeClassifier, ExtraTreeClassifier]:
        visualize_tree(model, features, save_directory, img_filename)
    elif model.__class__ == RandomForestClassifier:
        visualize_random_forest(model, features, save_directory, img_filename)
    elif model.__class__ == [LogisticRegression, GaussianNB]:
        visualize_logistic_regression(model, features, save_directory, img_filename, x_train, y_train)
    elif model.__class__ == GaussianNB:
        print(model)
