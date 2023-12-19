import string
import pandas as pd
from num2words import num2words
import pydotplus
from six import StringIO
from IPython.display import Image
import sklearn.tree as tree
import random
from PIL import Image
import numpy as np

from word2number import w2n
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)


import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def add_labels_to_subplots(axs, hfont, height, fontsize):
    labels_subplots = list(string.ascii_uppercase)
    for i, ax in enumerate(axs):
        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[1] * height,
            labels_subplots[i],
            fontsize=fontsize,
            **hfont,
        )
    return labels_subplots


def merge_and_rename_data(data1, data2, on_column, suffix1, suffix2):
    merged_data = pd.merge(
        data1, data2, on=on_column, suffixes=("_" + suffix1, "_" + suffix2)
    )

    new_column_names = [
        str(col)
        .replace(f"_{on_column}_{suffix1}", f"_{suffix1}")
        .replace(f"_{on_column}_{suffix2}", f"_{suffix2}")
        for col in merged_data.columns
    ]
    merged_data.rename(
        columns=dict(zip(merged_data.columns, new_column_names)), inplace=True
    )

    return merged_data


def pivot_data_by_HSA(data, index_column, columns_column, values_column):
    data_by_HSA = data[[index_column, columns_column, values_column]]
    pivot_table = data_by_HSA.pivot_table(
        index=index_column, columns=columns_column, values=values_column
    )
    return pivot_table


def create_column_names(categories_for_subsetting, num_of_weeks):
    column_names = ["HSA_ID"]

    for week in range(1, num_of_weeks + 1):
        week = num2words(week)
        for category in categories_for_subsetting:
            column_name = f"week_{week}_{category}"
            column_names.append(column_name)

    return column_names


def create_collated_weekly_data(
    pivoted_table, original_data, categories_for_subsetting, geography, column_names
):
    collated_data = pd.DataFrame(index=range(51), columns=column_names)

    x = 0
    for geo in original_data[geography].unique():
        # matching_indices = [i for i, geo_col in enumerate(pivoted_table) if geo_col == geo]
        collated_data.loc[x, geography] = geo
        columns_to_subset = [
            f"{geo}_{category}" for category in categories_for_subsetting
        ]
        j = 1
        try:
            for row in range(len(pivoted_table.loc[:, columns_to_subset])):
                collated_data.iloc[
                    x, j : j + len(categories_for_subsetting)
                ] = pivoted_table.loc[row, columns_to_subset]
                j += len(categories_for_subsetting)
        except:
            pass
        x += 1

    return collated_data


def add_changes_by_week(weekly_data_frame, outcome_column):
    for column in weekly_data_frame.columns[1:]:
        # Calculate the difference between each row and the previous row
        if outcome_column not in column.lower():  # want to leave out the outcome column
            diff = weekly_data_frame[column].diff()

            # Create a new column with the original column name and "delta"
            new_column_name = column + "_delta"

            column_index = weekly_data_frame.columns.get_loc(column)

            # Insert the new column just after the original column
            weekly_data_frame.insert(column_index + 1, new_column_name, diff)
            weekly_data_frame[new_column_name] = diff
    return weekly_data_frame


def prep_training_test_data_period(
    data, no_weeks, weeks_in_future, geography, weight_col, keep_output
):
    ## Get the weeks for the x and y datasets
    x_weeks = []
    y_weeks = []
    y_weeks_to_check = []  # check these weeks to see if any of them are equal to 1
    for week in no_weeks:
        test_week = int(week) + weeks_in_future
        x_weeks.append("_" + num2words(week) + "_")
        for week_y in range(week + 1, test_week + 1):
            y_weeks_to_check.append("_" + num2words(week_y) + "_")
        y_weeks.append("_" + num2words(test_week) + "_")

    ## Divide up the test/train split
    # if is_geographic:
    # Calculate the index to start slicing from
    #    start_index = len(data['county']) // proportion[0] * proportion[1]
    # Divide up the dataset based on this proportion
    #    first_two_thirds = data['county'][:start_index]
    #    last_third = data['county'][start_index:]
    X_data = pd.DataFrame()
    y_data = pd.DataFrame()
    weights_all = pd.DataFrame()
    missing_data = []
    ## Now get the training data
    k = 0
    for x_week in x_weeks:
        y_week = y_weeks[k]
        k += 1

        weeks_x = [col for col in data.columns if x_week in col]
        columns_x = [geography] + weeks_x + [weight_col]
        data_x = data[columns_x]

        weeks_y = [col for col in data.columns if y_week in col]
        columns_y = [geography] + weeks_y
        data_y = data[columns_y]
        ### now add the final column to the y data that has it so that it's if any week in the trhee week perdiod exceeded 15
        train_week = w2n.word_to_num(x_week.replace("_", ""))
        target_week = w2n.word_to_num(y_week.replace("_", ""))
        y_weeks_to_check = []
        for week_to_check in range(train_week + 1, target_week + 1):
            y_weeks_to_check.append("_" + num2words(week_to_check) + "_")

        y_weeks_to_check = [week + "beds_over_15_100k" for week in y_weeks_to_check]
        columns_to_check = [
            col for col in data.columns if any(week in col for week in y_weeks_to_check)
        ]
        y_over_in_period = data[columns_to_check].apply(max, axis=1)
        data_y = pd.concat([data_y, y_over_in_period], axis=1)
        # ensure they have the same amount of data
        # remove rows in test_data1 with NA in test_data2
        data_x = data_x.dropna()
        data_x = data_x[data_x[geography].isin(data_y[geography])]
        # remove rows in test_data2 with NA in test_data1
        data_y = data_y.dropna()
        data_y = data_y[data_y[geography].isin(data_x[geography])]
        data_x = data_x[data_x[geography].isin(data_y[geography])]
        data_x_no_HSA = len(data_x[geography].unique())

        missing_data.append(
            (
                (len(data[geography].unique()) - data_x_no_HSA)
                / len(data[geography].unique())
            )
            * 100
        )
        # get weights
        # weights = weight_data[weight_data[geography].isin(data_x[geography])][[geography, weight_col]]

        X_week = data_x.iloc[:, 1 : len(columns_x)]  # take away y, leave weights for mo
        y_week = data_y.iloc[:, -1]

        y_week = y_week.astype(int)
        weights = X_week.iloc[:, -1]
        if keep_output:
            X_week = X_week.iloc[
                :, : len(X_week.columns) - 1
            ]  # remove the weights and leave "target" for that week

            # rename columns for concatenation
            X_week.columns = range(1, len(data_x.columns) - 1)
        else:
            X_week = X_week.iloc[
                :, : len(X_week.columns) - 2
            ]  # remove the weights and  "target" for that week

            X_week.columns = range(
                1, len(data_x.columns) - 2
            )  # remove the weights and  "target" for that week

        y_week.columns = range(1, len(data_y.columns) - 2)
        X_data = pd.concat([X_data, X_week])
        y_data = pd.concat([y_data, y_week])

        weights_all = pd.concat([weights_all, weights])

    X_data.reset_index(drop=True, inplace=True)
    y_data.reset_index(drop=True, inplace=True)
    weights_all.reset_index(drop=True, inplace=True)

    return (X_data, y_data, weights_all, missing_data)


### this code it's exactly in the x weeks away
def prep_training_test_data(
    data, no_weeks, weeks_in_future, geography, weight_col, keep_output
):
    ## Get the weeks for the x and y datasets
    x_weeks = []
    y_weeks = []
    for week in no_weeks:
        test_week = int(week) + weeks_in_future
        x_weeks.append("_" + num2words(week) + "_")
        y_weeks.append("_" + num2words(test_week) + "_")

    X_data = pd.DataFrame()
    y_data = pd.DataFrame()
    weights_all = pd.DataFrame()
    missing_data = []
    ## Now get the training data
    k = 0
    for x_week in x_weeks:
        y_week = y_weeks[k]
        k += 1
        weeks_x = [col for col in data.columns if x_week in col]
        columns_x = [geography] + weeks_x + [weight_col]
        data_x = data[columns_x]

        weeks_y = [col for col in data.columns if y_week in col]
        columns_y = [geography] + weeks_y
        data_y = data[columns_y]
        # ensure they have the same amount of data
        # remove rows in test_data1 with NA in test_data2
        data_x = data_x.dropna()
        data_x = data_x[data_x[geography].isin(data_y[geography])]
        # remove rows in test_data2 with NA in test_data1
        data_y = data_y.dropna()
        data_y = data_y[data_y[geography].isin(data_x[geography])]
        data_x = data_x[data_x[geography].isin(data_y[geography])]
        data_x_no_HSA = len(data_x[geography].unique())

        missing_data.append(
            (
                (len(data[geography].unique()) - data_x_no_HSA)
                / len(data[geography].unique())
            )
            * 100
        )
        # get weights
        # weights = weight_data[weight_data[geography].isin(data_x[geography])][[geography, weight_col]]

        X_week = data_x.iloc[:, 1 : len(columns_x)]  # take away y, leave weights for mo
        y_week = data_y.iloc[:, -1]

        y_week = y_week.astype(int)
        weights = X_week.iloc[:, -1]
        if keep_output:
            X_week = X_week.iloc[
                :, : len(X_week.columns) - 1
            ]  # remove the weights and leave "target" for that week

            # rename columns for concatenation
            X_week.columns = range(1, len(data_x.columns) - 1)
        else:
            X_week = X_week.iloc[
                :, : len(X_week.columns) - 2
            ]  # remove the weights and  "target" for that week

            X_week.columns = range(
                1, len(data_x.columns) - 2
            )  # remove the weights and  "target" for that week

            # rename columns for concatenation
        y_week.columns = range(1, len(data_y.columns) - 1)
        X_data = pd.concat([X_data, X_week])
        y_data = pd.concat([y_data, y_week])

        weights_all = pd.concat([weights_all, weights])

    X_data.reset_index(drop=True, inplace=True)
    y_data.reset_index(drop=True, inplace=True)
    weights_all.reset_index(drop=True, inplace=True)

    return (X_data, y_data, weights_all, missing_data)


def calculate_metrics(confusion_matrix):
    # Extract values from the confusion matrix
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]

    # Calculate Sensitivity (True Positive Rate), Specificity (True Negative Rate),
    # PPV (Precision), and NPV
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    return sensitivity, specificity, ppv, npv


def pivot_data_by_HSA(data, index_column, columns_column, values_column):
    data_by_HSA = data[[index_column, columns_column, values_column]]
    pivot_table = data_by_HSA.pivot_table(
        index=index_column, columns=columns_column, values=values_column
    )
    return pivot_table


def add_changes_by_week(weekly_data_frame, outcome_column):
    for column in weekly_data_frame.columns[1:]:
        # Calculate the difference between each row and the previous row
        if outcome_column not in column.lower():  # want to leave out the outcome column
            diff = weekly_data_frame[column].diff()

            # Create a new column with the original column name and "delta"
            new_column_name = column + "_delta"

            column_index = weekly_data_frame.columns.get_loc(column)

            # Insert the new column just after the original column
            weekly_data_frame.insert(column_index + 1, new_column_name, diff)
            weekly_data_frame[new_column_name] = diff
    return weekly_data_frame


def determine_covid_outcome_indicator(
    new_cases_per_100k, new_admits_per_100k, percent_beds_100k
):
    if new_cases_per_100k < 200:
        if (new_admits_per_100k >= 10) | (percent_beds_100k > 0.10):
            if (new_admits_per_100k >= 20) | (percent_beds_100k >= 15):
                return 1
            else:
                return 0
        else:
            return 0
    elif new_cases_per_100k >= 200:
        if (new_admits_per_100k >= 10) | (percent_beds_100k >= 0.10):
            return 1
        elif (new_admits_per_100k < 10) | (percent_beds_100k < 10):
            return 1


def simplify_labels_graphviz(graph):
    for node in graph.get_node_list():
        if node.get_attributes().get("label") is None:
            continue
        else:
            split_label = node.get_attributes().get("label").split("<br/>")
            if len(split_label) == 4:
                split_label[3] = split_label[3].split("=")[1].strip()

                del split_label[1]  # number of samples
                del split_label[1]  # split of sample
            elif len(split_label) == 3:  # for a terminating node, no rule is provided
                split_label[2] = split_label[2].split("=")[1].strip()

                del split_label[0]  # number of samples
                del split_label[0]  # split of samples
                split_label[0] = "<" + split_label[0]
            node.set("label", "<br/>".join(split_label))


def generate_decision_tree_graph(classifier, class_names, feature_names):
    dot_data = StringIO()
    tree.export_graphviz(
        classifier,
        out_file=dot_data,
        class_names=class_names,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=False,
        precision=0,
        impurity=False,
    )

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    return graph


def cross_validation_leave_geo_out(
    data,
    geography_column,
    geo_split,
    no_iterations,
    cv,
    classifier,
    param_grid,
    no_iterations_param,
    no_weeks_train,
    no_weeks_test,
    weeks_in_future,
    weight_col,
    keep_output,
    time_period,
):
    best_hyperparameters_per_iter = []
    auROC_per_iter = []

    for i in range(no_iterations):
        print(i)
        # Subset the HSAs from the full dataset
        geo_names = data[geography_column].unique()
        num_names_to_select = int(geo_split * len(geo_names))
        geos_for_sample = random.sample(list(geo_names), num_names_to_select)
        subset_HSAs_for_train = data[data[geography_column].isin(geos_for_sample)]
        subset_HSAs_for_test = data[~data[geography_column].isin(geos_for_sample)]

        # Create training and test data
        if time_period == "period":
            (
                X_sample_train,
                y_sample_train,
                weights_train,
                missing_data_train_HSA,
            ) = prep_training_test_data_period(
                subset_HSAs_for_train,
                no_weeks=no_weeks_train,
                weeks_in_future=weeks_in_future,
                geography=geography_column,
                weight_col=weight_col,
                keep_output=keep_output,
            )
            (
                X_sample_test,
                y_sample_test,
                weights_test,
                missing_data_train_HSA,
            ) = prep_training_test_data_period(
                subset_HSAs_for_test,
                no_weeks=no_weeks_test,
                weeks_in_future=weeks_in_future,
                geography=geography_column,
                weight_col=weight_col,
                keep_output=keep_output,
            )
            weights_train = weights_train[0]
        elif time_period == "exact":
            (
                X_sample_train,
                y_sample_train,
                weights_train,
                missing_data_train_HSA,
            ) = prep_training_test_data(
                subset_HSAs_for_train,
                no_weeks=no_weeks_train,
                weeks_in_future=weeks_in_future,
                geography=geography_column,
                weight_col=weight_col,
                keep_output=keep_output,
            )
            (
                X_sample_test,
                y_sample_test,
                weights_test,
                missing_data_train_HSA,
            ) = prep_training_test_data(
                subset_HSAs_for_test,
                no_weeks=no_weeks_test,
                weeks_in_future=weeks_in_future,
                geography=geography_column,
                weight_col=weight_col,
                keep_output=keep_output,
            )
            weights_train = weights_train[0]
        elif time_period == "shifted":
            (
                X_sample_train,
                y_sample_train,
                weights_train,
                missing_data_train_HSA,
            ) = prep_training_test_data_shifted(
                subset_HSAs_for_train,
                no_weeks=no_weeks_train,
                weeks_in_future=weeks_in_future,
                geography=geography_column,
                weight_col=weight_col,
                keep_output=keep_output,
            )
            (
                X_sample_test,
                y_sample_test,
                weights_test,
                missing_data_train_HSA,
            ) = prep_training_test_data_shifted(
                subset_HSAs_for_test,
                no_weeks=no_weeks_test,
                weeks_in_future=weeks_in_future,
                geography=geography_column,
                weight_col=weight_col,
                keep_output=keep_output,
            )
            weights_train = weights_train[0]

        # Check if y_sample_test contains only 1's
        while (int(y_sample_test.sum().iloc[0]) / len(y_sample_test)) == 1:
            print("All 1")
            # Subset the HSAs from the full dataset
            geo_names = data[geography_column].unique()
            num_names_to_select = int(geo_split * len(geo_names))
            geos_for_sample = random.sample(list(geo_names), num_names_to_select)
            subset_HSAs_for_train = data[data[geography_column].isin(geos_for_sample)]
            subset_HSAs_for_test = data[~data[geography_column].isin(geos_for_sample)]

            # Create training and test data
            if time_period == "period":
                (
                    X_sample_train,
                    y_sample_train,
                    weights_train,
                    missing_data_train_HSA,
                ) = prep_training_test_data_period(
                    subset_HSAs_for_train,
                    no_weeks=no_weeks_train,
                    weeks_in_future=weeks_in_future,
                    geography=geography_column,
                    weight_col=weight_col,
                    keep_output=keep_output,
                )
                (
                    X_sample_test,
                    y_sample_test,
                    weights_test,
                    missing_data_train_HSA,
                ) = prep_training_test_data_period(
                    subset_HSAs_for_test,
                    no_weeks=no_weeks_test,
                    weeks_in_future=weeks_in_future,
                    geography=geography_column,
                    weight_col=weight_col,
                    keep_output=keep_output,
                )
                weights_train = weights_train[0]
            elif time_period == "exact":
                (
                    X_sample_train,
                    y_sample_train,
                    weights_train,
                    missing_data_train_HSA,
                ) = prep_training_test_data(
                    subset_HSAs_for_train,
                    no_weeks=no_weeks_train,
                    weeks_in_future=weeks_in_future,
                    geography=geography_column,
                    weight_col=weight_col,
                    keep_output=keep_output,
                )
                (
                    X_sample_test,
                    y_sample_test,
                    weights_test,
                    missing_data_train_HSA,
                ) = prep_training_test_data(
                    subset_HSAs_for_test,
                    no_weeks=no_weeks_test,
                    weeks_in_future=weeks_in_future,
                    geography=geography_column,
                    weight_col=weight_col,
                    keep_output=keep_output,
                )
                weights_train = weights_train[0]
            elif time_period == "shifted":
                (
                    X_sample_train,
                    y_sample_train,
                    weights_train,
                    missing_data_train_HSA,
                ) = prep_training_test_data_shifted(
                    subset_HSAs_for_train,
                    no_weeks=no_weeks_train,
                    weeks_in_future=weeks_in_future,
                    geography=geography_column,
                    weight_col=weight_col,
                    keep_output=keep_output,
                )
                (
                    X_sample_test,
                    y_sample_test,
                    weights_test,
                    missing_data_train_HSA,
                ) = prep_training_test_data_shifted(
                    subset_HSAs_for_test,
                    no_weeks=no_weeks_test,
                    weeks_in_future=weeks_in_future,
                    geography=geography_column,
                    weight_col=weight_col,
                    keep_output=keep_output,
                )
                weights_train = weights_train[0]

        random_search = RandomizedSearchCV(
            classifier, param_grid, n_iter=no_iterations_param, cv=cv, random_state=10
        )
        random_search.fit(X_sample_train, y_sample_train, sample_weight=weights_train)
        best_params = random_search.best_params_

        # Create the Decision Tree classifier with the best hyperparameters
        model = DecisionTreeClassifier(
            **best_params, random_state=10, class_weight="balanced"
        )
        model_fit = model.fit(
            X_sample_train, y_sample_train, sample_weight=weights_train
        )
        y_pred = model_fit.predict_proba(X_sample_test)

        # Evaluate the accuracy of the model
        best_hyperparameters_per_iter.append(best_params)
        auROC_per_iter.append(roc_auc_score(y_sample_test, y_pred[:, 1]))

    return best_hyperparameters_per_iter[np.argmax(np.array(auROC_per_iter))]


def LOOCV_by_HSA_dataset(dataframe, geo_ID, geo_ID_col):
    training_dataframe = dataframe[dataframe[geo_ID_col] != geo_ID]
    testing_dataframe = dataframe[dataframe[geo_ID_col] == geo_ID]
    return training_dataframe, testing_dataframe


def save_in_HSA_dictionary(
    prediction_week,
    ROC_by_week,
    accuracy_by_week,
    sensitivity_by_week,
    specificity_by_week,
    ppv_by_week,
    npv_by_week,
    ROC_by_HSA,
    accuracy_by_HSA,
    sensitivity_by_HSA,
    specificity_by_HSA,
    ppv_by_HSA,
    npv_by_HSA,
):
    ROC_by_HSA[prediction_week] = ROC_by_week
    accuracy_by_HSA[prediction_week] = accuracy_by_week
    sensitivity_by_HSA[prediction_week] = sensitivity_by_week
    specificity_by_HSA[prediction_week] = specificity_by_week
    ppv_by_HSA[prediction_week] = ppv_by_week
    npv_by_HSA[prediction_week] = npv_by_week


def prep_training_test_data_shifted(
    data, no_weeks, weeks_in_future, geography, weight_col, keep_output
):
    ## Get the weeks for the x and y datasets
    x_weeks = []
    y_weeks = []
    y_weeks_to_check = []  # check these weeks to see if any of them are equal to 1
    for week in no_weeks:
        test_week = int(week) + weeks_in_future
        x_weeks.append("_" + num2words(week) + "_")
        for week_y in range(week + 2, test_week + 2):
            y_weeks_to_check.append("_" + num2words(week_y) + "_")
        y_weeks.append("_" + num2words(test_week) + "_")
    ## Divide up the test/train split
    # if is_geographic:
    # Calculate the index to start slicing from
    #    start_index = len(data['county']) // proportion[0] * proportion[1]
    # Divide up the dataset based on this proportion
    #    first_two_thirds = data['county'][:start_index]
    #    last_third = data['county'][start_index:]
    X_data = pd.DataFrame()
    y_data = pd.DataFrame()
    weights_all = pd.DataFrame()
    missing_data = []
    ## Now get the training data
    k = 0
    for x_week in x_weeks:
        y_week = y_weeks[k]
        k += 1

        weeks_x = [col for col in data.columns if x_week in col]
        columns_x = [geography] + weeks_x + [weight_col]
        data_x = data[columns_x]

        weeks_y = [col for col in data.columns if y_week in col]
        columns_y = [geography] + weeks_y
        data_y = data[columns_y]
        ### now add the final column to the y data that has it so that it's if any week in the trhee week perdiod exceeded 15
        train_week = w2n.word_to_num(x_week.replace("_", ""))
        target_week = w2n.word_to_num(y_week.replace("_", ""))
        y_weeks_to_check = []
        for week_to_check in range(
            train_week + 2, target_week + 2
        ):  # have to ensure you skip the next week for getting the excess
            y_weeks_to_check.append("_" + num2words(week_to_check) + "_")
        y_weeks_to_check = [week + "beds_over_15_100k" for week in y_weeks_to_check]
        columns_to_check = [
            col for col in data.columns if any(week in col for week in y_weeks_to_check)
        ]
        y_over_in_period = data[columns_to_check].apply(max, axis=1)
        data_y = pd.concat([data_y, y_over_in_period], axis=1)
        # ensure they have the same amount of data
        # remove rows in test_data1 with NA in test_data2
        data_x = data_x.dropna()
        data_x = data_x[data_x[geography].isin(data_y[geography])]
        # remove rows in test_data2 with NA in test_data1
        data_y = data_y.dropna()
        data_y = data_y[data_y[geography].isin(data_x[geography])]
        data_x = data_x[data_x[geography].isin(data_y[geography])]
        data_x_no_HSA = len(data_x[geography].unique())

        missing_data.append(
            (
                (len(data[geography].unique()) - data_x_no_HSA)
                / len(data[geography].unique())
            )
            * 100
        )
        # get weights
        # weights = weight_data[weight_data[geography].isin(data_x[geography])][[geography, weight_col]]

        X_week = data_x.iloc[:, 1 : len(columns_x)]  # take away y, leave weights for mo
        y_week = data_y.iloc[:, -1]

        y_week = y_week.astype(int)

        weights = X_week.iloc[:, -1]
        if keep_output:
            X_week = X_week.iloc[
                :, : len(X_week.columns) - 1
            ]  # remove the weights and leave "target" for that week

            # rename columns for concatenation
            X_week.columns = range(1, len(data_x.columns) - 1)
        else:
            X_week = X_week.iloc[
                :, : len(X_week.columns) - 2
            ]  # remove the weights and  "target" for that week

            X_week.columns = range(
                1, len(data_x.columns) - 2
            )  # remove the weights and  "target" for that week

        y_week.columns = range(1, len(data_y.columns) - 2)
        X_data = pd.concat([X_data, X_week])
        y_data = pd.concat([y_data, y_week])

        weights_all = pd.concat([weights_all, weights])

    X_data.reset_index(drop=True, inplace=True)
    y_data.reset_index(drop=True, inplace=True)
    weights_all.reset_index(drop=True, inplace=True)

    return (X_data, y_data, weights_all, missing_data)


def LOOCV_by_HSA_dataset(dataframe, geo_ID, geo_ID_col):
    training_dataframe = dataframe[dataframe[geo_ID_col] != geo_ID]
    testing_dataframe = dataframe[dataframe[geo_ID_col] == geo_ID]
    return training_dataframe, testing_dataframe


def save_in_HSA_dictionary(
    prediction_week,
    ROC_by_week,
    accuracy_by_week,
    sensitivity_by_week,
    specificity_by_week,
    ppv_by_week,
    npv_by_week,
    ROC_by_HSA,
    accuracy_by_HSA,
    sensitivity_by_HSA,
    specificity_by_HSA,
    ppv_by_HSA,
    npv_by_HSA,
):
    ROC_by_HSA[prediction_week] = ROC_by_week
    accuracy_by_HSA[prediction_week] = accuracy_by_week
    sensitivity_by_HSA[prediction_week] = sensitivity_by_week
    specificity_by_HSA[prediction_week] = specificity_by_week
    ppv_by_HSA[prediction_week] = ppv_by_week
    npv_by_HSA[prediction_week] = npv_by_week


def convert_state_to_code(dataframe, column_name):
    # List of state names in alphabetical order, including Washington, D.C.
    state_names = [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "Washington D.C.",
    ]

    # Create a dictionary to map state names to numerical codes
    state_to_code = {
        state: f"{index + 1:02}" for index, state in enumerate(state_names)
    }

    # Create a new column "state_code" based on the mapping
    dataframe["state_code"] = dataframe[column_name].map(state_to_code)

    return dataframe


# Define functions for data preparation and evaluation
def prepare_data_and_model(
    data,
    weeks_in_future,
    geography,
    weight_col,
    keep_output,
    time_period,
    model_name,
    prediction_week,
    size_of_test_dataset,
    train_weeks_for_initial_model,
):
    model_name_to_load = model_name + f"_{time_period}_{prediction_week}.sav"
    clf_full = pickle.load(open(model_name_to_load, "rb"))
    if time_period == "period":
        (
            X_train,
            y_train,
            weights_train,
            missing_data_train_HSA,
        ) = prep_training_test_data_period(
            data=data,
            no_weeks=range(1, int(prediction_week + train_weeks_for_initial_model) + 1),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

        (
            X_test,
            y_test,
            weights_test,
            missing_data_test_HSA,
        ) = prep_training_test_data_period(
            data=data,
            no_weeks=range(
                int(prediction_week + train_weeks_for_initial_model) + 1,
                int(
                    prediction_week
                    + train_weeks_for_initial_model
                    + size_of_test_dataset
                )
                + 1,
            ),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )
    elif time_period == "exact":
        (
            X_train,
            y_train,
            weights_train,
            missing_data_train_HSA,
        ) = prep_training_test_data(
            data=data,
            no_weeks=range(1, int(prediction_week + train_weeks_for_initial_model) + 1),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

        (
            X_test,
            y_test,
            weights_test,
            missing_data_test_HSA,
        ) = prep_training_test_data(
            data=data,
            no_weeks=range(
                int(prediction_week + train_weeks_for_initial_model),
                int(
                    prediction_week
                    + train_weeks_for_initial_model
                    + size_of_test_dataset
                )
                + 1,
            ),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

    elif time_period == "shifted":
        (
            X_train,
            y_train,
            weights_train,
            missing_data_train_HSA,
        ) = prep_training_test_data_shifted(
            data=data,
            no_weeks=range(1, int(prediction_week + train_weeks_for_initial_model) + 1),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

        (
            X_test,
            y_test,
            weights_test,
            missing_data_test_HSA,
        ) = prep_training_test_data_shifted(
            data=data,
            no_weeks=range(
                int(prediction_week + train_weeks_for_initial_model) + 1,
                int(
                    prediction_week
                    + train_weeks_for_initial_model
                    + size_of_test_dataset
                )
                + 1,
            ),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

    weights_train = weights_train[0].to_numpy()
    clf_full.fit(X_train, y_train, sample_weight=weights_train)

    y_pred = clf_full.predict(X_test)
    y_pred_proba = clf_full.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    mcc = (matthews_corrcoef(y_test, y_pred) + 1) / 2

    return accuracy, roc_auc, mcc


def calculate_maximum_reget(
    metric, metrics_by_model, models, time_period, weeks_to_predict
):
    maximum_regret_by_model = {model: [] for model in models}

    for prediction_week in weeks_to_predict:
        print(prediction_week)
        best_metric = float("-inf")
        integers_names = [
            w2n.word_to_num(model.split("_")[0]) if "_week" in model else 0
            for model in models
        ]  ## if e.g. expanding model can go to very end

        for i, m in enumerate(metrics_by_model):
            if (
                integers_names[i] <= prediction_week
            ):  # & (abs(max(weeks_to_predict) -  integers_names[i]) >= prediction_week):
                model_metric = m[prediction_week]
            else:
                model_metric = 0
            if model_metric >= best_metric:
                best_metric = model_metric

        for i, m in enumerate(metrics_by_model):
            model = models[i]
            if (
                integers_names[i] <= prediction_week
            ):  # & (abs(max(weeks_to_predict) -  integers_names[i]) >= prediction_week):
                model_metric = m[prediction_week]

                if model_metric >= best_metric:
                    maximum_regret_by_model[model].append(0)
                else:
                    maximum_regret_by_model[model].append(best_metric - model_metric)
            else:
                maximum_regret_by_model[model].append(best_metric)
    return maximum_regret_by_model


# Define functions for data preparation and evaluation
def prepare_data_and_model(
    data,
    weeks_in_future,
    geography,
    weight_col,
    keep_output,
    time_period,
    model_name,
    prediction_week,
    size_of_test_dataset,
    train_weeks_for_initial_model,
):
    model_name_to_load = model_name + f"_{time_period}_{prediction_week}.sav"
    clf_full = pickle.load(open(model_name_to_load, "rb"))
    if time_period == "period":
        (
            X_train,
            y_train,
            weights_train,
            missing_data_train_HSA,
        ) = prep_training_test_data_period(
            data=data,
            no_weeks=range(1, int(prediction_week + train_weeks_for_initial_model) + 1),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

        (
            X_test,
            y_test,
            weights_test,
            missing_data_test_HSA,
        ) = prep_training_test_data_period(
            data=data,
            no_weeks=range(
                int(prediction_week + train_weeks_for_initial_model) + 1,
                int(
                    prediction_week
                    + train_weeks_for_initial_model
                    + size_of_test_dataset
                )
                + 1,
            ),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )
    elif time_period == "shifted":
        (
            X_train,
            y_train,
            weights_train,
            missing_data_train_HSA,
        ) = prep_training_test_data_shifted(
            data=data,
            no_weeks=range(1, int(prediction_week + train_weeks_for_initial_model) + 1),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

        (
            X_test,
            y_test,
            weights_test,
            missing_data_test_HSA,
        ) = prep_training_test_data_shifted(
            data=data,
            no_weeks=range(
                int(prediction_week + train_weeks_for_initial_model) + 1,
                int(
                    prediction_week
                    + train_weeks_for_initial_model
                    + size_of_test_dataset
                )
                + 1,
            ),
            weeks_in_future=weeks_in_future,
            geography=geography,
            weight_col=weight_col,
            keep_output=keep_output,
        )

    weights_train = weights_train[0].to_numpy()
    clf_full.fit(X_train, y_train, sample_weight=weights_train)

    y_pred = clf_full.predict(X_test)
    y_pred_proba = clf_full.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    mcc = (matthews_corrcoef(y_test, y_pred) + 1) / 2

    return accuracy, roc_auc, mcc


def simplify_labels_graphviz(graph):
    for node in graph.get_node_list():
        if node.get_attributes().get("label") is None:
            continue
        else:
            split_label = node.get_attributes().get("label").split("<br/>")
            if len(split_label) == 4:
                split_label[3] = split_label[3].split("=")[1].strip()

                del split_label[1]  # number of samples
                del split_label[1]  # split of sample
            elif len(split_label) == 3:  # for a terminating node, no rule is provided
                split_label[2] = split_label[2].split("=")[1].strip()

                del split_label[0]  # number of samples
                del split_label[0]  # split of samples
                split_label[0] = "<" + split_label[0]
            node.set("label", "<br/>".join(split_label))


def calculate_maximum_reget(
    metric, metrics_by_model, models, time_period, weeks_to_predict
):
    metric_data = metrics_by_model[metric]

    maximum_regret_by_model = {model: [] for model in models}
    for j, prediction_week in enumerate(weeks_to_predict):
        best_metric = float("-inf")

        for i, m in enumerate(metric_data):
            m = list(metric_data.values())[i]
            model_metric = m[prediction_week]
            if model_metric >= best_metric:
                best_metric = model_metric

        for i, m in enumerate(metric_data):
            m = list(metric_data.values())[i]
            model_metric = m[prediction_week]
            model = models[i]
            if model_metric >= best_metric:
                maximum_regret_by_model[model].append(0)
            else:
                maximum_regret_by_model[model].append(best_metric - model_metric)

    return maximum_regret_by_model


def calculate_ppv_npv(confusion_matrix):
    # Extract values from the confusion matrix
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]

    # Calculate PPV (Precision) and NPV
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    return ppv, npv


# Get summary statistics


def process_feature_data(strings, data_dataframe, reults_dataframe):
    for col in strings:
        feature_data_wide = data_dataframe.filter(regex=col)
        feature_data = pd.melt(
            feature_data_wide, var_name="Feature", value_name="Value"
        )

        mean_val = feature_data["Value"].mean()
        std_dev_val = feature_data["Value"].std()

        # Append the results to the result DataFrame
        reults_dataframe = pd.concat(
            [
                reults_dataframe,
                pd.DataFrame(
                    {
                        "Subset": [col],
                        "Mean": [mean_val],
                        "Std Deviation": [std_dev_val],
                    }
                ),
            ],
            ignore_index=True,
        )

    return reults_dataframe
