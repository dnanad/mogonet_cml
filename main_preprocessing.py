# introduce the author and what is the script about
"""
==============================================================================
Created on Wed 27 Dec 2023 22:00:00
Author: Anand Deshpande
Script: main_preprocessing_new.py
Description: This script is used to preprocess the omic data and save the dataframes for CML
==============================================================================
Steps:
1. Import and process the data
2. Find the common sample ids
3. Save the common sample ids
4. Save the labels
5. Save the features
6. Train test split
7. Save the train and test data
8. Differential analysis
9. Filter omic feature using the differential analysis results
10. Save the filtered differential analysis results
11. Split the omic data into train and test
12. Normalize the omic data with respect to the train data
13. Transform the test data with respect to the train data
14. Save the train and test data
15. Stag the all the omics data for CML
14. Save the train and test data for CML
15. Save the common features for CML
16. Save the labels for CML
"""
from calendar import c
from genericpath import exists
from turtle import st
import comm
import pandas as pd
import os
import pickle

from sklearn import preprocessing
from utils import (
    # import_process_datafile,
    # get_label_dict,
    create_dict_from_col,
    save_feat_name,
    train_test_save,
    # dataset_summary,
    find_numFolders_maxNumFolders,
    train_test_split,
    save_labels,
    save_sample_ids,
    # find_common_sample_ids,
    omicwise_filtering,
    preprocessing_omic_data,
    differential_analysis,
    filter_differential_results,
)


def main(
    data_folder: str,
    stratify: bool,
    strat_col_list: list,
    test_size: float,
    omic_normalize_dict: dict,  # values are boolean
    da_omics_dict: dict,  # values are boolean
    filter_wrt_dict: dict,  # values are string
    method_dict: str,  # values are string
    da_threshold_dict: dict,  # values are float
    CV: bool,
    n_splits: int,
):
    # creta a path to the data folder
    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, "data", data_folder)

    # create a path to the labels
    labels_folder_path = os.path.join(data_folder_path, "labels")
    label_files = os.listdir(labels_folder_path)
    labels = label_files[0]
    labels_path = os.path.join(data_folder_path, "labels", labels)

    # find the number of omics and the maximum number of folders
    omics_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    num_omics = len(omics_list)

    # preprocessing the omic data
    sample_ids_dict = preprocessing_omic_data(omics_list, data_folder_path)
    sample_folder = os.path.join(data_folder_path, "samples")

    # save sample ids
    save_sample_ids(sample_ids_dict, sample_folder)
    # read the labels
    labels_df = pd.read_csv(labels_path)
    # labels
    labels_dict = create_dict_from_col(
        labels_df, "Maternal_woman_id", "fetal_near_miss"
    )
    X = labels_df["Maternal_woman_id"]
    y = labels_df["fetal_near_miss"]
    # Assuming `df` is your DataFrame and `cols` is your list of column names
    if stratify:
        labels_df["strat_col"] = labels_df[strat_col_list].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )
        y_strat = labels_df["strat_col"]
        strat_dict = create_dict_from_col(labels_df, "Maternal_woman_id", "strat_col")
    else:
        y_strat = y
        strat_dict = labels_dict
    common_sample_ids = X.tolist()

    # save the common sample ids
    for j in omics_list:
        omicwise_filtering(j, data_folder_path, common_sample_ids)

    if CV:
        print("Cross validation")
        train_test_folder = train_test_split(
            X=X,
            y=y,
            y_strat=y_strat,
            test_size=test_size,
            sample_folder=sample_folder,
            labels_folder_path=labels_folder_path,
            n_splits=n_splits,
            CV=CV,
            stratify=stratify,
        )

        for i in range(n_splits):
            print("Split: ", i)
            cv_folder = os.path.join(train_test_folder, "CV_" + str(i))
            train = pd.read_pickle(
                os.path.join(cv_folder, "train_" + str(i) + ".pickle")
            )
            test = pd.read_pickle(os.path.join(cv_folder, "test_" + str(i) + ".pickle"))
            print(
                f"For cv split {i}: Splitting the omic data into train and test saving them plus saving the respective features"
            )

            for j in omics_list:
                if da_omics_dict[j]:
                    omic_path = os.path.join(data_folder_path, str(j))
                    common_processed_data_file_name = (
                        str(j) + "_common_processed_data.csv"
                    )
                    common_df = pd.read_csv(
                        os.path.join(omic_path, common_processed_data_file_name),
                        index_col=0,
                    )

                    filter_wrt = filter_wrt_dict[j]
                    method = method_dict[j]
                    da_threshold = da_threshold_dict[j]
                    differential_results_df, diff_folder = differential_analysis(
                        j=j,
                        omic_path=omic_path,
                        common_df=common_df,
                        train=train,
                        strat_dict=strat_dict,
                        filter_wrt=filter_wrt,
                        method=method,
                        da_threshold=da_threshold,
                        i=i,
                        CV=CV,
                        stratify=stratify,
                    )

                    filter_differential_results_df = filter_differential_results(
                        j=j,
                        diff_folder=diff_folder,
                        differential_results_df=differential_results_df,
                        filter_wrt=filter_wrt,
                        method=method,
                        da_threshold=da_threshold,
                        i=i,
                        CV=CV,
                    )
                else:
                    pass
            for j in omics_list:
                if da_omics_dict[j]:
                    omic_path = os.path.join(data_folder_path, str(j))
                    if stratify:
                        omic_path_sample_type = os.path.join(omic_path, "strat")
                    else:
                        omic_path_sample_type = os.path.join(omic_path, "no_strat")

                    diff_folder = os.path.join(
                        omic_path_sample_type, "differential_analysis", "CV", "filtered"
                    )
                    if not os.path.exists(diff_folder):
                        os.makedirs(diff_folder)
                    CV_folder = os.path.join(omic_path_sample_type, "feat_CV")
                    if not os.path.exists(CV_folder):
                        os.makedirs(CV_folder)
                    # open all the csv files in the folder diff_folder take the "feature" columns from each of the csv take union of all the features   and save it as a csv file
                    common_features = set()
                    for file in os.listdir(diff_folder):
                        if file.endswith(".csv"):
                            df = pd.read_csv(os.path.join(diff_folder, file))
                            common_features = common_features.union(set(df["feature"]))
                    common_features_df = pd.DataFrame(
                        common_features, columns=["feature"]
                    )
                    common_feature_file_path = os.path.join(
                        CV_folder, "common_features.csv"
                    )
                    common_features_df.to_csv(common_feature_file_path, index=False)
                else:
                    pass

        for i in range(n_splits):
            cv_folder = os.path.join(train_test_folder, "CV_" + str(i))
            train = pd.read_pickle(
                os.path.join(cv_folder, "train_" + str(i) + ".pickle")
            )
            test = pd.read_pickle(os.path.join(cv_folder, "test_" + str(i) + ".pickle"))
            omic_train_dfs = {}
            omic_test_dfs = {}
            omic_feat_names = {}
            for j in omics_list:
                omic_path = os.path.join(data_folder_path, str(j))
                if stratify:
                    omic_path_sample_type = os.path.join(omic_path, "strat")
                else:
                    omic_path_sample_type = os.path.join(omic_path, "no_strat")
                common_processed_data_file_name = str(j) + "_common_processed_data.csv"
                common_df = pd.read_csv(
                    os.path.join(omic_path, common_processed_data_file_name),
                    index_col=0,
                )
                if da_omics_dict[j]:
                    common_features_df = pd.read_csv(
                        os.path.join(
                            omic_path_sample_type,
                            "feat_CV",
                            "common_features.csv",
                        )
                    )
                    # common_features_df = common_features_df["feature"]
                    # drop nan values
                    common_features_df = common_features_df.dropna()
                    common_df_with_da = common_df[common_features_df["feature"]]

                    cv_split_folder = os.path.join(
                        omic_path_sample_type, "feat_CV", "CV_" + str(i)
                    )
                    if not os.path.exists(cv_split_folder):
                        os.makedirs(cv_split_folder)
                    common_df_with_da.to_csv(
                        os.path.join(
                            cv_split_folder,
                            str(j) + "_common_processed_data_with_da.csv",
                        )
                    )
                else:
                    common_df_with_da = common_df

                df_feat = save_feat_name(
                    j=j,
                    df=common_df_with_da,
                    data_folder_path=data_folder_path,
                    i=i,
                    CV=CV,
                    stratify=stratify,
                )
                omic_train, omic_test = train_test_save(
                    j=j,
                    df=common_df_with_da,
                    train=train,
                    test=test,
                    omic_normalize_dict=omic_normalize_dict,
                    data_folder_path=data_folder_path,
                    i=i,
                    CV=CV,
                    stratify=stratify,
                )

                # Store the dataframes in the dictionaries
                omic_train_dfs[j] = omic_train
                omic_test_dfs[j] = omic_test
                omic_feat_names[j] = df_feat

            # Concatenate the dataframes horizontally for cml
            cml_train = pd.concat(omic_train_dfs.values(), axis=1)
            cml_test = pd.concat(omic_test_dfs.values(), axis=1)
            cml_feat_names = pd.concat(omic_feat_names.values(), axis=0)

            print(f"Split {i}: Saving the dataframes for CML")
            # save the dataframes
            if stratify:
                sample_type_path = os.path.join(data_folder_path, "strat")
            else:
                sample_type_path = os.path.join(data_folder_path, "no_strat")

            folder_name = "CV_" + str(i)
            folder_path = os.path.join(sample_type_path, "CV", folder_name)
            cml_folder_path = os.path.join(folder_path, "cml")
            if not os.path.exists(cml_folder_path):
                os.makedirs(cml_folder_path)
            cml_train.to_csv(os.path.join(cml_folder_path, "cml_train.csv"))
            cml_test.to_csv(os.path.join(cml_folder_path, "cml_test.csv"))
            cml_feat_names.to_csv(os.path.join(cml_folder_path, "cml_feat_names.csv"))

            print(f"Split {i}: Saving the dataframes for labels")

            # labels_dict = get_label_dict(labels_path)
            save_labels(
                labels_dict=labels_dict,
                train=train,
                test=test,
                data_folder_path=sample_type_path,
                i=i,
                CV=CV,
            )
    else:
        CV = False
        print("No cross validation")
        # split  and save the common_samples_ids into train and test

        train_test_folder = train_test_split(
            X=X,
            y=y,
            y_strat=y_strat,
            test_size=test_size,
            sample_folder=sample_folder,
            labels_folder_path=labels_folder_path,
            n_splits=None,
            CV=CV,
            stratify=stratify,
        )

        # open saved pickel files
        train = pd.read_pickle(os.path.join(train_test_folder, "Xtrain.pickle"))
        test = pd.read_pickle(os.path.join(train_test_folder, "Xtest.pickle"))

        print(
            "Splitting the omic data into train and test saving them plus saving the respective features"
        )
        # omic wise save features and train test split

        omic_train_dfs = {}
        omic_test_dfs = {}
        omic_feat_names = {}
        for j in omics_list:
            omic_path = os.path.join(data_folder_path, str(j))
            if stratify:
                omic_path_sample_type = os.path.join(omic_path, "strat")
            else:
                omic_path_sample_type = os.path.join(omic_path, "no_strat")
            common_processed_data_file_name = str(j) + "_common_processed_data.csv"
            common_df = pd.read_csv(
                os.path.join(omic_path, common_processed_data_file_name),
                index_col=0,
            )
            if da_omics_dict[j]:
                filter_wrt = filter_wrt_dict[j]
                method = method_dict[j]
                da_threshold = da_threshold_dict[j]
                differential_results_df, diff_folder = differential_analysis(
                    j=j,
                    omic_path=omic_path,
                    common_df=common_df,
                    train=train,
                    strat_dict=strat_dict,
                    filter_wrt=filter_wrt,
                    method=method,
                    da_threshold=da_threshold,
                    i=None,
                    CV=CV,
                    stratify=stratify,
                )

                filter_differential_results_df = filter_differential_results(
                    j=j,
                    diff_folder=diff_folder,
                    differential_results_df=differential_results_df,
                    filter_wrt=filter_wrt,
                    method=method,
                    da_threshold=da_threshold,
                    i=None,
                    CV=CV,
                )
                common_feat_folder = os.path.join(omic_path_sample_type, "feat_no_CV")
                if not os.path.exists(common_feat_folder):
                    os.makedirs(common_feat_folder)
                common_df_with_da = common_df[filter_differential_results_df["feature"]]
                common_df_with_da.to_csv(
                    os.path.join(
                        common_feat_folder,
                        str(j) + "_common_processed_data_with_da.csv",
                    )
                )
            else:
                common_df_with_da = common_df

            # save features
            df_feat = save_feat_name(
                j=j,
                df=common_df_with_da,
                data_folder_path=data_folder_path,
                i=None,
                CV=CV,
                stratify=stratify,
            )

            # train test split
            omic_train, omic_test = train_test_save(
                j=j,
                df=common_df_with_da,
                train=train,
                test=test,
                omic_normalize_dict=omic_normalize_dict,
                data_folder_path=data_folder_path,
                i=None,
                CV=CV,
                stratify=stratify,
            )  # save the train and test data

            # Store the dataframes in the dictionaries
            omic_train_dfs[j] = omic_train
            omic_test_dfs[j] = omic_test
            omic_feat_names[j] = df_feat

        # Concatenate the dataframes horizontally for cml
        cml_train = pd.concat(omic_train_dfs.values(), axis=1)
        cml_test = pd.concat(omic_test_dfs.values(), axis=1)
        cml_feat_names = pd.concat(omic_feat_names.values(), axis=0)

        print("Saving the dataframes for CML")
        # save the dataframes
        if stratify:
            sample_type_path = os.path.join(data_folder_path, "strat")
        else:
            sample_type_path = os.path.join(data_folder_path, "no_strat")
        folder_name = "NO_CV"
        folder_path = os.path.join(sample_type_path, folder_name)
        cml_folder_path = os.path.join(folder_path, "cml")
        if not os.path.exists(cml_folder_path):
            os.makedirs(cml_folder_path)
        cml_train.to_csv(os.path.join(cml_folder_path, "cml_train.csv"))
        cml_test.to_csv(os.path.join(cml_folder_path, "cml_test.csv"))
        cml_feat_names.to_csv(os.path.join(cml_folder_path, "cml_feat_names.csv"))

        print("Saving the dataframes for labels")

        save_labels(
            labels_dict=labels_dict,
            train=train,
            test=test,
            data_folder_path=sample_type_path,
            i=None,
            CV=CV,
        )

    print("Preprocessing completed!")


if __name__ == "__main__":
    main(
        data_folder="0_new_data_0.05",  
        strat_col_list=["Trimester", "fetal_near_miss"],
        test_size=0.2,
        omic_normalize_dict={
            1: True,
            2: True,
        },  # {1: True, 2: True},  # {1: True, 2: True, 3: False},
        da_omics_dict={
            1: True,
            2: True,
        },  # {1: True, 2: True},  # {1: True, 2: True, 3: True},
        filter_wrt_dict={
            1: "p-value(f-test)",
            2: "p-value(f-test)",  
        },  # 'p-value(t-test)', 'p-value(f-test)', 'adjusted_pvalue(f-test)', 'adjusted_pvalue(t-test)'
        method_dict={
            1: "bonferroni",
            2: "bonferroni",
        },  # "fdr_bh",#  # "bonferroni", 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
        da_threshold_dict={1: 0.05, 2: 0.2},
        CV=True,  # True #Cross validation, False # No cross validation
        n_splits=5,
    )
