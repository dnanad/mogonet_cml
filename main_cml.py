# import packages
import os
import pickle
import pandas as pd

# import functions from project modules
from models import construct_model
from utils import get_pipelines, get_expname_datetime, find_numFolders_maxNumFolders


from train_test import fit_predict_evaluate


def main(
    data_folder,
    cml_model,
    n_splits,
    CV,
    stratify,
    test_mode,
):
    """
    Args: None
    Returns: None
    """
    # Set the mode
    mode = "test" if test_mode else "gridsearch"

    # General parameters
    options = {
        "model": cml_model,  # DTC, RFC, SVC
        "mode": mode,
    }
    rootfigname = options["model"] + "_" + options["mode"]
    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, "data", data_folder)

    if stratify:
        sample_type_path = os.path.join(data_folder_path, "strat")
    else:
        sample_type_path = os.path.join(data_folder_path, "no_strat")

    if CV:
        folder = os.path.join(sample_type_path, "CV")
    else:
        folder = os.path.join(sample_type_path, "NO_CV")

    folder_path = os.path.join(rootpath, folder)
    # data_folder_path = os.path.join(folder_path, "cml")
    model_folder_path = os.path.join(folder_path, "models")

    # construct an experiment name based on current date time
    expname = get_expname_datetime(options)
    latest_model = os.path.join(model_folder_path, "CML", expname)
    os.makedirs(
        latest_model
        if not os.path.exists(latest_model)
        else print("Overwriting contents in existing plot dir")
    )

    if CV:
        print("Cross validation")
        for i in range(n_splits):
            cv_folder = os.path.join(folder_path, "CV_" + str(i))
            cv_model_path = os.path.join(latest_model, "CV_" + str(i))
            print(cv_folder)
            X_train = pd.read_csv(
                os.path.join(cv_folder, "cml", "cml_train.csv"), header=0
            )
            y_train = pd.read_csv(os.path.join(cv_folder, "labels_tr.csv"), header=None)

            X_test = pd.read_csv(
                os.path.join(cv_folder, "cml", "cml_test.csv"), header=0
            )
            y_test = pd.read_csv(os.path.join(cv_folder, "labels_te.csv"), header=None)

            # Build the model
            model, GSparameters = construct_model(opt=options["model"])

            # Get the pipeline for the node and edge models
            pl = get_pipelines(options, X_train, model)

            print("\nTraining the model for the trial", i, ":")
            # do the fitting and predictions for nodes
            scores = fit_predict_evaluate(
                options,
                pl,
                X_train,
                y_train,
                X_test,
                y_test,
                GSparameters,
                Savepath=cv_model_path,
                rootfigname=rootfigname,
                threshold=0.5,
                cm_normalised=False,
            )
            if not os.path.exists(cv_model_path):
                os.makedirs(cv_model_path)
            with open(
                os.path.join(cv_model_path, expname + "_scores.pkl"), "wb"
            ) as handle:
                pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(
                os.path.join(cv_model_path, expname + "_pipeline.pkl"), "wb"
            ) as handle:
                pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"\nSaved the options, scores and pipeline for trial {i} in: {cv_model_path}",
            )
        # open the pickle file for saved the scores for each trial and create a dataframe with the model name and mean +/- std of the scores
        scores_df = pd.DataFrame()
        for i in range(n_splits):
            cv_model_path = os.path.join(latest_model, "CV_" + str(i))
            with open(
                os.path.join(cv_model_path, expname + "_scores.pkl"), "rb"
            ) as handle:
                scores = pickle.load(handle)
            # make dict to df
            scores = pd.DataFrame.from_dict(scores, orient="index")
            scores_df = pd.concat([scores_df, scores], ignore_index=True)
        # add the model name
        scores_df["model"] = options["model"]
        scores_df.to_csv(os.path.join(latest_model, expname + "_detail_scores.csv"))

        print(f"\nSaved the scores in: {latest_model}")

    else:
        # Load the data
        # DF_train, DF_test = get_data(CSVdatapath)

        X_train = pd.read_csv(
            os.path.join(folder_path, "cml", "cml_train.csv"), header=0
        )
        # drop the first column
        X_train = X_train.drop(X_train.columns[0], axis=1)
        y_train = pd.read_csv(os.path.join(folder_path, "labels_tr.csv"), header=None)

        X_test = pd.read_csv(os.path.join(folder_path, "cml", "cml_test.csv"), header=0)
        # drop the first column
        X_test = X_test.drop(X_test.columns[0], axis=1)
        y_test = pd.read_csv(os.path.join(folder_path, "labels_te.csv"), header=None)

        # Build the model
        model, GSparameters = construct_model(opt=options["model"])

        # Get the pipeline for the node and edge models
        pl = get_pipelines(options, X_train, model)

        print("\nTraining the model:")
        # do the fitting and predictions for nodes
        scores = fit_predict_evaluate(
            options,
            pl,
            X_train,
            y_train,
            X_test,
            y_test,
            GSparameters,
            Savepath=latest_model,
            rootfigname=rootfigname,
            threshold=0.5,
            cm_normalised=True,
        )
        with open(os.path.join(latest_model, expname + "_scores.pkl"), "wb") as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            os.path.join(latest_model, expname + "_pipeline.pkl"), "wb"
        ) as handle:
            pickle.dump(pl, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nSaved the options, scores and pipeline in: {latest_model}")
    return


if __name__ == "__main__":
    for cml_model in ["XGBC", "DTC", "RFC", "SVC", "LRC", "ElasticNet"]:
        main(
            data_folder="0_new_data_0.05",  # "cfrna_metabo_disc225(0.001_95%)_TriStrat2",
            # "cfrna_disc225(0.001_95%)",  # "cfrna_metabo_disc225",  # "cfrna_disc225",  # "metabo_disc225",  # "cfrna_disc225",  # "cfrna_metabo_disc225",  # "trial",  # "PE_cfRNA",  # "TEST_DATA",  # "PE_cfRNA"  # "TEST_DATA"  # "ROSMAP"
            cml_model=cml_model,
            n_splits=5,
            CV=True,
            stratify=True,
            test_mode=False,
        )
