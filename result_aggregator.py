# agrregate all the scores in one df

import pandas as pd
import os
import argparse
from termcolor import colored
from utils import score_plot_new


def main(
    data_folder: str,
    CV: bool,
    # n_splits: int,
    stratify: bool,
    num_epoch_pretrain: int,
    num_epoch: int,
    # mogonet_model: str,
) -> None:
    print(colored("Aggregating results", "red"))
    print(colored(f"Data folder: {data_folder}", "green"))
    print(colored(f"CV: {CV}", "green"))
    print(colored(f"Stratify: {stratify}", "green"))
    print(colored(f"Number of pretrain epochs: {num_epoch_pretrain}", "green"))
    print(colored(f"Number of epochs: {num_epoch}", "green"))
    # load the scores for mogonet for the mogonet_model

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, "data", data_folder)
    result_folder = os.path.join(data_folder_path, "results")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if stratify:
        sample_type_path = os.path.join(data_folder_path, "strat")
        result_file_name_suffix = "strat"
    else:
        sample_type_path = os.path.join(data_folder_path, "no_strat")
        result_file_name_suffix = "no_strat"
    if CV:
        # cv_folder_path = os.path.join(sample_type_path, "CV")
        model_folder_path = os.path.join(sample_type_path, "CV", "models")
        result_file_name_suffix = result_file_name_suffix + "_CV"
        # model_folder_path = os.path.join(rootpath, model_folder)
    else:
        # nocv_folder_path = os.path.join(sample_type_path, "NO_CV")
        model_folder_path = os.path.join(sample_type_path, "NO_CV", "models")
        result_file_name_suffix = result_file_name_suffix + "_NO_CV"
        # model_folder_path = os.path.join(rootpath, model_folder)

    # exact_model for mogonet
    location_folder = str(num_epoch_pretrain) + "_" + str(num_epoch)
    # extract the mogonet_model name data/0_new_data_0.05/strat/CV/models/400_1200/20250224-160935. The last part is the mogonet_model

    exact_model_location = os.path.join(model_folder_path, location_folder)

    # extract the folder name inside the exact_model_location
    mogonet_model = os.listdir(exact_model_location)[0]

    numerical_identity = location_folder + "_" + mogonet_model
    exact_model = os.path.join(exact_model_location, mogonet_model)

    mogonet_score_file = "mogonet_" + numerical_identity + "_detail_scores.csv"
    score_path = os.path.join(exact_model, mogonet_score_file)
    # print(colored(score_path, "green"))

    scores_df_exact = pd.read_csv(score_path, index_col=0)
    # print(scores_df_exact)

    # load all the scores from the cml folders
    cml_scores_df = pd.DataFrame()
    for folder in os.listdir(model_folder_path):
        if folder.startswith("CML"):
            # go to each folder one by one
            cml_folder_path = os.path.join(model_folder_path, folder)
            # open each foler at folder path and load the csv
            for subfolder in os.listdir(cml_folder_path):
                subfolder_path = os.path.join(cml_folder_path, subfolder)
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        score_path_cml = os.path.join(subfolder_path, file)
                        scores_df_cml = pd.read_csv(score_path_cml, index_col=0)
                        cml_scores_df = pd.concat(
                            [cml_scores_df, scores_df_cml], ignore_index=True
                        )
    # merge the two df
    final_scores = pd.concat([cml_scores_df, scores_df_exact], ignore_index=True)
    # save the final_scores at cv_folder_path
    # if CV:
    #     final_scores["model"] = "mogonet"
    #     # save the scores
    #     score_filename = (
    #         "mogonet_" + numerical_identity + "_detail_scores_aggregated.csv"
    #     )
    result_file_name = "final_scores_" + result_file_name_suffix + ".csv"
    result_file_path = os.path.join(result_folder, result_file_name)
    final_scores.to_csv(result_file_path)
    print("Final scores saved at: ", result_file_path)

    # score_plot_new(path, info, save_path)
    score_plot_new(
        result_file_path,
        "new_data_orig_0.05" + result_file_name_suffix,
        result_folder,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate results for multi-omic analysis"
    )
    parser.add_argument("--data_folder", required=True, help="Path to the data folder")
    parser.add_argument(
        "--CV", action="store_true", help="Whether to use cross-validation"
    )
    parser.add_argument(
        "--no_CV", action="store_false", dest="CV", help="Do not use cross-validation"
    )
    parser.add_argument("--stratify", action="store_true", help="Stratify!")
    parser.add_argument(
        "--no_stratify", action="store_false", dest="stratify", help="Do not stratify"
    )
    parser.add_argument(
        "--num_epoch_pretrain",
        type=int,
        required=True,
        help="Number of pretrain epochs",
    )
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        data_folder=args.data_folder,
        CV=args.CV,
        stratify=args.stratify,
        num_epoch_pretrain=args.num_epoch_pretrain,
        num_epoch=args.num_epoch,
    )
    # main(
    #     data_folder="0_new_data_0.05",
    #     CV=True,
    #     num_epoch_pretrain=400,
    #     num_epoch=1200,
    #     stratify=True,
    #     mogonet_model="20240610-151416",
    # )
