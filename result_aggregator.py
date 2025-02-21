# agrregate all the scores in one df
from unittest import result
import pandas as pd
import os
from termcolor import colored
from utils import score_plot_new


def main(
    data_folder: str,
    CV: bool,
    # n_splits: int,
    stratify: bool,
    num_epoch_pretrain: int,
    num_epoch: int,
    mogonet_model: str,
) -> None:
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
    numerical_identity = (
        str(num_epoch_pretrain) + "_" + str(num_epoch) + "_" + mogonet_model
    )

    exact_model = os.path.join(
        model_folder_path, str(num_epoch_pretrain) + "_" + str(num_epoch), mogonet_model
    )
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

    # score_plot_new(path, info, save_path)
    score_plot_new(
        result_file_path,
        "new_data_orig_0.05" + result_file_name_suffix,
        result_folder,
    )


if __name__ == "__main__":
    main(
        data_folder="0_new_data_0.05",
        # "cfrna_metabo_disc225(0.001_95%)_TriStrat2",
        # "cfrna_metabo_disc225(0.001_95%)",  # "cfrna_metabo_disc225",  # "cfrna_disc225",  # "cfrna_disc225",  # "cfrna_metabo_disc225",  # "trial"  # "discovery_cohort"  # "PE_cfRNA"
        CV=True,
        num_epoch_pretrain=400,
        num_epoch=1200,
        stratify=True,
        mogonet_model="20240610-151416",
        # "20240610-144922",  # 0.05 strat
        # "20240610-144001",  # 0.05 no_strat
        # "20240610-131020",  # new data orig no strat
        # "20240610-130634",  # new data orig strat
        # "20240610-120507",  # new data strat
        # "20240610-115805",  # new data no strat
        # "20240408-013504",  # "cfrna_metabo_disc225(0.001_95%)_TriStrat2"
        # "20240408-005942",  # "cfrna_metabo_disc225(0.001_95%)_TriStrat"
        # "20240124-204913"
        # "20240124-202506"
        # "20240123-160739"  # cpm95_cfrna001 strat_CV 400_1200
        # "20240122-190025"  # 10_CV
        # "20240122-180033"  # cpm95_cfrna001_metabo2 strat_CV 400_1200
        # "20240124-202506",  # cpm95_cfrna001_metabo2 no_strat_CV 400_1200
        # "20240122-161105"  # cpm_cfrna_metabo strat_CV 400_1200
        # "20240119-110327"  # mteabo no_strat_CV 400_1200
        # "20240119-105801"  # metabo strat_CV 400_1200
        # "20240119-102921"  # 70-30 start_CV 400_1200
        # "20240119-102406"  # 70-30 no_strat_CV 400_1200
        # "20240118-195725"  # cfRNA strat_CV 400_1200
        # "20240119-112622"  # cfRNA no_strat_CV 400_1200
        # "20240118-193318"  # "20240118-185531"  # strat_CV 400_1200
        # "20240118-192721"  # "20240118-183700"  # no_strat_CV 400_1200
        # "20240118-174447"  # no_strat_NO_CV 400_1200
        # "20240117-140446"  # strat_CV 400_1200
        # "20240117-133600",  # no_strat_CV 300_1200
        # "20240112-185800",  # no_strat_NO_CV
    )
    # "20230915-021156")
    # "20230914-223605")
