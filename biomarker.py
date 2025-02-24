""" Example for biomarker identification
"""

import os
import copy
import argparse
import json
from feat_importance import cal_feat_imp, summarize_imp_feat
from utils import find_numFolders_maxNumFolders


def main(
    data_folder: str,
    stratify: bool,
    CV: bool,
    n_splits: int,
    num_epoch_pretrain: int,
    num_epoch: int,
    num_class: int,
    adj_parameter: int,
    dim_he_list: list,
) -> None:
    print("Calculating feature importance")
    print(f"Data folder: {data_folder}")
    print(f"Stratify: {stratify}")
    print(f"CV: {CV}")
    print(f"Number of splits: {n_splits}")
    print(f"Number of pretrain epochs: {num_epoch_pretrain}")
    print(f"Number of epochs: {num_epoch}")

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, "data", data_folder)

    if stratify:
        sample_type_path = os.path.join(data_folder_path, "strat")
        feat_file_suffix = "_strat"
    else:
        sample_type_path = os.path.join(data_folder_path, "no_strat")
        feat_file_suffix = "_no_strat"

    if CV:
        folder = os.path.join(sample_type_path, "CV")
        folder_path = os.path.join(rootpath, folder)
        model_folder = os.path.join(folder, "models")
        model_folder_path = os.path.join(rootpath, model_folder)
        feat_file_suffix = feat_file_suffix + "_CV"
    else:
        folder = os.path.join(sample_type_path, "NO_CV")
        folder_path = os.path.join(rootpath, folder)
        model_folder = os.path.join(folder, "models")
        model_folder_path = os.path.join(rootpath, model_folder)
        feat_file_suffix = feat_file_suffix + "_NO_CV"

    exp_epoch = str(num_epoch_pretrain) + "_" + str(num_epoch)
    exp = os.path.join(model_folder_path, exp_epoch)
    print(f"Model folder: {exp}")
    print(f"Model: {os.listdir(exp)[0]}")
    dt_string = os.listdir(exp)[0]
    latest_model = os.path.join(exp, dt_string)

    view_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    num_view = len(view_list)
    topn = 40

    featimp_list_list = []
    if CV:
        for i in range(n_splits):
            cv_folder = os.path.join(folder_path, "CV_" + str(i))
            exp_path = os.path.join(latest_model, "CV_" + str(i))
            featimp_list = cal_feat_imp(
                cv_folder, exp_path, view_list, num_class, adj_parameter, dim_he_list
            )
            featimp_list_list.append(copy.deepcopy(featimp_list))
    else:
        featimp_list = cal_feat_imp(
            folder_path, latest_model, view_list, num_class, adj_parameter, dim_he_list
        )
        featimp_list_list.append(copy.deepcopy(featimp_list))

    summarize_imp_feat(
        featimp_list_list, exp, topn=topn, feat_file_suffix=feat_file_suffix
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Biomarker identification")
    parser.add_argument("--data_folder", required=True, help="Path to the data folder")
    parser.add_argument("--stratify", action="store_true", help="Stratify!")
    parser.add_argument(
        "--no_stratify", action="store_false", dest="stratify", help="Do not stratify"
    )
    parser.add_argument(
        "--CV", action="store_true", help="Whether to use cross-validation"
    )
    parser.add_argument(
        "--no_CV", action="store_false", dest="CV", help="Do not use cross-validation"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        required=True,
        help="Number of splits for cross-validation",
    )
    parser.add_argument(
        "--num_epoch_pretrain",
        type=int,
        required=True,
        help="Number of pretrain epochs",
    )
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs")

    parser.add_argument(
        "--num_class", type=int, required=True, help="Number of classes"
    )
    parser.add_argument(
        "--adj_parameter", type=int, required=True, help="Adjacency parameter"
    )
    parser.add_argument(
        "--dim_he_list",
        type=json.loads,
        required=True,
        help="Dimension list for hidden layers",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        data_folder=args.data_folder,
        stratify=args.stratify,
        CV=args.CV,
        n_splits=args.n_splits,
        num_epoch_pretrain=args.num_epoch_pretrain,
        num_epoch=args.num_epoch,
        num_class=args.num_class,
        adj_parameter=args.adj_parameter,
        dim_he_list=args.dim_he_list,
    )
