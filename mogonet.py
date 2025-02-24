import argparse
import json
import os
from datetime import datetime
import pickle
import pandas as pd
from train_test import train_test
from utils import save_model_dict, find_numFolders_maxNumFolders, plot_epoch_loss


def main(
    data_folder: str,
    stratify: bool,
    CV: bool,
    n_splits: int,
    num_epoch_pretrain: int,
    num_epoch: int,
    test_interval: int,
    lr_e_pretrain: float,
    lr_e: float,
    lr_c: float,
    num_class: int,
    adj_parameter: int,
    dim_he_list: list,
):
    print("Running MOGONET")
    print("Data folder: ", data_folder)
    print("Stratify: ", stratify)
    print("CV: ", CV)
    print("Number of splits: ", n_splits)
    print("Number of pretrain epochs: ", num_epoch_pretrain)
    print("Number of epochs: ", num_epoch)
    print("Test interval: ", test_interval)
    print("Learning rate for pretraining: ", lr_e_pretrain)
    print("Learning rate for encoder: ", lr_e)
    print("Learning rate for classifier: ", lr_c)
    print("Number of classes: ", num_class)
    print("Adjacency parameter: ", adj_parameter)
    print("Dimension list for hidden layers: ", dim_he_list)

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, "data", data_folder)

    if stratify:
        sample_type_path = os.path.join(data_folder_path, "strat")
    else:
        sample_type_path = os.path.join(data_folder_path, "no_strat")

    if CV:
        folder = os.path.join(sample_type_path, "CV")
        folder_path = os.path.join(rootpath, folder)
        model_folder = os.path.join(folder, "models")
        model_folder_path = os.path.join(rootpath, model_folder)
    else:
        folder = os.path.join(sample_type_path, "NO_CV")
        folder_path = os.path.join(rootpath, folder)
        model_folder = os.path.join(folder, "models")
        model_folder_path = os.path.join(rootpath, model_folder)

    exp_epoch = f"{num_epoch_pretrain}_{num_epoch}"
    exp = os.path.join(model_folder_path, exp_epoch)

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    latest_model = os.path.join(exp, dt_string)

    lr = f"{lr_e_pretrain}_{lr_e}_{lr_c}"
    fig_file_name = f"{exp_epoch}_{lr}_{dt_string}_epoch_loss.png"

    view_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    num_view = len(view_list)

    if CV:
        scores_df_main = pd.DataFrame()
        for i in range(n_splits):
            cv_folder = os.path.join(folder_path, f"CV_{i}")
            cv_model_path = os.path.join(latest_model, f"CV_{i}")
            model_dict, epoch_loss_dict, test_info, scores_df = train_test(
                cv_folder,
                view_list,
                num_class,
                adj_parameter,
                dim_he_list,
                lr_e_pretrain,
                lr_e,
                lr_c,
                num_epoch_pretrain,
                num_epoch,
                test_interval,
                cv_model_path,
            )
            scores_df_main = pd.concat([scores_df_main, scores_df], ignore_index=True)

            if not os.path.exists(cv_model_path):
                os.makedirs(cv_model_path)
            fig_path = os.path.join(cv_model_path, fig_file_name)
            save_model_dict(cv_model_path, model_dict)
            plot_epoch_loss(epoch_loss_dict, fig_path)
            with open(os.path.join(cv_model_path, "test_info.pickle"), "wb") as handle:
                pickle.dump(test_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        scores_df_main["model"] = "mogonet"
        score_filename = f"mogonet_{exp_epoch}_{dt_string}_detail_scores.csv"
        scores_df_main.to_csv(os.path.join(latest_model, score_filename))
    else:
        model_dict, epoch_loss_dict, test_info, scores_df = train_test(
            folder_path,
            view_list,
            num_class,
            adj_parameter,
            dim_he_list,
            lr_e_pretrain,
            lr_e,
            lr_c,
            num_epoch_pretrain,
            num_epoch,
            test_interval,
            latest_model,
        )

        fig_path = os.path.join(latest_model, fig_file_name)
        save_model_dict(latest_model, model_dict)
        plot_epoch_loss(epoch_loss_dict, fig_path)
        with open(os.path.join(latest_model, "test_info.pickle"), "wb") as handle:
            pickle.dump(test_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description="MOGONET")
    parser.add_argument("--data_folder", type=str, required=True, help="Data folder")
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
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits")
    parser.add_argument(
        "--num_epoch_pretrain",
        type=int,
        required=True,
        help="Number of pretrain epochs",
    )
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs")
    parser.add_argument(
        "--test_interval", type=int, required=True, help="Test interval"
    )
    parser.add_argument(
        "--lr_e_pretrain",
        type=float,
        required=True,
        help="Learning rate for pretraining",
    )
    parser.add_argument(
        "--lr_e", type=float, required=True, help="Learning rate for encoder"
    )
    parser.add_argument(
        "--lr_c", type=float, required=True, help="Learning rate for classifier"
    )
    parser.add_argument(
        "--num_class", type=int, required=True, help="Number of classes"
    )
    parser.add_argument(
        "--adj_parameter", type=int, required=True, help="Adjacency parameter"
    )
    parser.add_argument(
        "--dim_he_list",
        type=str,
        required=True,
        help="Dimension list for hidden layers",
    )

    args = parser.parse_args()
    args.dim_he_list = json.loads(args.dim_he_list)
    return args


if __name__ == "__main__":
    args = parse_args()
    # print(args)

    main(
        data_folder=args.data_folder,
        stratify=args.stratify,
        CV=args.CV,
        n_splits=args.n_splits,
        num_epoch_pretrain=args.num_epoch_pretrain,
        num_epoch=args.num_epoch,
        test_interval=args.test_interval,
        lr_e_pretrain=args.lr_e_pretrain,
        lr_e=args.lr_e,
        lr_c=args.lr_c,
        num_class=args.num_class,
        adj_parameter=args.adj_parameter,
        dim_he_list=args.dim_he_list,
    )
# python mogonet.py --data_folder "0_new_data_0.05" \
#                 --stratify True \
#                 --CV True \
#                 --n_splits 5 \
#                 --num_epoch_pretrain 400 \
#                 --num_epoch 1200 \
#                 --test_interval 50 \
#                 --lr_e_pretrain 1e-3 \
#                 --lr_e 5e-3 \
#                 --lr_c 1e-3 \
#                 --num_class 2 \
#                 --adj_parameter 2 \
#                 --dim_he_list [200, 200, 100]
