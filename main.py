import os
import argparse
import json
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute the entire workflow for multi-omic analysis"
    )
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
    return args


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(result.stderr)
    else:
        print(result.stdout)


def main():
    args = parse_args()

    # Preprocessing step
    preprocessing_command = f"""
    python3 preprocessing.py --data_folder "{args.data_folder}" \
                            --stratify {args.stratify} \
                            --strat_col_list '["Trimester", "fetal_near_miss"]' \
                            --test_size 0.2 \
                            --omic_normalize_dict '{{"1": true, "2": true}}' \
                            --da_omics_dict '{{"1": true, "2": true}}' \
                            --filter_wrt_dict '{{"1": "p-value(f-test)", "2": "p-value(f-test)"}}' \
                            --method_dict '{{"1": "bonferroni", "2": "bonferroni"}}' \
                            --da_threshold_dict '{{"1": 0.05, "2": 0.2}}' \
                            --CV {args.CV} \
                             --n_splits {args.n_splits}
    """
    run_command(preprocessing_command)

    # MOGONET step
    mogonet_command = f"""
    python3 mogonet.py --data_folder "{args.data_folder}" \
                    --stratify {args.stratify} \
                    --CV {args.CV} \
                    --n_splits {args.n_splits} \
                    --num_epoch_pretrain {args.num_epoch_pretrain} \
                    --num_epoch {args.num_epoch} \
                    --test_interval {args.test_interval} \
                    --lr_e_pretrain {args.lr_e_pretrain} \
                    --lr_e {args.lr_e} \
                    --lr_c {args.lr_c} \
                    --num_class {args.num_class} \
                    --adj_parameter {args.adj_parameter} \
                    --dim_he_list '{args.dim_he_list}'
    """
    run_command(mogonet_command)

    # CML step
    cml_models = ["XGBC", "DTC", "RFC", "SVC", "LRC", "ElasticNet"]
    for cml_model in cml_models:
        cml_command = f"""
        python3 cml.py --data_folder "{args.data_folder}" \
                    --cml_model {cml_model} \
                    --n_splits {args.n_splits} \
                    --CV {args.CV} \
                    --stratify {args.stratify} \
                    --no_test_mode
        """
        run_command(cml_command)

    # Result aggregation step
    result_aggregator_command = f"""
    python3 result_aggregator.py --data_folder "{args.data_folder}" \
                                --CV {args.CV} \
                                --stratify {args.stratify} \
                                --num_epoch_pretrain {args.num_epoch_pretrain} \
                                --num_epoch {args.num_epoch}
    """
    run_command(result_aggregator_command)


if __name__ == "__main__":
    main()
