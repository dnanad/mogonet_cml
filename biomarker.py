""" Example for biomarker identification
"""

import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat
from utils import find_numFolders_maxNumFolders

if __name__ == "__main__":
    # "discovery_cohort"  # "PE_cfRNA"  # "disc_coho3"  # "PE_cfRNA_pre"  # "TEST_DATA"  # "PE_cfRNA"  # "PE_cfRNA_pre"  # "TEST_DATA_copy"  # "ROSMAP" #"BRCA
    data_folder = "cfrna_metabo_disc225(0.001_95%)_TriStrat2"  # "cfrna_disc225(0.001_95%)"  # "metabo_disc225"  # "trial"  # discovery_cohort"  # "PE_cfRNA"  # "discovery_cohort"  # "PB2_cfRNA"  # "PE_cfRNA_pre"  # "TEST_DATA"  # "PE_cfRNA"  # "TEST_DATA"  # "PE_cfRNA"  # "TEST_DATA"  # "ROSMAP"
    stratify = True
    CV = True
    n_splits = 5
    num_epoch_pretrain = 400
    num_epoch = 1200

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

    dt_string = (
        "20240408-013504"
        # "20240123-160739"  # cpm95_cfrna001 strat_CV 400_1200
        # "20240122-180033"  # cpm95_cfrna001_metabo2 strat_CV 400_1200
        # "20240119-105801"  # metabo strat_CV 400_1200
        # "20240118-195725"  # cfRNA start_CV 400_1200
        # "20240118-193318"  # LATEST strat_CV 400_1200
        # "20240117-140446"  # strat_CV 400_1200
        # "20240117-133600"  # no_strat CV
        # "20240112-190638"
        # "20230914-223605"  # PE_cfRNA_pre CV
        # "20230907-182642"  # disc_coho3 CV
        # "20230907-182151"  # disc_coho1 CV
        # "20230907-181823"  # disc_coho1 NO_CV
        # "20230907-004649"  # disc coho CV
        # "20230906-153927"  # discovery cohort CV
        # "20230906-152343" discovery chohort NO_CV
        # "20230817-143951"
        # "20230818-161111"
        # "20230817-145402"
        # "20230817-150345"  # "20230817-143951"  # 20230817-122038"  # "20230816-154623"
    )
    latest_model = os.path.join(exp, dt_string)

    view_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    # view_list = [1]
    num_view = len(view_list)
    topn = 40

    if (
        data_folder == "cfrna_metabo_disc225(0.001_95%)_TriStrat2"
    ):  # "cfrna_metabo_disc225":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == "metabo_disc225":  # "cfrna_metabo_disc225":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == "disc_coho1":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "disc_coho3":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "disc_coho":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "discovery_cohort":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "PE_cfRNA_pre":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "PE_cfRNA":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "TEST_DATA_copy":
        num_class = 2  # number of classes
        adj_parameter = 2  # number of neighbors for each node
        dim_he_list = [400, 400, 200]  # hidden dimensions for each view
    if data_folder == "TEST_DATA":
        num_class = 2  # number of classes
        adj_parameter = 2  # number of neighbors for each node
        dim_he_list = [400, 400, 200]  # hidden dimensions for each view
    if data_folder == "ROSMAP":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == "BRCA":
        num_class = 5
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

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
