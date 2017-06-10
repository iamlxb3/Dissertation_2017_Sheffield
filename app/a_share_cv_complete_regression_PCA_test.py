# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# The complete CV test for a share MLP regressor
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import itertools
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers','mlp')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_classifier import MlpClassifier
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
# (1.) build classifer
mlp_regressor1 = MlpClassifier()

# # TODO ADD SOME REGRESSION TESTING DATA
# data_folder = os.path.join('a_share','a_share_regression_PCA_data')
# data_folder = os.path.join(parent_folder, 'data', data_folder)
# #



# a share data
data_folder = os.path.join('a_share','a_share_regression_PCA_data')
data_folder = os.path.join(parent_folder, 'data', data_folder)
#

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# FEATURE AND TOPOLOGY TEST
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# config the other parameters
# ----------------------------------------------------------------------------------------------------------------------
other_config_dict = {}
# (1.) learning_rate
other_config_dict['date_per'] = 1.0
other_config_dict['dev_per'] = 0.2
other_config_dict['learning_rate_init'] = 0.0001
other_config_dict['tol'] = 1e-6
include_top_list = [1, 3, 5, 7, 9]
print ("include_top_list: {}".format(include_top_list))
other_config_dict['include_top_list'] = include_top_list
other_config_dict['random_seed_list'] = [1,99,299]

# (2.) clf_path
clsfy_name = 'a_share_mlp_cv_PCA_regressor'
other_config_dict['clf_path'] = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# (3.) topology_result_path
cv_f_t_t_save_path_mres = os.path.join(parent_folder, 'topology_feature_test',
                                       'ashare_cv_regression_PCA_mres.txt')
cv_f_t_t_save_path_avg_pc = os.path.join(parent_folder, 'topology_feature_test',
                                         'ashare_cv_regression_PCA_avg_pc.txt')
cv_f_t_t_save_path_polar = os.path.join(parent_folder, 'topology_feature_test',
                                        'ashare_cv_regression_PCA_polar.txt')
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# decreasing the number of features for PCA 1 by 1
# ----------------------------------------------------------------------------------------------------------------------
# get_full_feature_switch_tuple, (1,1,1,1,1,1,....)
feature_switch_tuple_all_1 = mlp_regressor1.get_full_feature_switch_tuple(data_folder)
feature_switch_tuple_list = []
feature_switch_tuple_list.append(feature_switch_tuple_all_1)
for i in range(len(feature_switch_tuple_all_1)):
    if i == 0 or i == len(feature_switch_tuple_all_1):
        continue
    feature_switch_list = list(feature_switch_tuple_all_1[:])
    feature_switch_list[-i:] = [0 for x in range(i)]
    feature_switch_tuple = tuple(feature_switch_list)
    feature_switch_tuple_list.append(feature_switch_tuple)

# >>>>>>>>>DEBUG
#feature_switch_tuple_list = [feature_switch_tuple_list[1]]
#

print ("feature_switch_tuple_list: ", feature_switch_tuple_list)
# ----------------------------------------------------------------------------------------------------------------------

# TEST ONE FEATURE combination
#feature_switch_tuple_list = [(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
print("====================================================================")
print("PCA feature testing start!! Total feature combination: {}".format(len(feature_switch_tuple_list)))
print("====================================================================")


# testing the performance of classifier under different number of PCA features
for feature_switch_tuple in feature_switch_tuple_list:
    mlp_regressor1.read_selected_feature_list(data_folder, feature_switch_tuple)


    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # config hidden layer size
    # ----------------------------------------------------------------------------------------------------------------------
    hidden_layer_node_min = 20
    hidden_layer_node_max = 200
    hidden_layer_node_step = 1
    hidden_layer_depth_min = 3
    hidden_layer_depth_max = 12
    hidden_layer_node_min = 139
    hidden_layer_node_max = 139
    hidden_layer_node_step = 1
    hidden_layer_depth_min = 6
    hidden_layer_depth_max = 6
    hidden_layer_config_tuple = (hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min,
                                 hidden_layer_depth_max)
    # ----------------------------------------------------------------------------------------------------------------------
    print("====================================================================")
    print("Start testing for MLP's topology for regression!")
    print("====================================================================")

    # run topology test
    mlp_regressor1.cv_r_topology_test(data_folder, feature_switch_tuple, other_config_dict, hidden_layer_config_tuple)
    # ======================================================================================================================

mlp_regressor1.cv_r_save_feature_topology_result(cv_f_t_t_save_path_mres, key = 'mres')
mlp_regressor1.cv_r_save_feature_topology_result(cv_f_t_t_save_path_avg_pc, key = 'avg_pc')
mlp_regressor1.cv_r_save_feature_topology_result(cv_f_t_t_save_path_polar, key = 'polar')
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================

