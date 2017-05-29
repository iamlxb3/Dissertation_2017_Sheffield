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


# dow jones index dsata
data_folder = os.path.join('a_share','a_share_regression_data')
data_folder = os.path.join(parent_folder, 'data', data_folder)
#

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# FEATURE AND TOPOLOGY TEST
# ======================================================================================================================


# ======================================================================================================================
# test different paramters of MLP
# ======================================================================================================================


# get_full_feature_switch_tuple, (1,1,1,1,1,1,....)
feature_switch_tuple = mlp_regressor1.get_full_feature_switch_tuple(data_folder)
mlp_regressor1.read_selected_feature_list(data_folder, feature_switch_tuple)
#


# ----------------------------------------------------------------------------------------------------------------------
# config the other parameters
# ----------------------------------------------------------------------------------------------------------------------
other_config_dict = {}
# (1.) learning_rate
other_config_dict['learning_rate_init'] = 0.0001
other_config_dict['tol'] = 1e-8

# (2.) clf_path
clsfy_name = 'a_share_mlp_cv_regressor'
other_config_dict['clf_path'] = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# (3.) topology_result_path
cv_f_t_t_save_path_mres = os.path.join(parent_folder, 'topology_feature_test', 'ashare_cv_regression_mres.txt')
cv_f_t_t_save_path_avg_pc = os.path.join(parent_folder, 'topology_feature_test', 'ashare_cv_regression_avg_pc.txt')
cv_f_t_t_save_path_polar = os.path.join(parent_folder, 'topology_feature_test', 'ashare_cv_regression_polar.txt')

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# config hidden layer size
# ----------------------------------------------------------------------------------------------------------------------
hidden_layer_node_min = 1
hidden_layer_node_max = 2
hidden_layer_node_step = 1
hidden_layer_depth_min = 1
hidden_layer_depth_max = 1
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

