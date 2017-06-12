# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# TEST
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers','mlp')
path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import get_full_feature_switch_tuple
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ----------------------------------------------------------------------------------------------------------------------
# (1.) build classifer and read data
# ----------------------------------------------------------------------------------------------------------------------
mlp_general_clf = MlpTradeClassifier()
data_folder = os.path.join('test','gaussian_data')
data_folder = os.path.join(parent_folder, 'data', data_folder)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (2.) config the other parameters
# ----------------------------------------------------------------------------------------------------------------------
other_config_dict = {}
# (a.) learning_rate
other_config_dict['data_per'] = 0.5
other_config_dict['dev_per'] = 0.2
other_config_dict['learning_rate_init'] = 0.0001
other_config_dict['tol'] = 1e-6
other_config_dict['random_seed_list'] = [1,99,299]

# (b.) clf_path
clsfy_name = 'a_share_general_MLP_tp_test'
other_config_dict['clf_path'] = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (3.) decreasing the number of features for PCA 1 by 1
# ----------------------------------------------------------------------------------------------------------------------
# get_full_feature_switch_tuple, (1,1,1,1,1,1,....)
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(data_folder)
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
feature_switch_tuple_list = [feature_switch_tuple_all_1]
#
print(" Total feature combination: {}".format(len(feature_switch_tuple_list)))
print ("feature_switch_tuple_list: ", feature_switch_tuple_list)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (4.) topology and feature selection test
# ----------------------------------------------------------------------------------------------------------------------
for feature_switch_tuple in feature_switch_tuple_list:
    mlp_general_clf.read_selected_feature_list(data_folder, feature_switch_tuple)

    # ------------------------------------------------------------------------------------------------------------------
    # config hidden layer size
    # ------------------------------------------------------------------------------------------------------------------
    hidden_layer_node_min = 1
    hidden_layer_node_max = 10
    hidden_layer_node_step = 1
    hidden_layer_depth_min = 1
    hidden_layer_depth_max = 3

    hidden_layer_config_tuple = (hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min,
                                 hidden_layer_depth_max)
    # ------------------------------------------------------------------------------------------------------------------
    print("====================================================================")
    print("Start testing for MLP general!")
    print("====================================================================")

    # run topology test
    mlp_general_clf.general_cv_cls_topology_test(data_folder, feature_switch_tuple, other_config_dict, hidden_layer_config_tuple)
# ----------------------------------------------------------------------------------------------------------------------
