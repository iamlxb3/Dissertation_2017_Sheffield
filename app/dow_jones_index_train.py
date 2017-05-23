# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Train and test the dow_jones_index data based on MLP
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
mlp1 = MlpClassifier()


# (2.) feed data
data_per = 1.0  # the percentage of data using for training and testing
dev_per = 0.1 # the percentage of data using for developing

# dow jones index dsata
data_folder = os.path.join('dow_jones_index','dow_jones_index_labeled')
data_folder = os.path.join(parent_folder, 'data', data_folder)
#

mlp1.feed_and_seperate_data(data_folder, dev_per = dev_per, data_per = data_per)
# ==========================================================================================================







# ======================================================================================================================
# test different paramters of MLP
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# config the other parameters
# ----------------------------------------------------------------------------------------------------------------------
other_config_dict = {}
# (1.) learning_rate
other_config_dict['learning_rate_init'] = 0.00001
# (2.) clf_path
clsfy_name = 'dow_jones_mlp'
other_config_dict['clf_path'] = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
# (3.) topology_result_path
other_config_dict['topology_result_path'] = os.path.join(parent_folder, 'topology_feature_test', 'dow_jones_index.txt')
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# config hidden layer size
# ----------------------------------------------------------------------------------------------------------------------
hidden_layer_node_min = 10
hidden_layer_node_max = 20
hidden_layer_node_step = 1
hidden_layer_depth_min = 1
hidden_layer_depth_max = 1
hidden_layer_config_tuple = (hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min,
                             hidden_layer_depth_max)
# ----------------------------------------------------------------------------------------------------------------------

# run topology test
mlp1.topology_test(other_config_dict, hidden_layer_config_tuple)
# ======================================================================================================================





# # ======================================================================================================================
# # normal run
# # ======================================================================================================================
# hidden_layer_sizes = (10,1)
# learning_rate_init = 0.00001
# # set mlp
# mlp1.set_mlp(hidden_layer_sizes, learning_rate_init = learning_rate_init)
#
# # (3.) train and test
# clsfy_name = 'dow_jones_mlp'
# mlp_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
# mlp1.count_label(data_folder)
# mlp1.train(save_clsfy_path= mlp_path)
# mlp1.dev(save_clsfy_path= mlp_path)
# # ======================================================================================================================