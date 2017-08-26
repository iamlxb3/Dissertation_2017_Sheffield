# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# The complete PCA CV test for a share MLP regressor
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================================================================================
# Build MLP classifier
# ==========================================================================================================
print ("Build MLP classifier for dow jones data with window shift!")

# (1.) build classifer
mlp_classifier1 = MlpTradeClassifier()
hidden_layer_sizes = (26,6)
learning_rate_init = 0.0001
print ("hidden_layer_sizes: ", hidden_layer_sizes)
mlp_classifier1.set_mlp_clf(hidden_layer_sizes, learning_rate_init = learning_rate_init)
clsfy_name = 'dow_jones_mlp_trade_classifier_window_shift'
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# (2.) data folder
data_folder = os.path.join('dow_jones_index_extended','dow_jones_index_extended_labeled')
data_folder = os.path.join(parent_folder, 'data', data_folder)
#

# (.) feature_switch
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(data_folder)

# (3.) feed data
data_per = 1.0  # the percentage of data using for training and testing
shifting_size = 26
shift_num = 5
training_window_size = 74


pca_n_component = read_pca_component(data_folder)
mode = 'clf'
mlp_classifier1.trade_feed_and_separate_data_window_shift(data_folder, data_per=data_per, shifting_size=shifting_size,
                                                          feature_switch_tuple=feature_switch_tuple_all_1,mode = mode,
                                                          training_window_size =training_window_size, shift_num = shift_num,
                                     is_standardisation = True, is_PCA = True, pca_n_component = pca_n_component)

# (4.) train window shift
random_seed = 'window_shift'
average_f1_list = []
accuracy_list = []
is_baseline = True
for shift in mlp_classifier1.validation_dict[random_seed].keys():
    mlp_classifier1.trade_rs_cv_load_train_dev_data(random_seed, shift)
    if is_baseline:
        average_f1, accuracy = mlp_classifier1.baseline_clf_dev(data_folder)
    else:
        mlp_classifier1.clf_train(save_clsfy_path=clf_path)
        average_f1, accuracy = mlp_classifier1.clf_dev(save_clsfy_path=clf_path, is_return = True)
    average_f1_list.append(average_f1)
    accuracy_list.append(accuracy)

print ("is baseline: {}".format(is_baseline))
print ("average f1 over window shifting: {}".format(np.average(average_f1_list)))
print ("average accuracy over window shifting: {}".format(np.average(accuracy_list)))
# # ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# BASELINE
# ----------------------------------------------------------------------------------------------------------
# average f1 over window shifting: 0.5282581617125446
# average accuracy over window shifting: 0.5566666666666668
# ----------------------------------------------------------------------------------------------------------
