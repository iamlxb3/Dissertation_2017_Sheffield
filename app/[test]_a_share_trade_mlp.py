# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP classifier.
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
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_classifier import MlpTradeClassifier
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
print ("Testing MLP trade regressor and classifier!!")
# (1.) build classifer
mlp_classifier1 = MlpTradeClassifier()
hidden_layer_sizes = (20,5)
learning_rate_init = 0.0001
print ("hidden_layer_sizes: ", hidden_layer_sizes)
mlp_classifier1.set_mlp_clf(hidden_layer_sizes, learning_rate_init = learning_rate_init)


# (2.) feed data
data_per = 1.0  # the percentage of data using for training and testing
dev_per = 0.1 # the percentage of data using for developing
data_folder = os.path.join('test','gaussian_data')
data_folder = os.path.join(parent_folder, 'data', data_folder)
mlp_classifier1.general_feed_and_separate_data_1_fold(data_folder, dev_per = dev_per, data_per = data_per, mode ='clf')


# (3.) train and test
clsfy_name = 'a_share_mlp_general_classifier'
mlp_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
mlp_classifier1.clf_train(save_clsfy_path= mlp_path)
mlp_classifier1.clf_dev(save_clsfy_path= mlp_path)
# ==========================================================================================================