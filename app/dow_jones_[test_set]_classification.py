# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor.
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
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
sys.path.append(clf_path)
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
print ("Build MLP classifier for dow_jones extended test data!")
mode = 'clf' #'reg'

# (1.) build classifer
mlp1 = MlpTradeClassifier()
clsfy_name = 'dow_jones_extened_test_mlp_classifier'
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# (2.) load training data, save standardisation_file and pca_file
data_per = 1.0 # the percentage of data using for training and testing
dev_per = 0.0 # the percentage of data using for developing
train_data_folder = os.path.join('dow_jones_index_extended','dow_jones_index_extended_labeled')
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
standardisation_file_path = os.path.join(parent_folder, 'data_processor','z_score')
pca_file_path = os.path.join(parent_folder,'data_processor','pca')
mlp1.trade_feed_and_separate_data(train_data_folder, dev_per = dev_per, data_per = data_per,
                                  standardisation_file_path = standardisation_file_path,
                                  pca_file_path = pca_file_path, mode = mode)

# (3.) load hyper parameters and training
# ------------------------------------------------------------------------------------------------------------
# hyper parameters
# ------------------------------------------------------------------------------------------------------------
verbose = True
hidden_layer_sizes = (33,3)
tol = 1e-8
learning_rate_init = 0.001
random_state = 10
learning_rate = 'constant'
early_stopping = False
activation  = 'relu'
validation_fraction  = 0.1 # The proportion of training data to set aside as validation set for early stopping.
                           # Must be between 0 and 1. Only used if early_stopping is True.
alpha  = 0.0001
# ------------------------------------------------------------------------------------------------------------
mlp1.set_mlp_clf(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init, random_state=random_state,
                           verbose = verbose, learning_rate = learning_rate, early_stopping =early_stopping,
                            activation  = activation, validation_fraction  = validation_fraction, alpha  = alpha)
mlp1.clf_train(save_clsfy_path= clf_path)
print ("Classifier for test trained successfully!")

# (4.) test
data_per = 1.0
dev_per = 1.0
test_data_folder = os.path.join('dow_jones_index_extended','dow_jones_index_extended_labeled_test')
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
mlp1.trade_feed_and_separate_data(test_data_folder, dev_per = dev_per, data_per = data_per,
                                  is_test_folder=True,
                                  standardisation_file_path = standardisation_file_path,
                                  pca_file_path=pca_file_path, mode = mode)
mlp1.clf_dev(save_clsfy_path= clf_path)
# ==========================================================================================================