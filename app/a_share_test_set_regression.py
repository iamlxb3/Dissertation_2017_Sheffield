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
from mlp_trade_regressor import MlpTradeRegressor
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
print ("Build MLP regressor for a-share data!")
# (1.) build classifer
mlp_regressor1 = MlpTradeRegressor()
clsfy_name = 'a_share_mlp_cv_PCA_regressor'
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)


# (3.) build standardisation and PCA and pickle the trained classifier
data_per = 1.0 # the percentage of data using for training and testing
dev_per = 0.0 # the percentage of data using for developing
train_data_folder = os.path.join('a_share','a_share_regression_data')
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
standardisation_file_path = os.path.join(parent_folder, 'data_processor','z_score')
pca_file_path = os.path.join(parent_folder,'data_processor','pca')
mlp_regressor1.trade_feed_and_separate_data(train_data_folder, dev_per = dev_per, data_per = data_per,
                                            standardisation_file_path = standardisation_file_path,
                                                            pca_file_path = pca_file_path)
hidden_layer_sizes = (33,3)
mlp_regressor1.set_regressor(hidden_layer_sizes)
mlp_regressor1.regressor_train(save_clsfy_path= clf_path)
print ("Test regressor trained successfully!")

# (4.) test
data_per = 1.0
dev_per = 1.0

test_data_folder = os.path.join('a_share','a_share_regression_data_test')
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
mlp_regressor1.trade_feed_and_separate_data(test_data_folder, dev_per = dev_per, data_per = data_per,
                                            is_test_folder=True,
                                            standardisation_file_path = standardisation_file_path,
                                            pca_file_path=pca_file_path)

mlp_regressor1.regressor_dev(save_clsfy_path= clf_path)


# ==========================================================================================================