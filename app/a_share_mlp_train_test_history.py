# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP classifier. Train the classifier based on a share history data and test on development set.
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
from mlp_classifier import MlpClassifier
# ==========================================================================================================






# ==========================================================================================================
# Training and developing
# ==========================================================================================================

# (1.) build classifer
mlp1 = MlpClassifier()
hidden_layer_sizes = (10, 1)
learning_rate_init = 0.001
mlp1.set_mlp(hidden_layer_sizes, learning_rate_init = learning_rate_init)

# (2.) feed data
data_per = 1.0  # the percentage of data using for training and testing
dev_per = 0.2 # the percentage of data using for developing
a_share_folder = os.path.join('a_share','a_share_processed_data')
a_share_folder = os.path.join(parent_folder, 'data', a_share_folder)
mlp1.feed_and_seperate_data(a_share_folder, dev_per = dev_per, data_per = data_per)

# (3.) train and test
clsfy_name = 'a_share_mlp'
mlp_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
mlp1.train(clsfy_path = mlp_path)
mlp1.dev(clsfy_path = mlp_path)
# ==========================================================================================================