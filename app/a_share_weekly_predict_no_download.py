# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Weekly predict the a share stocks
# (0.) download the latest data
# (1.) Use the constructed classifier and predict
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import datetime
import time
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_processor_path = os.path.join(parent_folder, 'data_processor')
data_generator_path = os.path.join(parent_folder, 'data_generator')
mlp_path = os.path.join(parent_folder, 'classifiers','mlp')
sys.path.append(mlp_path)
sys.path.append(data_processor_path)
sys.path.append(data_generator_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_classifier import MlpClassifier
from data_preprocessing import DataPp
from stock_pca import StockPca
from a_share import Ashare
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PR
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
# (1.) build classifer
mlp_predictor = MlpClassifier()


# (2.) predict and save result

# prediction_input_folder
prediction_input_folder = os.path.join('a_share','[pred]_a_share_prediction_data')
prediction_input_folder = os.path.join(parent_folder, 'data', prediction_input_folder)

# classifier_path
classifier_name = 'a_share_mlp_regressor_v1.0'
classifier_path = os.path.join(parent_folder, 'trained_classifiers', classifier_name)

# prediction result save path
result_name = datetime.datetime.today().date().strftime("%Y-%m-%d")
result_name = 'a_share_' + result_name + '_prediction.txt'
prediction_save_path = os.path.join(parent_folder, 'prediction', 'a_share', result_name)

mlp_predictor.weekly_predict(prediction_input_folder, classifier_path, prediction_save_path)
# ==========================================================================================================