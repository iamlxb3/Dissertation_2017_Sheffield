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
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================================================================================
# [0.] Delete all the files in the prediction folder
# ==========================================================================================================
a_share1 = Ashare()

prediction_folder_set = {'[pred]_a_share_f_engineered_data', '[pred]_a_share_prediction_data',
                         '[pred]_a_share_processed_data', '[pred]_a_share_raw_data', '[pred]_a_share_scaled_data'}
prediction_folder_list = [os.path.join(parent_folder, 'data', 'a_share', x) for x in prediction_folder_set]

a_share1.delete_all_prediction_folder(prediction_folder_list)
# ==========================================================================================================


# ==========================================================================================================
# [1.] Download the raw data for prediction
# ==========================================================================================================
a_share1 = Ashare()
today_obj = datetime.datetime.today().date()
days_ago_obj = today_obj - datetime.timedelta(days=15)
days_ago_str = days_ago_obj.strftime("%Y-%m-%d")
weekend_set = {5,6}

if not today_obj.weekday() in weekend_set:
    print ("WARNING! Today is not weekend!")
# get the data of up to five days ago

start_date = days_ago_str
print ("Data from {} will be downloaded.".format(start_date))
save_folder = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_raw_data')
a_share1.read_a_share_history_date(save_folder, start_date = start_date, is_prediction = True)
# ==========================================================================================================



# ==========================================================================================================
# [2.] Manually add features to a-share data.
# ==========================================================================================================
a_share1 = Ashare()
input_folder = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_raw_data')
save_folder = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_f_engineered_data')
a_share1.feature_engineering(input_folder , save_folder)
# ==========================================================================================================


# ==========================================================================================================
# [3.] Clean data. <a> get rid of nan feature value.
# ==========================================================================================================
data_cleaner = DataPp()
input_folder = '[pred]_a_share_f_engineered_data'
input_folder = os.path.join(parent_folder, 'data', 'a_share', input_folder)
save_folder = '[pred]_a_share_processed_data'
save_folder = os.path.join(parent_folder, 'data', 'a_share', save_folder)
data_cleaner.correct_non_float_feature(input_folder, save_folder)
#data_cleaner.examine_data(input_folder)  # examine the feature to see whether it is float

# ==========================================================================================================



# # ==========================================================================================================
# # [4.b] scaling
# # ==========================================================================================================
data_cleaner = DataPp()
input_folder = '[pred]_a_share_processed_data'
input_folder = os.path.join(parent_folder, 'data', 'a_share', input_folder)
save_folder = '[pred]_a_share_scaled_data'
save_folder = os.path.join(parent_folder, 'data', 'a_share', save_folder)
features_scale_list = []


# pickle load and scale the prediction data
trained_classifiers_folder = os.path.join(parent_folder, 'trained_data_processor')
scaler1_name = 'a_share_scaler_0_1'
scaler1_path = os.path.join(trained_classifiers_folder, scaler1_name)
scaler_path_list = []
scaler_path_list.append(scaler1_path)

data_cleaner.pred_scale_data(input_folder, save_folder, scaler_path_list)

# # ============================================================================================================




# # ==========================================================================================================
# # [4.3] predicion final transfrom, delete the 'priceChange'
# # ==========================================================================================================
a_share1 = Ashare()
input_folder = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_scaled_data')
save_folder = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_prediction_data')
a_share1.prediction_transfrom(input_folder, save_folder)
# # ==========================================================================================================





# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PREDICT PR
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
# (1.) build classifer
mlp_predictor = MlpClassifier()


# prediction_input_folder
prediction_input_folder = os.path.join('a_share','[pred]_a_share_prediction_data')
prediction_input_folder = os.path.join(parent_folder, 'data', prediction_input_folder)

# classifier_path
classifier_name = 'a_share_mlp_regressor'
classifier_path = os.path.join(parent_folder, 'trained_classifiers', classifier_name)

# prediction result save path
result_name = datetime.datetime.today().date().strftime("%Y-%m-%d")
result_name = 'a_share_' + result_name + '_prediction.txt'
prediction_save_path = os.path.join(parent_folder, 'prediction', 'a_share', result_name)

mlp_predictor.weekly_predict(prediction_input_folder, classifier_path, prediction_save_path)
# ==========================================================================================================