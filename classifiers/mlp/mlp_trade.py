# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import collections
import datetime
import time
import pickle
import os
import re
import numpy as np
# ==========================================================================================================


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================

# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_general import MultilayerPerceptron
# ==========================================================================================================


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTrade(MultilayerPerceptron):

    def __init__(self):
        super().__init__()

        # create validation_dict, store all the cross validation data
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))


    def weekly_predict(self, input_folder, classifier_path, prediction_save_path):
        '''weekly predict could be based on regression or classification
        '''
        mlp = pickle.load(open(classifier_path, "rb"))
        file_name_list = os.listdir(input_folder)
        prediction_set = []

        # find the nearest date
        date_set = set()
        for file_name in file_name_list:
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            date_set.add(date)
        nearest_date = sorted(list(date_set), reverse=True)[0]
        nearest_date_temp = time.strptime(nearest_date, '%Y-%m-%d')
        nearest_date_obj = datetime.datetime(*nearest_date_temp[:3])

        print("==============================================================================")
        print("Prediction complete! This prediction is based on the data of DATE:[{}]".format(nearest_date))
        print("==============================================================================")

        if nearest_date_obj.weekday() != 4:
            print("WARNING! The nearest date for prediction is not friday!")

        # accumulate the prediction result
        for file_name in file_name_list:

            stock_id = re.findall(r'_([0-9]+).txt', file_name)[0]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            if date != nearest_date:
                continue
            file_path = os.path.join(input_folder, file_name)

            with open(file_path, 'r') as f:
                feature_value_list = f.readlines()[0].strip().split(',')[1::2]
                feature_value_list = [float(x) for x in feature_value_list]
                # =================================================================================================
                # construct features_set and predict
                # =================================================================================================
                feature_array = np.array(feature_value_list).reshape(1, -1)
                pred_value = float(mlp.predict(feature_array)[0])
                prediction_set.append((stock_id, pred_value))
                # =================================================================================================

        # write the prediciton result to file
        prediction_set = sorted(prediction_set, key=lambda x: x[1], reverse=True)
        with open(prediction_save_path, 'w', encoding='utf-8') as f:
            for stock_id, pred_value in prediction_set:
                f.write(stock_id + ' ' + str(pred_value) + '\n')
        #
        print("Prediction result save to {} successful!".format(prediction_save_path))
































