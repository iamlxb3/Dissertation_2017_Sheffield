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
import sys
import math
import random
import numpy as np

# ==========================================================================================================


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path1 = os.path.join(parent_folder, 'general_functions')
path2 = os.path.join(parent_folder, 'strategy')
sys.path.append(path1)
sys.path.append(path2)
# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_general import MultilayerPerceptron
from trade_general_funcs import feature_degradation
from trade_general_funcs import list_by_index


# ==========================================================================================================


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTrade(MultilayerPerceptron):
    def __init__(self):
        super().__init__()

        # create validation_dict, store all the cross validation data
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # --------------------------------------------------------------------------------------------------------------
        # container for 1-fold validation training and dev data for both classifier and regressor
        # --------------------------------------------------------------------------------------------------------------
        self.training_set = []
        self.training_value_set = []
        self.dev_set = []
        self.dev_value_set = []
        self.dev_date_set = []
        self.dev_stock_id_set = []
        # --------------------------------------------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------------------------
    # [C.1] read, feed and load data for 1 validation
    # ------------------------------------------------------------------------------------------------------------------
    def _feed_data(self, folder, data_per, feature_switch_tuple=None, is_random=False, random_seed=1, mode='reg'):
        # feature_switch_tuple : (0,1,1,0,1,1,0,...) ?
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)
        # ::: _feed_data :::
        # TODO test the folder exists
        file_name_list = os.listdir(folder)
        file_path_list = [os.path.join(folder, x) for x in file_name_list]
        file_total_number = len(file_name_list)
        file_used_number = math.floor(data_per * file_total_number)  # restrict the number of training sample
        file_path_list = file_path_list[0:file_used_number]
        samples_feature_list = []
        samples_value_list = []
        date_str_list = []
        stock_id_list = []
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            if mode == 'reg':
                regression_value = float(re.findall(r'#([0-9\.\+\-e]+)#', f_name)[0])
            elif mode == 'clf':
                regression_value = re.findall(r'_([A-Za-z\-0-9]+).txt', f_name)[0]
            else:
                print("Please enter the correct mode!")
                sys.exit()
            date_str = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', f_name)[0]
            stock_id = re.findall(r'_([A-Za-z0-9]{1,6})_', f_name)[0]
            with open(f_path, 'r') as f:
                features_list = f.readlines()[0].split(',')
                features_list = features_list[1::2]
                features_list = [float(x) for x in features_list]
                if feature_switch_tuple:
                    features_list = feature_degradation(features_list, feature_switch_tuple)
                features_array = np.array(features_list)
                # features_array = features_array.reshape(-1,1)
                samples_feature_list.append(features_array)
                samples_value_list.append(regression_value)
                date_str_list.append(date_str)
                stock_id_list.append(stock_id)
        print("read feature list and {}_value list for {} successful!".format(mode, folder))

        # random by random seed
        if is_random:
            print("Start shuffling the data...")
            import random
            combind_list = list(zip(samples_feature_list, samples_value_list, date_str_list, stock_id_list))
            random_seed = random_seed
            random.seed(random_seed)
            random.shuffle(combind_list)
            samples_feature_list, samples_value_list, date_str_list, stock_id_list = zip(*combind_list)
            print("Data set shuffling complete! Random Seed: {}".format(random_seed))
        #

        return samples_feature_list, samples_value_list, date_str_list, stock_id_list

    def load_train_dev_trade_data_for_1_validation(self, samples_feature_list, samples_value_list,
                                                   date_str_list, stock_id_list, dev_date_set, is_production = False):
        all_date_set = set(date_str_list)
        if is_production:
            training_date_set = all_date_set
        else:
            training_date_set = all_date_set - dev_date_set

        # get the dev index
        dev_index_list = []
        for j, date_str in enumerate(date_str_list):
            if date_str in dev_date_set:
                dev_index_list.append(j)
        #

        # get the training index
        training_index_list = []
        for k, date_str in enumerate(date_str_list):
            if date_str in training_date_set:
                training_index_list.append(k)
        #


        self.training_set = list_by_index(samples_feature_list, training_index_list)
        self.training_value_set = list_by_index(samples_value_list, training_index_list)
        self.dev_set = list_by_index(samples_feature_list, dev_index_list)
        self.dev_value_set = list_by_index(samples_value_list, dev_index_list)
        self.dev_date_set = list_by_index(date_str_list, dev_index_list)
        self.dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)

        print("Load train, dev data complete! Train size: {}, dev size: {}".
              format(len(self.training_value_set), len(self.dev_value_set)))
    # ------------------------------------------------------------------------------------------------------------------






    # ------------------------------------------------------------------------------------------------------------------
    # [C.2] read, feed and load data for cross validation and random seed
    # ------------------------------------------------------------------------------------------------------------------
    def trade_feed_and_separate_data(self, folder, dev_per=0.1, data_per=1.0, feature_switch_tuple=None,
                                     random_seed='normal', mode='reg', is_production = False,
                                     is_standardisation = True, is_PCA = True):
        '''feed and seperate data in the normal order
        '''
        # (1.) read all the data, feature customizable
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._feed_data(folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode=mode)

        # (2.) compute the dev part index
        dev_date_num = math.floor(len(set(date_str_list)) * dev_per)
        if dev_date_num == 0.0:
            dev_date_set = set()
            print ("WARNING!! dev_date_num = 0, data set or dev_per may be too small!")
        else:
            dev_date_set = set(sorted(list(set(date_str_list)))[-1 * dev_date_num:])
        print("dev_date_set: ", dev_date_set)

        # (3.) load_train_dev_data_for_1_validation
        self.load_train_dev_trade_data_for_1_validation(samples_feature_list, samples_value_list,
                                                        date_str_list, stock_id_list, dev_date_set,
                                                        is_production = is_production)

        # (4.) data pre_processing
        trans_fit, trans_obj = self.mlp_data_pre_processing(self.training_set, self.dev_set, is_standardisation
                                                            , is_PCA)
        self._update_train_dev_value_set(trans_fit, trans_obj)



    def create_train_dev_vdict_stock(self, samples_feature_list, samples_value_list,
                                     date_str_list, stock_id_list, date_random_subset_list, random_seed, is_cv=True,
                                     is_standardisation = True, is_PCA = True):
        # (0.) reset validation_dict
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # get the date set and count the number of unique dates

        for i, dev_date_set in enumerate(date_random_subset_list):
            all_date_set = set(date_str_list)
            training_date_set = all_date_set - dev_date_set

            # get the dev index
            dev_index_list = []
            for j, date_str in enumerate(date_str_list):
                if date_str in dev_date_set:
                    dev_index_list.append(j)
            #

            # get the training index
            training_index_list = []
            for k, date_str in enumerate(date_str_list):
                if date_str in training_date_set:
                    training_index_list.append(k)
            #

            training_set = list_by_index(samples_feature_list, training_index_list)
            training_value_set = list_by_index(samples_value_list, training_index_list)
            dev_set = list_by_index(samples_feature_list, dev_index_list)
            dev_value_set = list_by_index(samples_value_list, dev_index_list)
            dev_date_set = list_by_index(date_str_list, dev_index_list)
            dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)

            # data pre-processing
            # (.) standardisation, PCA
            training_set, dev_set = self.mlp_data_pre_processing(training_set, dev_set, is_standardisation
                                                                 , is_PCA)
            #

            self.validation_dict[random_seed][i]['training_set'] = training_set
            self.validation_dict[random_seed][i]['training_value_set'] = training_value_set
            self.validation_dict[random_seed][i]['dev_set'] = dev_set
            self.validation_dict[random_seed][i]['dev_value_set'] = dev_value_set
            self.validation_dict[random_seed][i]['dev_date_set'] = dev_date_set
            self.validation_dict[random_seed][i]['dev_stock_id_set'] = dev_stock_id_set

        validation_num = len(date_random_subset_list)
        print("Create validation_dict sucessfully! {}-fold cross validation".format(validation_num))

    def create_train_dev_vdict_window_shift(self, samples_feature_list, samples_value_list,
                               date_str_list, stock_id_list, is_cv=True, shifting_size_percent = 0.1, shift_num = 5,
                                            priority = 'training_set', is_standardisation = True, is_PCA = True,
                                            pca_n_component = None):

        # (0.) reset validation_dict
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # (.) window size
        random_seed = 'window_shift'
        sorted_date_list = sorted(list(set(date_str_list)))
        date_num = len(sorted_date_list)

        shifting_size_floor = math.floor(date_num * shifting_size_percent)
        shifting_size_ceil = math.ceil(date_num * shifting_size_percent)


        if priority == 'dev_set':
            shifting_size = shifting_size_ceil
        elif priority == 'training_set':
            shifting_size = shifting_size_floor

        if shifting_size*shift_num >= date_num:
            print ("TOO BIG shift_num or shifting_size!")
            sys.exit()


        window_size = date_num - shifting_size*shift_num

        if shifting_size*shift_num >= window_size:
            print("Training set too small!!")
            sys.exit()
        #

        print("date_num: ", date_num)
        print("shift_num: ", shift_num)
        print("window_size: ", window_size)
        print("shifting_size: ", shifting_size)

        for shift in range(shift_num):

            # (1.) get the training and dev date
            training_date_start_index = shift*shifting_size
            training_date_end_index = training_date_start_index + window_size
            dev_date_end_index = training_date_end_index + shifting_size
            if dev_date_end_index > len(sorted_date_list) - 1:
                print ("Error! dev_date_end_index exceed! Please check shift_num or shifting_size_percent!")
            training_date_list = sorted_date_list[training_date_start_index:training_date_end_index]
            dev_date_list = sorted_date_list[training_date_end_index:dev_date_end_index]
            print ("---------------------------------------------------------------------")
            print ("shift_index: {}".format(shift))
            print ("training_date_list: ", training_date_list)
            print ("dev_date_list: ", dev_date_list)

            #

            # (2.) get the dev index
            dev_index_list = []
            for j, date_str in enumerate(date_str_list):
                if date_str in dev_date_list:
                    dev_index_list.append(j)
            #

            # (3.) get the training index
            training_index_list = []
            for k, date_str in enumerate(date_str_list):
                if date_str in training_date_list:
                    training_index_list.append(k)
            #

            # (4.) load the training and dev data
            training_set = list_by_index(samples_feature_list, training_index_list)
            training_value_set = list_by_index(samples_value_list, training_index_list)
            dev_set = list_by_index(samples_feature_list, dev_index_list)
            dev_value_set = list_by_index(samples_value_list, dev_index_list)
            dev_date_set = list_by_index(date_str_list, dev_index_list)
            dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)


            # data pre-processing
            # (.) standardisation, PCA
            training_set, dev_set = self.mlp_data_pre_processing(training_set, dev_set, is_standardisation
                                                                 , is_PCA, pca_n_component = pca_n_component)
            #

            print ("Training_set_size: {}".format(len(training_set)))
            print ("Dev_set_size: {}".format(len(dev_set)))

            self.validation_dict[random_seed][shift]['training_set'] = training_set
            self.validation_dict[random_seed][shift]['training_value_set'] = training_value_set
            self.validation_dict[random_seed][shift]['dev_set'] = dev_set
            self.validation_dict[random_seed][shift]['dev_value_set'] = dev_value_set
            self.validation_dict[random_seed][shift]['dev_date_set'] = dev_date_set
            self.validation_dict[random_seed][shift]['dev_stock_id_set'] = dev_stock_id_set
            #

        print("Create window-shifting validation_dict sucessfully! {}-fold window shifting".format(shift_num))

    def trade_feed_and_separate_data_window_shift(self, folder, data_per=1.0, feature_switch_tuple=None,
                                                  shifting_size_percent = 0.1, shift_num = 5, mode = 'reg',
                                     is_standardisation = True, is_PCA = True, pca_n_component = None):
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._feed_data(folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode = mode)
        self.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                                                 date_str_list, stock_id_list, is_cv=True,
                                                 shifting_size_percent=shifting_size_percent,
                                                 shift_num=shift_num,
                                                 is_standardisation=is_standardisation, is_PCA=is_PCA,
                                                 pca_n_component=pca_n_component)



    def trade_rs_cv_load_train_dev_data(self, random_seed, cv_index):
        self.training_set = self.validation_dict[random_seed][cv_index]['training_set']
        self.training_value_set = self.validation_dict[random_seed][cv_index]['training_value_set']
        self.dev_set = self.validation_dict[random_seed][cv_index]['dev_set']
        self.dev_value_set = self.validation_dict[random_seed][cv_index]['dev_value_set']
        self.dev_date_set = self.validation_dict[random_seed][cv_index]['dev_date_set']
        self.dev_stock_id_set = self.validation_dict[random_seed][cv_index]['dev_stock_id_set']
    # ------------------------------------------------------------------------------------------------------------------

