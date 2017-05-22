# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import os
import datetime
import time
import tushare as ts
import collections
import re
import numpy as np
import urllib
import math
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path1 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from general_funcs import daterange, split_list_by_percentage
from pjslib.logger import logger1
# ==========================================================================================================





class Ashare:
    def __init__(self):
        self.a_share_samples_f_dict = collections.defaultdict(lambda :0)
        self.a_share_samples_t_dict = collections.defaultdict(lambda: 0)
        self.a_share_samples_dict = collections.defaultdict(lambda: 0)
        self.stock_set =  set(ts.get_stock_basics().to_dict()['profit'].keys())
        self.t_attributors = []
        self.f_attributors = []


    def read_tech_history_data(self, start_date, is_prediction = False):
        # clear
        self.a_share_samples_t_dict = collections.defaultdict(lambda: 0)
        #

        start_date_temp = time.strptime(start_date, '%Y-%m-%d')
        start_date_obj = datetime.datetime(*start_date_temp[:3]).date()
        today_obj = datetime.datetime.today().date()
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        t_attributors_set = set(ts.get_k_data("600883", start="2017-05-09", ktype='W').keys())
        t_attributors_set -= {'code', 'date'}
        if is_prediction is False:
            t_attributors_set.add('priceChange')
        t_attributors_set.add('candleLength')
        t_attributors_set.add('candlePos')
        t_attributors = sorted(list(t_attributors_set))
        self.t_attributors = t_attributors

        stock_list = list(self.stock_set)[:]
        is_close_price_exist = True

        for stock_id in stock_list:
            fund_dict = ts.get_k_data(stock_id, start=start_date, ktype='W').to_dict()
            # date_list: ['2017-05-05', '2017-05-12', '2017-05-19']
            try:
                date_items = list(fund_dict['date'].items())
            except KeyError:
                logger1.error("{} stock has no key data".format(stock_id))
                continue

            for i, (id, date_str) in enumerate(date_items):
                feature_list = []
                # skip the last date object, because the price change can not be calculated
                if i == len(date_items) - 1 and is_prediction is False:
                    print ("Skip {} because of date_str{}".format(id, date_str))
                    continue
                for attributor in t_attributors:
                    # for pricechange
                    if attributor == 'priceChange' and is_prediction is False:
                        close_price = fund_dict['close'][id]
                        close_price_next_week = fund_dict['close'][date_items[i+1][0]]
                        priceChange = "{:.5f}".format((close_price_next_week - close_price) / close_price)
                        feature_list.append(priceChange)
                        continue
                    elif attributor == 'candleLength':
                        close_price = fund_dict['close'][id]
                        open_price = fund_dict['open'][id]
                        high_price = fund_dict['high'][id]
                        low_price = fund_dict['low'][id]
                        candle_length = "{:.5f}".format(abs((close_price- open_price)/(high_price-low_price)))
                        feature_list.append(candle_length)
                        continue
                    elif attributor == 'candlePos':
                        close_price = fund_dict['close'][id]
                        open_price = fund_dict['open'][id]
                        high_price = fund_dict['high'][id]
                        low_price = fund_dict['low'][id]
                        price = max(close_price, open_price)
                        candle_pos = "{:.5f}".format(abs((high_price- price)/(high_price-low_price)))
                        feature_list.append(candle_pos)
                        continue

                    # for other attributors
                    feature_list.append(fund_dict[attributor][id])
                feature_array = np.array(feature_list)
                sample_name = date_str + '_' + stock_id
                self.a_share_samples_t_dict[sample_name] = feature_array
            print ("saving {} stock t features".format(stock_id))

        print ("t_attributors: {}".format(t_attributors))
        print ("a_share_samples_t_dict: {}".format(self.a_share_samples_t_dict.values()))
        print ("a_share_samples_t_dict_value: {}".format(list(self.a_share_samples_t_dict.values())[0]))


    def read_fundamental_data(self, start_date):
        # clear
        self.a_share_samples_f_dict = collections.defaultdict(lambda: 0)
        #
        start_date_temp = time.strptime(start_date, '%Y-%m-%d')
        start_date_obj = datetime.datetime(*start_date_temp[:3]).date()
        today_obj = datetime.datetime.today().date()
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        f_attributors_set = set(ts.get_stock_basics(date = "2017-05-09").to_dict().keys())
        filter_set = {'name', 'industry', 'area'}
        f_attributors_set = f_attributors_set - filter_set
        f_attributors = sorted(list(f_attributors_set))
        self.f_attributors = f_attributors

        for single_date in daterange(start_date_obj, today_obj):
            temp_stock_feature_dict = collections.defaultdict(lambda :[])

            # if it is not friday, skip!
            if single_date.weekday() != 4:
                continue
            date_str = single_date.strftime("%Y-%m-%d")

            try:
                ts_temp = ts.get_stock_basics(date = date_str)
                if ts_temp is None:
                    logger1.error("{} not found any data!".format(date_str))
                    continue
                fund_dict = ts_temp.to_dict()
            except urllib.error.HTTPError:
                logger1.error("{} not found any data!".format(date_str))
                continue



            for key, stock_key_value_dict in sorted(fund_dict.items()):

                # filter name,industry,,area,
                if key in filter_set:
                    continue
                #

                for stock_id, value in stock_key_value_dict.items():
                    temp_stock_feature_dict[stock_id].append((key, value))

            for stock_id, feature_list in temp_stock_feature_dict.items():
                feature_list = sorted(feature_list, key = lambda x: x[0])
                feature_value_list = [x[1] for x in feature_list]
                feature_array = np.array(feature_value_list)
                sample_name = date_str + '_' + stock_id

                # save samples
                self.a_share_samples_f_dict[sample_name] = feature_array
            print ("saving {}'s stock feature to a_share_samples_f_dict".format(single_date))

        print ("f_attributors: {}".format(f_attributors))
        print ("a_share_samples_f_dict_value: {}".format(list(self.a_share_samples_f_dict.values())[0]))


    def integrate_tech_fundamental_feature(self, feature1, feature2):
        new_feature = np.concatenate((feature1, feature2))
        return new_feature

    def save_raw_data(self, is_f = True, is_prediction = False):

        for sample, t_feature_array in self.a_share_samples_t_dict.items():
            feature_array_list = []
            # (0.) add technical features
            feature_array_list.append(t_feature_array)
            # (1.) add fundamental features
            if is_f:
                is_sample_exist = self.a_share_samples_f_dict.get(sample)
                if is_sample_exist is None:
                    continue
                f_feature_array = self.a_share_samples_f_dict[sample]
                feature_array_list.append(f_feature_array)

            # concatenate all features
            feature_array_final = np.array([])
            for feature_array in feature_array_list:
                feature_array_final = self.integrate_tech_fundamental_feature(feature_array_final, feature_array)

            # convert every feature to float
            feature_array_final = feature_array_final.astype(float)
            #
            feature_list_final = list(feature_array_final)

            attribitors = self.t_attributors + self.f_attributors

            if len(attribitors) != len(feature_list_final):
                logger1.error('sample: {}, feature_list_final and attribitors are not the same length! {}, {}'
                              .format(sample, len(attribitors), len(feature_list_final)))
                continue

            save_zip = zip(attribitors, feature_list_final)

            # save file
            save_name = sample + '.csv'
            if is_prediction == True:
                folder = 'pred_raw_data'
            else:
                folder = 'raw_data'
            save_path = os.path.join(folder, save_name)
            with open(save_path, 'w', encoding = 'utf-8') as f:
                for attribitor, feature_value in save_zip:
                    f.write(str(attribitor) + ',' + str(feature_value) + '\n')

    def label_raw_data(self, is_prediction = False):
        if is_prediction:
            folder = 'pred_raw_data'
        else:
            folder = 'raw_data'
        samples_list = []
        raw_data_file_name_list = os.listdir(folder)
        for raw_data_file_name in raw_data_file_name_list:
            sample_id = raw_data_file_name[0:-4]
            sample_feature_list = []
            sample_price_change = 0.0
            raw_data_file_path = os.path.join(folder, raw_data_file_name)
            with open(raw_data_file_path, 'r') as f:
                for line in f:
                    if line == '\n':
                        continue
                    line_list = line.split(',')
                    feature_name = line_list[0]
                    feature = float(line_list[1])
                    if feature_name == 'priceChange':
                        sample_price_change = float(feature)
                        continue
                    sample_feature_list.append(feature_name)
                    sample_feature_list.append(feature)
            samples_list.append([sample_id, sample_feature_list, sample_price_change])

        # sort by pricechange
        samples_list = sorted(samples_list, key = lambda x:x[2], reverse = True)
        neg_samples_list = [x for x in samples_list if x[2] < 0]
        pos_samples_list = [x for x in samples_list if x[2] >= 0]
        per_tuple = (0.05, 0.3, 1)
        pos_label_tuple = ('top','good','pos')
        neg_label_tuple = ('bottom', 'bad', 'neg')
        pos_samples_split_list = split_list_by_percentage(per_tuple, pos_samples_list)
        neg_samples_split_list = split_list_by_percentage(per_tuple, neg_samples_list)

        # label postive the data and output
        for i, small_pos_samples_list in enumerate(pos_samples_split_list):
            label = pos_label_tuple[i]
            for pos_sample in small_pos_samples_list:
                pos_sample[2] = label

        # label negative the data and output
        for i, small_neg_samples_list in enumerate(neg_samples_split_list):
            label = neg_label_tuple[i]
            for pos_sample in small_neg_samples_list:
                pos_sample[2] = label

        print (neg_samples_list)

        # save labeled data to local
        samples_list = pos_samples_list + neg_samples_list
        if is_prediction == True:
            folder = 'pred_labeled_data'
        else:
            folder = 'labeled_data'
        for sample_list in samples_list:
            file_name = sample_list[0] + '_' +sample_list[2] + '.txt'
            file_path = os.path.join(folder, file_name)
            feature_list = sample_list[1]
            feature_list = [str(x) for x in feature_list]
            feature_str = ','.join(feature_list)
            with open (file_path, 'w', encoding = 'utf-8') as f:
                f.write(feature_str)



        # print (pos_samples_split_list)
        # print (len(pos_samples_split_list))

    def get_stocks_feature_this_week(self):
        nearest_friday = datetime.datetime.today().date()
        delta = datetime.timedelta(days=1)
        while nearest_friday.weekday() != 4:
            nearest_friday -= delta

        start_date = nearest_friday.strftime("%Y-%m-%d")

        self.read_fundamental_data(start_date = start_date)
        self.read_tech_history_data(start_date = start_date, is_prediction = True)
        self.save_raw_data(is_prediction = True)

    def feature_engineering(self, input_folder , save_folder):

        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, file_name) for file_name in file_name_list]

        successful_save_count = 0
        original_data_count = len(file_name_list)

        for i, file_path in enumerate(file_path_list):
            file_name = file_name_list[i]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            stock_id = re.findall(r'_([0-9]+).csv', file_name)[0]
            # find the data of the previous friday
            date_obj_temp = time.strptime(date, '%Y-%m-%d')
            date_obj = datetime.datetime(*date_obj_temp[:3])

            previous_friday_obj = date_obj - datetime.timedelta(days = 7)
            previous_friday_str = previous_friday_obj.strftime("%Y-%m-%d")
            previous_friday_full_path = previous_friday_str + '_' + stock_id + '.csv'
            previous_friday_full_path = os.path.join(input_folder, previous_friday_full_path)


            try:
                with open (previous_friday_full_path, 'r', encoding = 'utf-8') as f:
                    previous_f_feature_pair_dict = {}
                    for line in f:
                        line_list = line.split(',')
                        feature_name = line_list[0]
                        feature_value = float(line_list[1].strip())
                        previous_f_feature_pair_dict[feature_name] = feature_value
            except FileNotFoundError:
                logger1.error("{} cannot find the previous friday data".format(file_name))
                continue


            feature_pair_dict = {}
            with open(file_path, 'r', encoding = 'utf-8') as f:
                for line in f:
                    line_list = line.split(',')
                    feature_name = line_list[0]
                    feature_value = float(line_list[1].strip())
                    feature_pair_dict[feature_name] = feature_value

            # ===================================================================================
            # add features
            # ===================================================================================
            # (1.) open change
            pre_f = previous_f_feature_pair_dict['open']
            f = feature_pair_dict['open']
            feature_pair_dict['openChange'] = "{:5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (2.) close change
            pre_f = previous_f_feature_pair_dict['close']
            f = feature_pair_dict['close']
            feature_pair_dict['closeChange'] = "{:5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (3.) high change
            pre_f = previous_f_feature_pair_dict['high']
            f = feature_pair_dict['high']
            feature_pair_dict['highChange'] = "{:5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (4.) low change
            pre_f = previous_f_feature_pair_dict['low']
            f = feature_pair_dict['low']
            feature_pair_dict['lowChange'] = "{:5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (5.) volume change
            pre_f = previous_f_feature_pair_dict['volume']
            f = feature_pair_dict['volume']
            feature_pair_dict['volumeChange'] = "{:5f}".format((f - pre_f) / pre_f)
            # ===================================================================================

            # ===================================================================================
            # delete features: close, high, low, open
            # ===================================================================================
            delete_features_set = {'close', 'high', 'low', 'open'}
            for feature_name in delete_features_set:
                feature_pair_dict.pop(feature_name)
            # ===================================================================================

            # write the feature engineered file to folder
            save_file_path = os.path.join(save_folder, file_name)
            with open(save_file_path, 'w', encoding = 'utf-8') as f:
                feature_pair_list = []
                feature_pair_tuple_list = sorted(list(feature_pair_dict.items()), key = lambda x:x[0])
                for feature_pair in feature_pair_tuple_list:
                    feature_pair_list.append(feature_pair[0])
                    feature_pair_list.append(feature_pair[1])

                feature_pair_list = [str(x) for x in feature_pair_list]
                feature_pair_str = ','.join(feature_pair_list)

                f.write(feature_pair_str)
                successful_save_count += 1
        print ("Succesfully engineered {} raw data! original count: {}, delete {} files"
               .format(successful_save_count, original_data_count, original_data_count - successful_save_count))