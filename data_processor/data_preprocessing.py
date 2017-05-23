import re
import os
import sys
import numpy as np

class DataPp():
    def __init__(self):
        pass

    def examine_data(self, input_folder):
        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        for f_path in file_path_list:
            with open (f_path, 'r') as f:
                features_list  = f.readlines()[0].split(',')
                features_list = [x for i, x in enumerate(features_list) if i%2 != 0]
                for feature in features_list:
                    is_float_found = re.findall(r'[0-9]+', feature)
                    if is_float_found:
                        continue
                    else:
                        print ("f_path: {}".format(f_path))
                        print ("feature: {}".format(feature))

    def correct_non_float_feature(self, input_folder, save_folder):
        file_name_list = os.listdir(input_folder)
        original_data_count = len(file_name_list)
        succeful_data_count = 0
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            with open (f_path, 'r') as f:
                features_list_raw  = f.readlines()[0].split(',')
                features_value_list = [x for i, x in enumerate(features_list_raw) if i%2 != 0]
                # ( 1.) convert nan to 0.0
                for i, feature_value in enumerate(features_value_list):
                    feature_name = features_list_raw[2*i]
                    if feature_value == 'nan':
                        features_list_raw[2*i+1] = '0.0'
                        print ("{} found nan value for {} feature".format(f_name, feature_name))

            # # sort the feature by alphabet
            # features_name_list = features_list_raw[::2]
            # features_value_list = features_list_raw[1::2]
            # features_tuple_list = sorted(list(zip(features_name_list, features_value_list)), key = lambda x:x[0])
            # features_list = []
            # for feature_tuple in features_tuple_list:
            #     features_list.append(feature_tuple[0])
            #     features_list.append(feature_tuple[1])
            # #

            obj_f_path = os.path.join(save_folder, f_name)
            with open(obj_f_path, 'w', encoding = 'utf-8') as f:
                f.write(','.join(features_list_raw))
            succeful_data_count += 1

        print ("process all data succesful! Total: {} Delete: {}"
               .format(succeful_data_count, original_data_count-succeful_data_count))

    def fill_in_nan_data(self, input_folder, save_folder, mode = 'average'):
        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        features_value_list = []

        # find the sample with nan and accumulate valus for each feature
        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_value_list = f.readlines()[0].split(',')[1::2]
                feature_value_list = [float(x) if x != "nan" else x for x in feature_value_list]
                if 'nan' in feature_value_list:
                    pass
                else:
                    features_value_list.append(np.array(feature_value_list))

        # count the average value for each feature
        if mode == 'average':
            feature_average_value_list = list(np.average(features_value_list, axis = 0))

        sample_count = 0
        nan_count = 0
        # fill in the nan and write to new file
        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                line_list = f.readlines()[0].split(',')
                feature_value_list = line_list[1::2]
                if 'nan' in feature_value_list:
                    sample_count += 1
                    nan_indices_set_n = set([i for i, x in enumerate(feature_value_list) if x == "nan"])
                    for nan_index in nan_indices_set_n:
                        nan_count += 1
                        feature_value_list[nan_index] = str(feature_average_value_list[nan_index])

                    # write file
                    feature_name_list = line_list[::2]
                    feature_value_name_list = [j for i in zip(feature_name_list, feature_value_list) for j in i]
                    feature_value_name_str = ','.join(feature_value_name_list)

                else:
                    feature_value_name_str = ','.join(line_list)

                save_path = os.path.join(save_folder, file_name_list[i])
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(feature_value_name_str)

        print ("Fill in nan succesful! nan sample count: {}, nan count: {}".format(sample_count, nan_count))