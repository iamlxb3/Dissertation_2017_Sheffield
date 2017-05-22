import re
import os
import sys

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