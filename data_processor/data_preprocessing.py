import re
import os
import sys
import pickle
import numpy as np
import copy

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

    def scale_data(self, input_folder, save_folder, features_scale_list, scaler_save_folder, scaler_save_name):

        # intialize sk-learn preprocessing
        from sklearn import preprocessing


        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        file_save_path_list = [os.path.join(save_folder, x) for x in file_name_list]

        # read the feature name list
        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_name_list = f.readlines()[0].split(',')[::2]
                break
        #

        # read X
        X = []
        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_value_list = f.readlines()[0].split(',')[1::2]
                feature_value_list = [float(x) for x in feature_value_list]
                X.append(feature_value_list)
        X = np.array(X)
        #

        # scaling
        for features_scale in features_scale_list:
            s_feature_name_tuple = features_scale[0]
            scale_range = features_scale[1]
            f_name_index_list = [feature_name_list.index(f_name) for f_name in s_feature_name_tuple]
            scaler = preprocessing.MinMaxScaler(feature_range = scale_range)
            scaler.fit(X)
            trans_X = scaler.transform(X)
            for index in f_name_index_list:
                X[:, index] = trans_X[:, index]


            # ----------------------------------------------------------------------------------------------------------
            # save scaler
            # ----------------------------------------------------------------------------------------------------------
            scaler_name = scaler_save_name + '_' + str(scale_range[0]) + '_' + str(scale_range[1])
            scaler_save_path = os.path.join(scaler_save_folder, scaler_name)
            scaler_config_name = scaler_name + '_config.txt'
            scaler_config_save_path = os.path.join(scaler_save_folder, scaler_config_name)

            # (1.) save scaler pickle
            pickle.dump(scaler, open(scaler_save_path, "wb"))
            #

            # (2.) save scaler config
            with open(scaler_config_save_path, 'w', encoding = 'utf-8') as f:
                s_feature_name_str = ','.join(s_feature_name_tuple)
                scale_range = ','.join([str(x) for x in list(scale_range)])
                f.write(s_feature_name_str)
                f.write('\n')
                f.write(scale_range)
            #
            print ("save scaler to {} successfully!".format(scaler_save_path))
            print ("save scaler config to {} successfully!".format(scaler_config_save_path))
            # ----------------------------------------------------------------------------------------------------------
        #

        # save file
        for i, file_save_path in enumerate(file_save_path_list):
            with open(file_save_path, 'w', encoding = 'utf-8') as f:
                feature_write_list = [str(j) for i in zip(feature_name_list,X[i]) for j in i]
                feature_write_str = ','.join(feature_write_list)
                f.write(feature_write_str)

        print ("Scaling data succesful!")
        print ("features_scale_list: ", features_scale_list)


    def pred_scale_data(self, input_folder, save_folder, scaler_path_list):

        # intialize sk-learn preprocessing
        from sklearn import preprocessing


        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        file_save_path_list = [os.path.join(save_folder, x) for x in file_name_list]

        # read the feature name list
        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_name_list = f.readlines()[0].split(',')[::2]
                break
        #

        # read X
        X = []
        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_value_list = f.readlines()[0].split(',')[1::2]
                feature_value_list = [float(x) for x in feature_value_list]
                X.append(feature_value_list)
        X = np.array(X)
        #

        # transfrom the X by the trained scaler
        for scaler_path in scaler_path_list:
            scaler = pickle.load(open(scaler_path, "rb"))
            scaler_config_path = scaler_path + '_config.txt'
            with open(scaler_config_path, 'r', encoding ='utf-8') as f:
                file_list = f.readlines()
                s_feature_name_tuple = tuple(file_list[0].strip().split(','))
                scale_range = tuple(file_list[1].strip().split(','))
                f_name_index_list = [feature_name_list.index(f_name) for f_name in s_feature_name_tuple]
                trans_X = scaler.transform(X)
                for index in f_name_index_list:
                    X[:, index] = trans_X[:, index]
        #

        # save file
        for i, file_save_path in enumerate(file_save_path_list):
            with open(file_save_path, 'w', encoding = 'utf-8') as f:
                feature_write_list = [str(j) for i in zip(feature_name_list,X[i]) for j in i]
                feature_write_str = ','.join(feature_write_list)
                f.write(feature_write_str)

        print ("Scaling data for prediction succesful!")