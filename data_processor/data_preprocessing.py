import re
import os

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
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        total_f_c = len(file_path_list)
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

            obj_f_path = os.path.join(save_folder, f_name)
            with open(obj_f_path, 'w', encoding = 'utf-8') as f:
                f.write(','.join(features_list_raw))
        print ("process all data succesful! Total: {}".format(total_f_c))