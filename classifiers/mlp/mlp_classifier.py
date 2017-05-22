from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import collections
import re
import pickle
import math
import datetime
from pjslib.logger import logger2


class MlpClassifier:
    def __init__(self):
        self.mlp_hidden_layer_sizes_list = []
        self.training_set = []
        self.training_label = []
        self.dev_set = []
        self.dev_label = []
        self.test_set = []
        self.test_label = []

    def set_mlp(self, hidden_layer_sizes, tol = 1e-6, learning_rate_init = 0.001):
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=1,
                                     tol = tol, learning_rate_init = learning_rate_init, verbose = True)


    def _feed_data(self, folder, data_per):
        # TODO test the folder exists
        file_name_list = os.listdir(folder)
        file_path_list = [os.path.join(folder, x) for x in file_name_list]
        file_total_number = len(file_name_list)
        file_used_number = math.floor(data_per*file_total_number)  # restrict the number of training sample
        file_path_list = file_path_list[0:file_used_number]
        samples_feature_list = []
        samples_label_list = []
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            label = re.findall(r'[A-Za-z]+', f_name)[0]
            with open (f_path, 'r') as f:
                features_list  = f.readlines()[0].split(',')
                features_list = [x for i, x in enumerate(features_list) if i%2 != 0]
                features_list = [float(x) for x in features_list]
                features_array = np.array(features_list)
                samples_feature_list.append(features_array)
                samples_label_list.append(label)
        print ("read feature list and label list for {} successful!".format(folder))
        return samples_feature_list, samples_label_list


    def feed_and_seperate_data(self, folder, dev_per = 0.2, data_per = 1.0):

        # cut the number of training sample
        samples_feature_list, samples_label_list = self._feed_data(folder, data_per)

        #
        sample_number = math.floor(len(samples_feature_list))
        dev_sample_num = math.floor(sample_number*dev_per)
        print("dev_sample_num: ", dev_sample_num)
        dev_sample_num = dev_sample_num * -1
        self.training_set = samples_feature_list[0:dev_sample_num]
        self.training_label = samples_label_list[0:dev_sample_num]
        self.dev_set = samples_feature_list[dev_sample_num:]
        self.dev_label = samples_label_list[dev_sample_num:]


    def train(self, save_clsfy_path ="mlp_classifier"):
        print ("self.training_set_size: ", len(self.training_set))
        print ("self.training_label_size: ", len(self.training_label))
        self.mlp_clf.fit(self.training_set, self.training_label)
        pickle.dump(self.mlp_clf, open(save_clsfy_path, "wb"))

    def dev(self, save_clsfy_path ="mlp_classifier"):
        mlp = pickle.load(open(save_clsfy_path, "rb"))
        pred_label_list = []
        for feature_array in self.dev_set:
            feature_array = feature_array.reshape(1, -1)
            pred_label = mlp.predict(feature_array)[0]
            pred_label_list.append(pred_label)

        correct = 0
        for i, pred_label in enumerate(pred_label_list):
            if pred_label == self.dev_label[i]:
                correct += 1
        accuracy = correct/len(self.dev_label)
        print ("pred_label_list: ", pred_label_list)
        print ("accuracy: ", accuracy)


    def weekly_predict(self):
        mlp = pickle.load(open("mlp_classifier", "rb"))
        # read feature txt for check
        standard_feature = []
        with open('features.txt', 'r') as f:
            for line in f:
                standard_feature.append(line.strip())
        #

        folder = "pred_raw_data"
        nearest_friday = datetime.datetime.today().date()
        delta = datetime.timedelta(days=1)
        while nearest_friday.weekday() != 4:
            nearest_friday -= delta
        nearest_friday_str = nearest_friday.strftime("%Y-%m-%d")
        file_name_list = os.listdir(folder)
        prediction_set = []

        for file_name in file_name_list:
            is_file_valid = True
            stock_id = re.findall(r'_([0-9]+).csv', file_name)[0]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            if date != nearest_friday_str:
                print ("{} is out dated!".format(file_name))
                continue
            file_path = os.path.join(folder, file_name)
            feature_tuple_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    line_list = line.split(',')
                    feature_tuple_list.append(tuple(line_list))

            # =================================================================================================
            # check features are complete
            # =================================================================================================
            # print ("feature_tuple_list: ", feature_tuple_list)
            # print ("standard_feature: ", standard_feature)
            if len(feature_tuple_list) != len(standard_feature):
                logger2.error("{} file has more or less features!".format(file_name))
                continue

            for i, (feature_name, _) in enumerate(feature_tuple_list):
                if feature_name == standard_feature[i]:
                    pass
                else:
                    logger2.error("{} file has missing or wrong feature!".format(file_name))
                    is_file_valid = False
                    break
            if not is_file_valid:
                continue
            else:
                feature_tuple_list = [float(x[1]) for x in feature_tuple_list]
            # =================================================================================================

            # =================================================================================================
            # construct features_set and predict
            # =================================================================================================
            feature_array = np.array(feature_tuple_list).reshape(1,-1)
            pred_label = mlp.predict(feature_array)[0]
            label_proba = mlp.predict_log_proba(feature_array)
            prob = 0.0
            prediction_set.append((stock_id, pred_label, prob))
            # =================================================================================================

        # manual debug



        # sort by label
        final_prediction_set = []
        is_prediction_set_not_found = True
        print ("prediction_set_size: ", len(prediction_set))
        #
        while is_prediction_set_not_found:
            prediction_set_top = [x for x in prediction_set if x[1] == 'top']
            if not prediction_set_top:
                prediction_set_good = [x for x in prediction_set if x[1] == 'good']
            else:
                final_prediction_set = prediction_set_top
                is_prediction_set_not_found = False
                break

            if not prediction_set_good:
                prediction_set_pos = [x for x in prediction_set if x[1] == 'pos']
            else:
                final_prediction_set = prediction_set_good
                is_prediction_set_not_found = False
                break

            if prediction_set_top == [] and prediction_set_good == []:
                final_prediction_set = prediction_set_pos
                is_prediction_set_not_found = False
        #

        prediction_set = sorted(final_prediction_set, key = lambda x:x[2], reverse = True)

        # write prediction result
        with open('prediction.txt', 'w') as f:
            for prediction_tuple in prediction_set:
                f.write(str(prediction_tuple)+ '\n')
