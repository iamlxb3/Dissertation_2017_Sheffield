# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import random
import os
import itertools
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(clf_path)
sys.path.append(path2)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_ensemble_regressor import MlpTradeEnsembleRegressor
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component

# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# SETTINGS
# ==========================================================================================================
RANDOM_SEED_OFFSET = 54385438
EXPERIMENT_RANDOM_SEED_OFFSET = 38453845

data_set = 'dow_jones'
mode = 'adaboost' #adaboost, bagging
classifier = '{}_regressor'.format(mode)

ensemble_number = 3

# classifier = 'bagging_regressor'

EXPERIMENTS = 10
TRAILS = 100
random_state_total = 50
is_standardisation = True
is_PCA = True
tol = 1e-10
# ==========================================================================================================




# (1.) build classifer
mlp_regressor1 = MlpTradeEnsembleRegressor(ensemble_number, mode)



clsfy_name = 'dow_jone_hyper_parameter_{}'.format(classifier)
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)


# (2.) GET TRANINING SET
data_per = 1.0 # the percentage of data using for training and testing
dev_per = 0.0 # the percentage of data using for developing


input_folder = os.path.join('dow_jones_index','dow_jones_index_regression')
input_folder = os.path.join(parent_folder, 'data', input_folder)
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(input_folder)

# (3.)
samples_feature_list, samples_value_list, \
date_str_list, stock_id_list = mlp_regressor1._feed_data(input_folder, data_per=data_per,
                                               feature_switch_tuple=feature_switch_tuple_all_1,
                                               is_random=False)


# (4.)
if is_standardisation and is_PCA:
    data_preprocessing = 'pca_standardization'
elif is_standardisation and not is_PCA:
    data_preprocessing = 'standardization'
elif not is_standardisation and is_PCA:
    data_preprocessing = 'pca'
elif not is_standardisation and not is_PCA:
    data_preprocessing = 'origin'
else:
    print ("Check data preprocessing switch")
    sys.exit()
shift_num = 5
shifting_size_percent = 0.1
pca_n_component = read_pca_component(input_folder)


include_top_list = [1]

# ----------------------------------------------------------------------------------------------------------------------
# generator builder
# ----------------------------------------------------------------------------------------------------------------------
def build_generator_from_pool(random_pool, trails, experiment_count):
    for i in range(trails):
        random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
        random_sample = random.sample(random_pool, 1)[0]
        yield random_sample

def build_generator_from_range(target_range, trails, experiment_count):
    for i in range(trails):
        random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
        random_value = random.uniform(*target_range)
        yield random_value
# ----------------------------------------------------------------------------------------------------------------------


for experiment_count, experiment in enumerate(range(EXPERIMENTS)):


    # (1.) activation function
    activation_list = ['identity', 'logistic', 'tanh', 'relu']
    activation_random_sample_generator = build_generator_from_pool(activation_list, TRAILS, experiment_count)

    # (2.) L2 penalty (regularization term) parameter.
    alpha_range = (0.00001, 0.001)
    alpha_random_sample_generator = build_generator_from_range(alpha_range, TRAILS, experiment_count)

    # (3.) learning_rate
    learning_rate_list = ['constant', 'invscaling']
    learning_rate_random_sample_generator = build_generator_from_pool(learning_rate_list, TRAILS, experiment_count)

    # (4.) learning_rate_init
    learning_rate_init_range = (0.00001, 0.1)
    learning_rate_init_random_sample_generator = build_generator_from_range(learning_rate_init_range, TRAILS,
                                                                            experiment_count)

    # (5.) early-stopping
    early_stopping_list = [True, False]
    early_stopping_random_sample_generator = build_generator_from_pool(early_stopping_list, TRAILS, experiment_count)

    # (6.) early-stopping validation_fraction
    validation_fraction_range = (0.1, 0.3)
    validation_fraction_random_sample_generator = build_generator_from_range(validation_fraction_range, TRAILS,
                                                                             experiment_count)

    # (7.) HIDDEN LAYERS
    hidden_layer_depth = (1,3)
    hidden_layer_node  = (20,400)

    def hidden_layer_generator(hidden_layer_depth, hidden_layer_node, experiment_count):
        for i in range(TRAILS):
            hidden_layer_sizes = []
            random.seed(i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
            layer_depth = random.randint(*hidden_layer_depth)
            for j in range(layer_depth):
                random.seed(j + i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                layer_node = random.randint(*hidden_layer_node)
                hidden_layer_sizes.append(layer_node)
            hidden_layer_sizes_tuple = tuple(hidden_layer_sizes)
            yield hidden_layer_sizes_tuple


    hidden_layer_size= hidden_layer_generator(hidden_layer_depth, hidden_layer_node, experiment_count)



    hyper_parameter_trail_zip = zip(activation_random_sample_generator, alpha_random_sample_generator,
                                          learning_rate_random_sample_generator, learning_rate_init_random_sample_generator,
                                          early_stopping_random_sample_generator, validation_fraction_random_sample_generator,
                                    hidden_layer_size)

    # hyper_parameter_size = len(hyper_parameter_trail_list)
    # print ("hyper_parameter_size: ", hyper_parameter_size)
    # print ("hyper_parameter_trail_list: ", hyper_parameter_trail_list)



    for i, hyper_paramter_tuple in enumerate(hyper_parameter_trail_zip):

        # (0.) PCA n component
        if data_preprocessing == 'pca' or data_preprocessing == 'pca_standardization':
            random.seed(i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
            pca_n_component = random.randint(2, pca_n_component)
        else:
            pca_n_component = None

        mlp_regressor1.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                                                           date_str_list, stock_id_list, is_cv=True,
                                                           shifting_size_percent=shifting_size_percent,
                                                           shift_num=shift_num,
                                                           is_standardisation=is_standardisation, is_PCA=is_PCA,
                                                           pca_n_component=pca_n_component)




        activation_function = hyper_paramter_tuple[0]
        alpha = hyper_paramter_tuple[1]
        learning_rate = hyper_paramter_tuple[2]
        learning_rate_init = hyper_paramter_tuple[3]
        early_stopping = hyper_paramter_tuple[4]
        if early_stopping:
            validation_fraction = hyper_paramter_tuple[5]
        else:
            validation_fraction = 0.0
        hidden_layer_sizes = hyper_paramter_tuple[6]

        trail = i
        n_iter_list = []
        loss_list = []
        rmse_list = []
        avg_pc_list = []
        random_state_list = []
        polar_percent_list = []


        for random_state in range(random_state_total):
            random_seed = 'window_shift'

            shift_n_iter_list = []
            shift_loss_list = []
            shift_rmse_list = []
            shift_avg_pc_list = []
            shift_polar_percent_list = []

            mlp_regressor1.set_regressor(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init,
                                              random_state=random_state,
                               verbose = False, learning_rate = learning_rate, early_stopping =early_stopping, alpha= alpha,
                                              validation_fraction = validation_fraction, activation = activation_function)

            for shift in mlp_regressor1.validation_dict[random_seed].keys():
                mlp_regressor1.trade_rs_cv_load_train_dev_data(random_seed, shift)
                n_iter, loss = mlp_regressor1.regressor_train(save_clsfy_path=clf_path)
                mrse, avg_price_change_tuple, polar_percent = mlp_regressor1.regressor_dev(save_clsfy_path=clf_path, is_cv=True, include_top_list=include_top_list)
                shift_n_iter_list.append(n_iter)
                shift_loss_list.append(loss)
                shift_rmse_list.append(mrse)
                shift_avg_pc_list.append(avg_price_change_tuple[0])
                shift_polar_percent_list.append(polar_percent)


            avg_n_iter = np.average(shift_n_iter_list)
            avg_loss= np.average(shift_loss_list)
            avg_rmse = np.average(np.average(shift_rmse_list))
            avg_pc = np.average(shift_avg_pc_list)
            avg_polar_percent = np.average(np.average(shift_polar_percent_list))



            random_state_list.append(random_state)
            n_iter_list.append(avg_n_iter)
            loss_list.append(avg_loss)
            rmse_list.append(avg_rmse)
            avg_pc_list.append(avg_pc)
            polar_percent_list.append(avg_polar_percent)

            print ("-----------------------------------------------------------------------------------")
            print ("regressor_ensemble_number: ", ensemble_number)
            print ("pca_n_component: ", pca_n_component)
            print ("random_state: {}|{}".format(random_state, random_state_total))
            print ("activation_function: ", activation_function)
            print ("alpha: ", alpha)
            print ("learning_rate: ", learning_rate)
            print ("learning_rate_init: ", learning_rate_init)
            print ("early_stopping: ", early_stopping)
            print ("validation_fraction: ", validation_fraction)
            print ("hidden_layer_sizes: ", hidden_layer_sizes)
            print ("avg_n_iter: ", avg_n_iter)
            print ("avg_loss: ", avg_loss)
            print ("avg_rmse: ", avg_rmse)
            print ("avg_pc: ", avg_pc)
            print ("avg_polar_percent: ", avg_polar_percent)
            print ("Testing percent: {:.7f}%".format(100*i/TRAILS))
            print ("experiment: {}/{}".format(experiment, EXPERIMENTS))

        # ==========================================================================================================
        write_list = list(zip(loss_list, n_iter_list, rmse_list, avg_pc_list, polar_percent_list, random_state_list))
        write_list = sorted(write_list, key = lambda x:x[0])


        hidden_layer_write_str = '_'.join([str(x) for x in hidden_layer_sizes])

        write_tuple = (experiment, trail, random_state_total, pca_n_component, activation_function, alpha, learning_rate, learning_rate_init, early_stopping,
                       validation_fraction, hidden_layer_write_str)
        save_folder = os.path.join(parent_folder, 'hyper_parameter_test', data_set, classifier, data_preprocessing)
        csv_file_path = os.path.join(save_folder, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.
                                     format(ensemble_number, *write_tuple))
        txt_file_path = os.path.join(save_folder, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.
                                     format(ensemble_number, *write_tuple))



        with open(csv_file_path, 'w') as f:
            for i, tuple1 in enumerate(write_list):
                list1 = list(tuple1)
                list1 = [str(x) for x in list1]
                tuple_str = ','.join(list1)
                f.write(tuple_str + '\n')
        print ("Save csv to {}".format(csv_file_path))


        # ------------------------------------------------------------------------------------------------------------------
        # Write txt file
        # ------------------------------------------------------------------------------------------------------------------
        # with open (txt_file_path, 'w') as f:
        #     for i,tuple in enumerate(write_list):
        #         id = str(i)
        #         loss = str(tuple[0])
        #         n_iter = str(tuple[1])
        #         rmse = str(tuple[2])
        #         avg_pc = str(tuple[3])
        #         polar_percent = str(tuple[4])
        #         random_state = str(tuple[5])
        #
        #         f.write("--------------------------------------------\n".format(id))
        #         f.write("id: {}\n".format(id))
        #         f.write("loss: {}\n".format(loss))
        #         f.write("n_iter: {}\n".format(n_iter))
        #         f.write("rmse: {}\n".format(rmse))
        #         f.write("avg_pc: {}\n".format(avg_pc))
        #         f.write("polar_percent: {}\n".format(polar_percent))
        #         f.write("random_state: {}\n".format(random_state))
        # print ("Save txt to {}".format(txt_file_path))
        # ------------------------------------------------------------------------------------------------------------------