# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# hyper parameter correlation only test with PCA, Z-score
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
import bisect
import math
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
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component

# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
unique_id = 0
unique_start = 0

# CHOSEN_HYPER_PARAMETER
#CHOSEN_HYPER_PARAMETER = 'learning_rate'
#CHOSEN_HYPER_PARAMETER = 'learning_rate_init_constant'
#CHOSEN_HYPER_PARAMETER = 'learning_rate_init_invscaling'
#CHOSEN_HYPER_PARAMETER = 'activation_function'
#CHOSEN_HYPER_PARAMETER = 'alpha'
#CHOSEN_HYPER_PARAMETER = 'early_stopping'
CHOSEN_HYPER_PARAMETER = 'validation_fraction'
#CHOSEN_HYPER_PARAMETER = 'pca_n_component'

#CHOSEN_HYPER_PARAMETER = 'hidden_layer_depth'
#CHOSEN_HYPER_PARAMETER = 'hidden_layer_nodes'



#

EXPERIMENTS = 30 # the times that every CHOSEN_HYPER_PARAMETER with different value are trained
is_standardisation_list = [True, False]
is_PCA_list = [True, False]
TRAILS= 30 # the number of different combination of hyper-parameters
pca_min_component = 8
RANDOM_SEED_OFFSET = 54385438
EXPERIMENT_RANDOM_SEED_OFFSET = 38453845
data_set = 'dow_jones_extended'
random_state_total = 50
tol = 1e-10
classifier = 'classifier'
shifting_size = 13
shift_num = 10
training_window_size = 74 # weeks, make sure shift_size*shift_num is fixed
is_training_window_flexible = False
shifting_size_min = 5
shifting_size_max = 30


input_folder = os.path.join('dow_jones_index_extended', 'dow_jones_index_extended_labeled')
input_folder = os.path.join(parent_folder, 'data', input_folder)

save_folder_temp = os.path.join(parent_folder, 'hyper_parameter_correlation_test', data_set, classifier)

# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
is_standardisation = True
is_PCA = True

for is_standardisation, is_PCA in list(itertools.product(is_standardisation_list, is_PCA_list)):

    # (1.) build classifer
    mlp_regressor1 = MlpTradeClassifier()
    clsfy_name = '{}_hyper_parameter_{}_{}'.format(data_set, classifier, CHOSEN_HYPER_PARAMETER)
    clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)


    # (2.) GET TRANINING SET
    data_per = 1.0 # the percentage of data using for training, validation and testing
    #dev_per = 0.0 # the percentage of data using for developing



    feature_switch_tuple_all_1 = get_full_feature_switch_tuple(input_folder)

    # (3.)
    samples_feature_list, samples_value_list, date_str_list, stock_id_list = mlp_regressor1._feed_data(input_folder,
                                                                  data_per=data_per,
                                                                  feature_switch_tuple=feature_switch_tuple_all_1,
                                                                is_random=False, mode='clf')


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

    pca_n_component_max = read_pca_component(input_folder)


    # ----------------------------------------------------------------------------------------------------------------------
    # generator builder
    # ----------------------------------------------------------------------------------------------------------------------
    def build_generator_from_pool(random_pool, trails, experiment_count, constant = False,
                                  constant_value = '', no_replicate = False, even_split = False):

        even_split_gap_list = [k * int(trails/len(random_pool)) for k, x in enumerate(random_pool)]
        for i in range(trails):
            random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
            if constant:
                if constant_value:
                    random_sample = constant_value
                else:
                    print ("Constant value is empty!")
                    sys.exit()
            elif even_split:
                index = bisect.bisect(even_split_gap_list, i) - 1
                random_sample = random_pool[index]
            else:
                random_sample = random.sample(random_pool, 1)[0]
                if no_replicate:
                    random_pool.remove(random_sample)
            yield random_sample

    def build_generator_from_range(target_range, trails, experiment_count, linspace = False):
        if linspace:
            value_list = np.linspace(*target_range, num = trails)
            for value in value_list:
                yield value
        else:
            for i in range(trails):
                random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
                random_value = random.uniform(*target_range)
                yield random_value
    # ----------------------------------------------------------------------------------------------------------------------


    for experiment_count, experiment in enumerate(range(EXPERIMENTS)):


        # (1.) activation function
        activation_list = ['identity', 'logistic', 'tanh', 'relu']
        if CHOSEN_HYPER_PARAMETER == 'activation_function':
            activation_random_sample_generator = build_generator_from_pool(activation_list, TRAILS, experiment_count,
                                                                           even_split=True)
        else:
            activation_random_sample_generator = build_generator_from_pool(activation_list, TRAILS, experiment_count)

        # (2.) L2 penalty (regularization term) parameter.
        alpha_range = (0.00001, 0.001)
        if CHOSEN_HYPER_PARAMETER =='alpha':
            alpha_random_sample_generator = build_generator_from_range(alpha_range, TRAILS, experiment_count,
                                                                       linspace=True)
        else:
            alpha_random_sample_generator = build_generator_from_range(alpha_range, TRAILS, experiment_count)

        # (3.) learning_rate
        learning_rate_list = ['constant', 'invscaling']
        if CHOSEN_HYPER_PARAMETER == 'learning_rate':
            learning_rate_random_sample_generator = build_generator_from_pool(learning_rate_list, TRAILS,
                                                                                    experiment_count,
                                                                                    even_split=True)
        elif CHOSEN_HYPER_PARAMETER == 'learning_rate_init_constant':
            learning_rate_random_sample_generator = build_generator_from_pool(learning_rate_list, TRAILS,
                                                                                    experiment_count, constant=True,
                                                                                    constant_value='constant')
        elif CHOSEN_HYPER_PARAMETER == 'learning_rate_init_invscaling':
            learning_rate_random_sample_generator = build_generator_from_pool(learning_rate_list, TRAILS,
                                                                                    experiment_count, constant=True,
                                                                                    constant_value='invscaling')
        else:
            learning_rate_random_sample_generator = build_generator_from_pool(learning_rate_list, TRAILS, experiment_count)

        # (4.) learning_rate_init
        learning_rate_init_range = (0.00001, 0.1)
        if CHOSEN_HYPER_PARAMETER == 'learning_rate_init_constant' or CHOSEN_HYPER_PARAMETER == 'learning_rate_init_invscaling':
            learning_rate_init_random_sample_generator = build_generator_from_range(learning_rate_init_range, TRAILS,
                                                                                    experiment_count, linspace=True)
        else:
            learning_rate_init_random_sample_generator = build_generator_from_range(learning_rate_init_range, TRAILS,
                                                                                experiment_count)

        # (5.) early-stopping
        early_stopping_list = [True, False]
        if CHOSEN_HYPER_PARAMETER == 'early_stopping':
            early_stopping_random_sample_generator = build_generator_from_pool(early_stopping_list, TRAILS,
                                                                               experiment_count,even_split=True)
        elif CHOSEN_HYPER_PARAMETER == 'validation_fraction':

            early_stopping_random_sample_generator = build_generator_from_pool(early_stopping_list, TRAILS,
                                                                               experiment_count,constant=True,
                                                                               constant_value=True)
        else:
            early_stopping_random_sample_generator = build_generator_from_pool(early_stopping_list, TRAILS,
                                                                               experiment_count)

        # (6.) early-stopping validation_fraction
        validation_fraction_range = (0.1, 0.3)
        if CHOSEN_HYPER_PARAMETER == 'validation_fraction':
            validation_fraction_random_sample_generator = build_generator_from_range(validation_fraction_range, TRAILS,
                                                                                     experiment_count, linspace=True)
        else:
            validation_fraction_random_sample_generator = build_generator_from_range(validation_fraction_range, TRAILS,
                                                                                 experiment_count)

        # (7.) HIDDEN LAYERS
        hidden_layer_depth = (1,2)
        hidden_layer_node  = (20,400)
        hidden_layer_node_max_range = (40,800)

        def hidden_layer_generator(hidden_layer_depth, hidden_layer_node, experiment_count, hidden_layer_node_max_range):
            hidden_layer_depth_pool = [x for x in range(hidden_layer_depth[0],hidden_layer_depth[1]+1)]
            even_split_hidden_layer_depth_list = [k * int(TRAILS / len(hidden_layer_depth_pool))
                                                  for k, x in enumerate(hidden_layer_depth_pool)]
            #hidden_layer_node_pool = [x for x in range(*hidden_layer_node)]
            # even_split_hidden_layer_node_list = [k * int(TRAILS / len(hidden_layer_node_pool))
            #                                      for k, x in enumerate(hidden_layer_node_pool)]
            if CHOSEN_HYPER_PARAMETER == 'hidden_layer_nodes':
                MIN_NODE_IN_1_LAYER = 1
                hidden_layer_node_max_range = (40,800)
                hidden_layer_node_max_list = list(build_generator_from_range(hidden_layer_node_max_range, TRAILS,
                                           experiment_count, linspace=True))
                for i in range(TRAILS):
                    hidden_layer_node_max = math.floor(hidden_layer_node_max_list[i])
                    random.seed(i + experiment_count * EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                    layer_depth = random.randint(hidden_layer_depth[0], hidden_layer_depth[1])
                    hidden_layer_sizes = [0 for x in range(0,layer_depth)]
                    node_left = hidden_layer_node_max
                    layer_left = layer_depth - 1
                    for j, layer in enumerate(hidden_layer_sizes):
                        if j == len(hidden_layer_sizes) - 1:
                            hidden_layer_sizes[j] = node_left
                        else:
                            random.seed(i + j + experiment_count * EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                            layer_node = random.randint(MIN_NODE_IN_1_LAYER,
                                                        node_left-layer_left*MIN_NODE_IN_1_LAYER)
                            hidden_layer_sizes[j] = layer_node
                            node_left -= layer_node
                            layer_left -= 1
                    hidden_layer_sizes_tuple = tuple(hidden_layer_sizes)
                    yield hidden_layer_sizes_tuple
            else:

                for i in range(TRAILS):
                    hidden_layer_sizes = []
                    if CHOSEN_HYPER_PARAMETER == 'hidden_layer_depth':
                        index = bisect.bisect(even_split_hidden_layer_depth_list, i) - 1
                        layer_depth = hidden_layer_depth_pool[index]
                    else:
                        random.seed(i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                        layer_depth = random.randint(hidden_layer_depth[0],hidden_layer_depth[1])
                    for j in range(layer_depth):
                        random.seed(j + i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                        layer_node = random.randint(*hidden_layer_node)
                        hidden_layer_sizes.append(layer_node)
                    hidden_layer_sizes_tuple = tuple(hidden_layer_sizes)
                    yield hidden_layer_sizes_tuple

        hidden_layer_size= hidden_layer_generator(hidden_layer_depth, hidden_layer_node, experiment_count,
                                                  hidden_layer_node_max_range)


        # (8.) random seed for weight initialisation
        random_state_range = [x for x in range(0,99999)]
        random_state_random_sample_generator = build_generator_from_pool(random_state_range, TRAILS,
                                                                          experiment_count, no_replicate = True)

        # (9.) principle component
        pca_component_pool = [x for x in range(int(pca_n_component_max/2),pca_n_component_max)]
        if CHOSEN_HYPER_PARAMETER == 'pca_n_component':
            pca_component_random_sample_generator = build_generator_from_pool(pca_component_pool, TRAILS,
                                                                          experiment_count, even_split = True)
        else:
            pca_component_random_sample_generator = build_generator_from_pool(pca_component_pool, TRAILS,
                                                                          experiment_count)





        hyper_parameter_trail_zip = zip(activation_random_sample_generator, alpha_random_sample_generator,
                                        learning_rate_random_sample_generator, learning_rate_init_random_sample_generator,
                                        early_stopping_random_sample_generator, validation_fraction_random_sample_generator,
                                        hidden_layer_size, random_state_random_sample_generator,
                                        pca_component_random_sample_generator)

        # hyper_parameter_size = len(hyper_parameter_trail_list)
        # print ("hyper_parameter_size: ", hyper_parameter_size)
        # print ("hyper_parameter_trail_list: ", hyper_parameter_trail_list)



        for i, hyper_paramter_tuple in enumerate(hyper_parameter_trail_zip):
            if unique_id < unique_start:
                unique_id += 1
                continue

            # read hyper parameters
            activation_function = hyper_paramter_tuple[0]
            alpha = float("{:7f}".format(hyper_paramter_tuple[1]))
            learning_rate = hyper_paramter_tuple[2]
            learning_rate_init = float("{:7f}".format(hyper_paramter_tuple[3]))
            early_stopping = hyper_paramter_tuple[4]
            if early_stopping:
                validation_fraction = float("{:7f}".format(hyper_paramter_tuple[5]))
            else:
                validation_fraction = 0.0
            hidden_layer_sizes = hyper_paramter_tuple[6]
            random_state = hyper_paramter_tuple[7]
            # PCA n component
            if data_preprocessing == 'pca' or data_preprocessing == 'pca_standardization':
                pca_n_component = hyper_paramter_tuple[8]
            else:
                pca_n_component = None
            #

            mlp_regressor1.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                                                               date_str_list, stock_id_list, is_cv=True,
                                                               shifting_size=shifting_size,
                                                               shift_num=shift_num,
                                                               is_standardisation=is_standardisation, is_PCA=is_PCA,
                                                               pca_n_component=pca_n_component,
                                                               training_window_size =training_window_size)



            trail = i
            n_iter_list = []
            loss_list = []
            f1_list = []
            accuracy_list = []
            random_state_list = []



            random_seed = 'window_shift'

            shift_n_iter_list = []
            shift_loss_list = []
            shift_f1_list = []
            shift_accuracy_list = []


            mlp_regressor1.set_mlp_clf(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init,
                                              random_state=random_state,
                               verbose = False, learning_rate = learning_rate, early_stopping =early_stopping, alpha= alpha,
                                              validation_fraction = validation_fraction, activation = activation_function)

            for shift in mlp_regressor1.validation_dict[random_seed].keys():
                mlp_regressor1.trade_rs_cv_load_train_dev_data(random_seed, shift)
                n_iter, loss = mlp_regressor1.clf_train(save_clsfy_path=clf_path)
                average_f1, accuracy = mlp_regressor1.clf_dev(save_clsfy_path=clf_path, is_cv=True, is_return=True)
                shift_n_iter_list.append(n_iter)
                shift_loss_list.append(loss)
                shift_f1_list.append(average_f1)
                shift_accuracy_list.append(accuracy)


            avg_n_iter = np.average(shift_n_iter_list)
            avg_loss= np.average(shift_loss_list)
            avg_f1 = np.average(shift_f1_list)
            avg_accuracy = np.average(shift_accuracy_list)


            hidden_layer_write_str = '_'.join([str(x) for x in hidden_layer_sizes])
            hidden_layer_depth = len(hidden_layer_sizes)
            hidden_layer_nodes = sum(hidden_layer_sizes)

            print ("-----------------------------------------------------------------------------------")
            print ("unique_id ", unique_id)
            print ("hidden_layer_sizes: ", hidden_layer_sizes)
            print ("is_PCA", is_PCA)
            print ("is_standardisation", is_standardisation)
            print ("pca_n_component: ", pca_n_component)
            print ("random_state: {}|{}".format(random_state, random_state_total))
            print ("activation_function: ", activation_function)
            print ("alpha: ", alpha)
            print ("learning_rate: ", learning_rate)
            print ("learning_rate_init: ", learning_rate_init)
            print ("early_stopping: ", early_stopping)
            print ("validation_fraction: ", validation_fraction)
            print ("avg_n_iter: ", avg_n_iter)
            print ("avg_loss: ", avg_loss)
            print ("avg_f1: ", avg_f1)
            print ("avg_accuracy: ", avg_accuracy)
            print ("Testing percent: {:.7f}%".format(100*i/TRAILS))
            print ("experiment: {}/{}".format(experiment, EXPERIMENTS))

            # ==========================================================================================================


            write_tuple = (experiment, trail, random_state_total, pca_n_component, activation_function, alpha,
                           learning_rate, learning_rate_init, early_stopping,validation_fraction, hidden_layer_write_str
                           , hidden_layer_depth, hidden_layer_nodes, random_state)
            save_folder = os.path.join(save_folder_temp, data_preprocessing,CHOSEN_HYPER_PARAMETER)
            csv_file_path = os.path.join(save_folder, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(*write_tuple))

            print ("csv_file_path: ", csv_file_path)

            with open(csv_file_path, 'w') as f:
                write_list = [avg_loss, avg_n_iter, avg_f1, avg_accuracy]
                write_list = [str(x) for x in write_list]
                f.write(','.join(write_list))
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

            unique_id += 1