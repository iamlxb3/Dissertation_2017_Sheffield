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
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
sys.path.append(clf_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_regressor import MlpTradeRegressor
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
print ("Build MLP regressor for a-share data!")
# (1.) build classifer
mlp_regressor1 = MlpTradeRegressor()
clsfy_name = 'a_share_mlp_cv_PCA_regressor'
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)


# (3.) GET TRANINING SET
data_per = 1.0 # the percentage of data using for training and testing
dev_per = 0.0 # the percentage of data using for developing
train_data_folder = os.path.join('a_share','a_share_regression_data')
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
standardisation_file_path = os.path.join(parent_folder, 'data_processor','z_score')
pca_file_path = os.path.join(parent_folder,'data_processor','pca')
mlp_regressor1.trade_feed_and_separate_data(train_data_folder, dev_per = dev_per, data_per = data_per,
                                            standardisation_file_path = standardisation_file_path,
                                                            pca_file_path = pca_file_path)

TRAINING_SET = mlp_regressor1.training_set
TRAINING_VALUE_SET = mlp_regressor1.training_value_set


# (4.) GET TESTING SET
data_per = 1.0
dev_per = 1.0

test_data_folder = os.path.join('a_share', 'a_share_regression_data_test')
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
mlp_regressor1.trade_feed_and_separate_data(test_data_folder, dev_per=dev_per, data_per=data_per,
                                            is_test_folder=True,
                                            standardisation_file_path=standardisation_file_path,
                                            pca_file_path=pca_file_path)

TEST_TRAINING_SET = mlp_regressor1.dev_set
TEST_TRAINING_VALUE_SET = mlp_regressor1.dev_value_set
TEST_DATE_SET = mlp_regressor1.dev_date_set
TEST_STOCK_ID_SET = mlp_regressor1.dev_stock_id_set



hidden_layer_depth = [x for x in range(1, 4)]
hidden_layer_node = [x for x in range(100, 600)][::200]


hidden_layer_sizes_list = []
for layer_depth in hidden_layer_depth:
    hidden_layer_sizes_list_temp = list(itertools.product(hidden_layer_node, repeat=layer_depth))
    hidden_layer_sizes_list.extend(hidden_layer_sizes_list_temp)

random.shuffle(hidden_layer_sizes_list)
print ("hidden_layer_sizes_list: ", hidden_layer_sizes_list)
print ("Total {} hidden layers to test".format(len(hidden_layer_sizes_list)))
#sys.exit()

#hidden_layer_sizes_list = hidden_layer_sizes_list[-2:]

for hidden_layer_sizes in hidden_layer_sizes_list:

    n_iter_list = []
    loss_list = []
    rmse_list = []
    avg_pc_list = []
    random_state_list = []
    polar_percent_list = []

    hidden_layer_sizes = (200, 20)
    for i in range(100):
        print ("Hidden-layer-size-now: ", hidden_layer_sizes)
        random_state = i
        tol = 1e-10
        learning_rate_init = 0.1
        #learning_rate = 'invscaling'    #constant
        learning_rate = 'constant'    #constant
        early_stopping = False

        mlp_regressor1.set_regressor_test(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init,
                                          random_state = random_state, verbose = False, learning_rate = learning_rate,
                                          early_stopping = early_stopping)
        n_iter, loss = mlp_regressor1.regressor_train_test(TRAINING_SET, TRAINING_VALUE_SET, save_clsfy_path= clf_path)
        print ("Test regressor trained successfully for random_state: {}!".format(random_state))


        rmse, avg_pc, polar_percent = mlp_regressor1.regressor_dev_test(TEST_TRAINING_SET, TEST_TRAINING_VALUE_SET,
                                                         TEST_DATE_SET, TEST_STOCK_ID_SET,save_clsfy_path= clf_path)

        random_state_list.append(random_state)
        n_iter_list.append(n_iter)
        loss_list.append(loss)
        rmse_list.append(rmse)
        avg_pc_list.append(avg_pc)
        polar_percent_list.append(polar_percent)



    # ==========================================================================================================
    write_list = list(zip(loss_list, n_iter_list, rmse_list, avg_pc_list, polar_percent_list, random_state_list))
    write_list = sorted(write_list, key = lambda x:x[0])


    hidden_layer_write_str = '_'.join([str(x) for x in hidden_layer_sizes])

    save_folder = os.path.join(parent_folder, 'loss_test')
    csv_file_path = os.path.join(save_folder, 'loss_test_{}_{}_{}_ES_{}.csv'.format(learning_rate, learning_rate_init,
                                                                              hidden_layer_write_str, early_stopping))
    txt_file_path = os.path.join(save_folder, 'loss_test_{}_{}_{}_ES_{}.txt'.format(learning_rate, learning_rate_init,
                                                                              hidden_layer_write_str,early_stopping))

    with open (txt_file_path, 'w') as f:
        for i,tuple in enumerate(write_list):
            id = str(i)
            loss = str(tuple[0])
            n_iter = str(tuple[1])
            rmse = str(tuple[2])
            avg_pc = str(tuple[3])
            polar_percent = str(tuple[4])
            random_state = str(tuple[5])

            f.write("--------------------------------------------\n".format(id))
            f.write("id: {}\n".format(id))
            f.write("loss: {}\n".format(loss))
            f.write("n_iter: {}\n".format(n_iter))
            f.write("rmse: {}\n".format(rmse))
            f.write("avg_pc: {}\n".format(avg_pc))
            f.write("polar_percent: {}\n".format(polar_percent))
            f.write("random_state: {}\n".format(random_state))
    print ("Save txt to {}".format(txt_file_path))

    with open(csv_file_path, 'w') as f:
        for i, tuple1 in enumerate(write_list):
            list1 = list(tuple1)
            list1 = [str(x) for x in list1]
            tuple_str = ','.join(list1)
            f.write(tuple_str + '\n')
    print ("Save csv to {}".format(csv_file_path))
