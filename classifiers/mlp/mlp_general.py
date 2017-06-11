# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP for general functions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# General import
# ==========================================================================================================
import os
import sys
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
from trade_general_funcs import feature_degradation
# ==========================================================================================================


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class MultilayerPerceptron:

    def __init__(self):

        # hidden_layer
        self.mlp_hidden_layer_sizes_list = []
        self.hidden_size_list = []
        #

        # feature switch
        self.feature_switch_list = []
        self.feature_selected_list = []
        #

        # iteration_loss
        self.iteration_loss_list = []
        self.tp_cv_iteration_loss_list = []
        #


    def read_selected_feature_list(self, folder, feature_switch_list):
        '''Used in topology test, for initializing self.feature_selected_list'''
        file_name_list = os.listdir(folder)
        file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
        with open(file_path_0, 'r', encoding='utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
            selected_feature_list = feature_degradation(feature_name_list, feature_switch_list)
        self.feature_selected_list.append(selected_feature_list)



    def _update_feature_switch_list(self, i):
        if i != 0:
            # --------------------------------------------------------------------------
            # update feature_switch_list and feature_selected list for easy output
            # --------------------------------------------------------------------------
            self.feature_switch_list.append(self.feature_switch_list[-1])
            self.feature_selected_list.append(self.feature_selected_list[-1])
            # --------------------------------------------------------------------------
