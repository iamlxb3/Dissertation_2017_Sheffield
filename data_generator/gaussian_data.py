import numpy as np
import os

class RandomGaussianData:
    def __init__(self):
        pass

    def generate_data(self, save_folder):

        sample_num = 5000
        sample_count = 0
        feature_num = 20
        label_list = ['dog1', 'dog2', 'cat1', 'cat2']
        mu_list = [0, 1, 2, 4]
        sigma_list = [0.1, 0.2, 0.5, 0.4]

        # dog1

        while sample_count <= sample_num:
            for i, label in enumerate(label_list):
                sample_count += 1
                mu = mu_list[i]
                sigma = sigma_list[i]
                feature_name = ["[{}]".format(x) for x in range(feature_num)]
                feature_value_list = list(np.random.normal(mu, sigma, feature_num))
                feature_list = [j for i in zip(feature_name, feature_value_list) for j in i]
                feature_list = [str(x) for x in feature_list]
                feature_str = ','.join(feature_list)

                # save_file
                file_name = "{}_{}.txt".format(sample_count, label)
                file_path = os.path.join(save_folder, file_name)
                with open (file_path, 'w', encoding = 'utf-8') as f:
                    f.write(feature_str)

        print ("Generated {} samples with {} features complete!".format(sample_num, feature_num))
