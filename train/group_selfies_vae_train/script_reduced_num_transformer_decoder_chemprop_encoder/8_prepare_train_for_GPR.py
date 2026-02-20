import os
import torch

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


if __name__ == '__main__':
    """
    This script performs K-means to train GPR.  
    """
    ####### MODIFY #######
    # set dir for trained GenML (linear)
    dir_name = 'polymerization_step_growth_750k'
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    train_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/group_selfies_vae_train/{dir_name}'
    model_save_directory = os.path.join(
        train_dir,
        'divergence_weight_2.949_latent_dim_104_learning_rate_0.003'
    )
    model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_54')

    # non-linear
    # dir_name = 'polymerization_step_growth_750k_non_linear'
    # data_dir = f'/home/sk77/PycharmProjects/omg_multi_task/train/data/group_selfies_vae_data/{dir_name}'
    # train_dir = f'/home/sk77/PycharmProjects/omg_multi_task/train/group_selfies_vae_train/{dir_name}'
    # model_save_directory = os.path.join(
    #     train_dir,
    #     'divergence_weight_9.569_latent_dim_64_learning_rate_0.001',
    # )
    # model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_39')
    ####### MODIFY #######

    #### Load z_mean & property_z_mean
    pca_dir = os.path.join(model_check_point_directory, 'pca')
    z_mean = np.load(os.path.join(pca_dir, 'z_mean.npy'))  # latent representations - mean vectors for training data

    # preparation training data
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    train_idx = torch.load(os.path.join(data_dir, 'train_idx.pth'))
    property_train = property_value[train_idx]
    assert z_mean.shape[0] == property_train.shape[0]

    # idx to use
    monomer_phi_property_idx = 0
    lumo_energy_property_idx = 2
    homo_energy_property_idx = 1
    chi_water_property_idx = 8

    # get targeted functionalities
    monomer_phi_arr = property_train[:, monomer_phi_property_idx].reshape(-1, 1)

    lumo_energy_arr = property_train[:, lumo_energy_property_idx]
    homo_energy_arr = property_train[:, homo_energy_property_idx]
    homo_lumo_gap_arr = (lumo_energy_arr - homo_energy_arr).reshape(-1, 1)

    chi_water_arr = property_train[:, chi_water_property_idx].reshape(-1, 1)

    # targeted properties
    targeted_property = np.hstack((monomer_phi_arr, homo_lumo_gap_arr, chi_water_arr))

    # K-means clustering
    def diverse_sampling_kmeans(X, k):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        centers = kmeans.cluster_centers_  # (k, features)
        labels = kmeans.labels_  # (k,)
        sampled_indices = []

        for i in range(k):
            cluster_points = X[labels == i]
            center = centers[i]
            # Pick the point closest to the center
            idx = np.argmin(np.linalg.norm(cluster_points - center, axis=1))
            sampled_indices.append(np.where(labels == i)[0][idx])

        return sampled_indices

    # K-means
    k = 2000
    sampled_indices = diverse_sampling_kmeans(z_mean, k=k)

    # save
    k_means_dir = Path(os.path.join(model_check_point_directory, 'gpr', f'k_means_num_{k}'))
    k_means_dir.mkdir(parents=True, exist_ok=True)
    sampled_z_mean = z_mean[sampled_indices]
    sampled_properties = targeted_property[sampled_indices]
    np.save(os.path.join(k_means_dir, 'sampled_z_mean'), sampled_z_mean)
    np.save(os.path.join(k_means_dir, 'sampled_properties'), sampled_properties)
