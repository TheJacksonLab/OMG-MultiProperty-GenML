import os
import json
import torch

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    """
    This script prepares data for the PCA of latent space with a property direction for training samples. 
    1) Monomer Phi index (relevant to Tg)
    2) Band gap (electronic property)
    3) Chi water (solubility).   
    """
    ####### MODIFY #######
    # property name
    dir_name = 'polymerization_step_growth_750k'
    train_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/group_selfies_vae_train/{dir_name}'
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    model_save_directory = os.path.join(
        train_dir,
        'divergence_weight_2.949_latent_dim_104_learning_rate_0.003',
    )
    model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_54')

    # set dir for fitting to polymer properties
    model_fit_dir = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer'
    ####### MODIFY #######

    #### STEP 1) Load relevant data & models
    # load VAE
    device = torch.device('cpu')
    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    dtype = vae_parameters['dtype']

    # state dict for property prediction network
    state_dict = torch.load(os.path.join(model_check_point_directory, 'property_predictor.pth'), map_location=device)

    # load training data
    df = pd.read_csv(os.path.join(data_dir, 'two_reactants.csv'))
    train_idx = torch.load(os.path.join(data_dir, f'train_idx_randomly_sampled_for_latent_space_visualization.pth'))  # the whole training data
    print(f'The number of training samples is {len(train_idx)}', flush=True)
    df_train = df.iloc[train_idx]

    #### STEP 2) Prepare dataframe
    # PCA
    pca_results_dir = os.path.join(model_check_point_directory, 'pca')
    train_z_mean = np.load(os.path.join(pca_results_dir, 'z_mean.npy'))  # (num_data, latent_space_dim)
    train_property_z_mean = np.load(os.path.join(pca_results_dir, 'property_z_mean.npy'))

    latent_space_dim = train_z_mean.shape[1]
    n_components = latent_space_dim
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(train_z_mean)

    # load PCA data & create df columns
    whole_PCA_name_list = ['PC%d' % (num + 1) for num in range(n_components)]
    df_columns = whole_PCA_name_list + df_train.columns.to_list()
    z_coordinates_name = [f'z_{num + 1}_mean' for num in range(train_z_mean.shape[1])]
    df_columns.extend(z_coordinates_name)  # Idx of latent axis starting from 1
    df_columns.extend(['monomer_phi_prediction', 'homo_lumo_gap_prediction', 'chi_water_prediction'])  # prediction columns
    df_columns.extend(['HOMO_LUMO_gap_Boltzmann_average'])  # HOMO-LUMO gap
    df_columns.extend(['phi_prediction_error', 'homo_lumo_gap_prediction_error', 'chi_water_prediction_error', 'total_prediction_error'])  # prediction error
    df_columns.extend(['fitted_Tg_[K]_true', 'fitted_Tg_[K]_pred', 'fitted_band_gap_[eV]_true', 'fitted_band_gap_[eV]_pred', 'chi_water_true', 'chi_water_pred'])
    print('PCA is done', flush=True)

    # explained variance to determine the threshold
    exp_var_pca = pca.explained_variance_ratio_  # normalized ratio
    threshold = exp_var_pca.mean()
    cnt = (exp_var_pca >= threshold).sum()
    print(f'The number of principal components having a high variance: {cnt}')

    # create df
    principal_df = pd.DataFrame(data=None, columns=df_columns)

    # add z data
    for idx, arr in enumerate(train_z_mean.T):
        principal_df[f'z_{idx + 1}_mean'] = arr

    # add polymerization mechanism idx
    for column in df_train.columns:
        principal_df[column] = df_train[column].to_numpy()

    # add principal components
    for pca_idx, pca_name in enumerate(whole_PCA_name_list):  # pca_idx starting from zero
        principal_df[pca_name] = principal_components[:, pca_idx]

    # preparation training data
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    train_idx_for_property_scaling = torch.load(os.path.join(data_dir, 'train_idx.pth'))  # whole train_idx for train property scaling
    property_train_for_property_scaling = property_value[train_idx_for_property_scaling]

    # add prediction information for monomer phi
    monomer_phi_property_idx = 0
    monomer_phi_direction_vector = state_dict[f'property_network_module.{monomer_phi_property_idx}.weight'].cpu().numpy().squeeze()
    monomer_phi_bias_vector = state_dict[f'property_network_module.{monomer_phi_property_idx}.bias'].cpu().numpy().squeeze()
    monomer_phi_prediction = principal_df[z_coordinates_name].to_numpy() @ monomer_phi_direction_vector + monomer_phi_bias_vector

    property_scaler = StandardScaler()  # property scaling
    property_scaler = property_scaler.fit(property_train_for_property_scaling[:, monomer_phi_property_idx].reshape(-1, 1))  # fit
    principal_df['monomer_phi_prediction'] = property_scaler.inverse_transform(monomer_phi_prediction.reshape(-1, 1))

    # fit monomer phi to Tg
    model_fit_json_monomer_phi = os.path.join(model_fit_dir, 'Tg', 'model_fit', 'inverse_params.json')

    def inverse_function(x, a, b, c):  # inverse function
        return a / (x + b) + c
    with open(model_fit_json_monomer_phi, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_inverse = lambda x: inverse_function(x, **params)  # define function
    principal_df['fitted_Tg_[K]_true'] = principal_df['normalized_monomer_phi_Boltzmann_average'].apply(lambda x: fitted_inverse(x))
    principal_df['fitted_Tg_[K]_pred'] = principal_df['monomer_phi_prediction'].apply(lambda x: fitted_inverse(x))

    # add prediction information for homo lumo gap
    lumo_energy_property_idx = 2
    lumo_energy_property_direction_vector = state_dict[f'property_network_module.{lumo_energy_property_idx}.weight'].cpu().numpy().squeeze()
    lumo_energy_bias_vector = state_dict[f'property_network_module.{lumo_energy_property_idx}.bias'].cpu().numpy().squeeze()
    lumo_energy_prediction = principal_df[z_coordinates_name].to_numpy() @ lumo_energy_property_direction_vector + lumo_energy_bias_vector

    property_scaler = StandardScaler()  # property scaling
    property_scaler = property_scaler.fit(property_train_for_property_scaling[:, lumo_energy_property_idx].reshape(-1, 1))  # fit
    lumo_energy_prediction_unscaled = property_scaler.inverse_transform(lumo_energy_prediction.reshape(-1, 1))

    homo_energy_property_idx = 1
    homo_energy_property_direction_vector = state_dict[f'property_network_module.{homo_energy_property_idx}.weight'].cpu().numpy().squeeze()
    homo_energy_bias_vector = state_dict[f'property_network_module.{homo_energy_property_idx}.bias'].cpu().numpy().squeeze()
    homo_energy_prediction = principal_df[z_coordinates_name].to_numpy() @ homo_energy_property_direction_vector + homo_energy_bias_vector

    property_scaler = StandardScaler()  # property scaling
    property_scaler = property_scaler.fit(property_train_for_property_scaling[:, homo_energy_property_idx].reshape(-1, 1))  # fit
    homo_energy_prediction_unscaled = property_scaler.inverse_transform(homo_energy_prediction.reshape(-1, 1))

    principal_df['homo_lumo_gap_prediction'] = lumo_energy_prediction_unscaled - homo_energy_prediction_unscaled  # append

    # add homo lumo gap
    principal_df['HOMO_LUMO_gap_Boltzmann_average'] = principal_df['LUMO_Boltzmann_average'].to_numpy() - principal_df['HOMO_Boltzmann_average'].to_numpy()

    # fit to computational band gap
    model_fit_json_band_gap = os.path.join(model_fit_dir, 'gap_bulk', 'model_fit', 'linear_params.json')
    def linear_function(x, gradient, bias):  # linear function
        return gradient * x + bias
    with open(model_fit_json_band_gap, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_linear_band_gap = lambda x: linear_function(x, **params)  # define function
    principal_df['fitted_band_gap_[eV]_true'] = principal_df['HOMO_LUMO_gap_Boltzmann_average'].apply(lambda x: fitted_linear_band_gap(x))
    principal_df['fitted_band_gap_[eV]_pred'] = principal_df['homo_lumo_gap_prediction'].apply(lambda x: fitted_linear_band_gap(x))

    # add prediction information for chi water
    chi_water_property_idx = 8
    chi_water_direction_vector = state_dict[f'property_network_module.{chi_water_property_idx}.weight'].cpu().numpy().squeeze()
    chi_water_bias_vector = state_dict[f'property_network_module.{chi_water_property_idx}.bias'].cpu().numpy().squeeze()
    chi_water_prediction = principal_df[z_coordinates_name].to_numpy() @ chi_water_direction_vector + chi_water_bias_vector

    property_scaler = StandardScaler()  # property scaling
    property_scaler = property_scaler.fit(property_train_for_property_scaling[:, chi_water_property_idx].reshape(-1, 1))  # fit
    principal_df['chi_water_prediction'] = property_scaler.inverse_transform(chi_water_prediction.reshape(-1, 1))

    # fit chi_water to experimental chi_water
    model_fit_json_chi_water = os.path.join(model_fit_dir, 'Chi', 'model_fit', 'water_chi.json')
    with open(model_fit_json_chi_water, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_linear_chi_water = lambda x: linear_function(x, **params)  # define function
    principal_df['chi_water_true'] = principal_df['chi_parameter_water_mean'].apply(lambda x: fitted_linear_chi_water(x))
    principal_df['chi_water_pred'] = principal_df['chi_water_prediction'].apply(lambda x: fitted_linear_chi_water(x))

    # update idx
    principal_df.index = df_train.index

    # calculate the error
    phi_error = np.abs(principal_df['normalized_monomer_phi_Boltzmann_average'].to_numpy() - principal_df['monomer_phi_prediction'].to_numpy())
    homo_lumo_gap_error = np.abs(principal_df['HOMO_LUMO_gap_Boltzmann_average'].to_numpy() - principal_df['homo_lumo_gap_prediction'].to_numpy())
    chi_water_error = np.abs(principal_df['chi_parameter_water_mean'].to_numpy() - principal_df['chi_water_prediction'].to_numpy())

    principal_df['phi_prediction_error'] = phi_error
    principal_df['homo_lumo_gap_prediction_error'] = homo_lumo_gap_error
    principal_df['chi_water_prediction_error'] = chi_water_error
    principal_df['total_prediction_error'] = phi_error + homo_lumo_gap_error + chi_water_error

    # filter based on polymer properties for
    # 1) filter based on the min & max value of experimental Tg
    df_Tg = pd.read_csv('/home/sk77/PycharmProjects/omg_multi_task/train/fitting_to_polymer/Tg/normalized_phi_estimation.csv')
    experimental_Tg_arr = df_Tg['experimental_Tg'].to_numpy()
    Tg_pass_true = (principal_df['fitted_Tg_[K]_true'].to_numpy() >= experimental_Tg_arr.min()) & (principal_df['fitted_Tg_[K]_true'].to_numpy() <= experimental_Tg_arr.max())
    Tg_pass_pred = (principal_df['fitted_Tg_[K]_pred'].to_numpy() >= experimental_Tg_arr.min()) & (principal_df['fitted_Tg_[K]_pred'].to_numpy() <= experimental_Tg_arr.max())
    Tg_pass = Tg_pass_true & Tg_pass_pred

    # 2) filter based on the min & max value of computational band gap
    df_polymer_gap = pd.read_csv('/home/sk77/PycharmProjects/omg_multi_task/train/fitting_to_polymer/gap_bulk/data/Egb.csv')
    polymer_gap_arr = df_polymer_gap['value'].to_numpy()
    band_gap_pass_true = (principal_df['fitted_band_gap_[eV]_true'].to_numpy() >= polymer_gap_arr.min()) & (principal_df['fitted_band_gap_[eV]_true'].to_numpy() <= polymer_gap_arr.max())
    band_gap_pass_pred = (principal_df['fitted_band_gap_[eV]_pred'].to_numpy() >= polymer_gap_arr.min()) & (principal_df['fitted_band_gap_[eV]_pred'].to_numpy() <= polymer_gap_arr.max())
    band_gap_pass = band_gap_pass_true & band_gap_pass_pred

    # 3) (No) filter based on the min & max value of experimental chi water -> only five data points.
    # df_experimental_chi = pd.read_csv('/home/sk77/PycharmProjects/omg_multi_task/train/fitting_to_polymer/Chi/data/reference_chi_canon.csv')
    # df_experimental_chi = df_experimental_chi[df_experimental_chi['polymer_volume_fraction'] >= 0.2]  # avoid a critical regime
    # df_experimental_chi_water = df_experimental_chi[df_experimental_chi['solvent'] == 'water']
    # experimental_chi_water_arr = df_experimental_chi_water['experimental_chi'].to_numpy()
    # chi_water_pass = (principal_df['chi_water_true'].to_numpy() >= experimental_chi_water_arr.min()) & (principal_df['chi_water_true'].to_numpy() <= experimental_chi_water_arr.max())

    # filter
    pass_bool = Tg_pass & band_gap_pass
    principal_df_pass = principal_df[pass_bool]

    # save
    pca_dir_3d_landscape = Path(os.path.join(pca_results_dir, '3d_landscape'))  # save dir
    pca_dir_3d_landscape.mkdir(exist_ok=True, parents=True)
    principal_df_pass.to_csv(os.path.join(pca_dir_3d_landscape, 'principal_df.csv'))
