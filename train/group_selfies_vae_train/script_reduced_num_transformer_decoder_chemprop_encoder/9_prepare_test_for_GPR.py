import os
import sys
import torch

from pathlib import Path

import numpy as np
import pandas as pd
import time

from math import ceil

from tqdm import tqdm

PATH_OMG_MULTI_TASK = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML'
sys.path.append(PATH_OMG_MULTI_TASK)
from vae.encoder_chemprop.chemprop_encoder import Encoder

CHEMPROP_PATH = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/vae/encoder_chemprop/chemprop'
sys.path.append(CHEMPROP_PATH)
from chemprop.data import get_data, MoleculeDataset
from chemprop.models import MoleculeModel


if __name__ == '__main__':
    """
    This script searches for a latent space for multi-property optimizations for Tg, band gap, and Chi water. 
    This script also decodes the latent points to see molecular structures.  
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

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # load  data
    # data_idx = torch.load(os.path.join(data_dir, 'test_idx.pth'))
    data_idx = torch.load(os.path.join(data_dir, 'valid_idx.pth'))
    ####### MODIFY #######

    df = pd.read_csv(os.path.join(data_dir, 'two_reactants.csv'))
    df_data = df.iloc[data_idx]

    # save property data
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    property_data = property_value[data_idx]

    # idx to use
    monomer_phi_property_idx = 0
    lumo_energy_property_idx = 2
    homo_energy_property_idx = 1
    chi_water_property_idx = 8

    # get targeted functionalities
    monomer_phi_arr = property_data[:, monomer_phi_property_idx].reshape(-1, 1)

    lumo_energy_arr = property_data[:, lumo_energy_property_idx]
    homo_energy_arr = property_data[:, homo_energy_property_idx]
    homo_lumo_gap_arr = (lumo_energy_arr - homo_energy_arr).reshape(-1, 1)

    chi_water_arr = property_data[:, chi_water_property_idx].reshape(-1, 1)

    # targeted properties
    targeted_property = np.hstack((monomer_phi_arr, homo_lumo_gap_arr, chi_water_arr))

    # save
    gpr_dir = Path(os.path.join(model_check_point_directory, 'gpr'))
    gpr_dir.mkdir(parents=True, exist_ok=True)
    # np.save(os.path.join(gpr_dir, 'test_property'), targeted_property)
    np.save(os.path.join(gpr_dir, 'valid_property'), targeted_property)

    # z - encodings
    print('Loading data for Chemprop ...', flush=True)
    data_chemprop = get_data(smi_list=[[smi] for smi in df_data['methyl_terminated_product'].tolist()])
    data_chemprop.reset_features_and_targets()  # before doing training

    # load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    print(device, flush=True)
    dtype = vae_parameters['dtype']

    chemprop_mpn = MoleculeModel(
        hidden_size=vae_parameters['encoder_feature_dim'],
        bias=vae_parameters['mpn_bias'],
        depth=vae_parameters['mpn_depth'],
        device=device,
        dropout_rate=vae_parameters['mpn_dropout_rate'],
    )
    # set encoder
    vae_encoder = Encoder(
        chemprop_mpn=chemprop_mpn,
        in_dimension=vae_parameters['encoder_feature_dim'],
        layer_1d=vae_parameters['encoder_layer_1d'],
        layer_2d=vae_parameters['encoder_layer_2d'],
        layer_3d=vae_parameters['encoder_layer_3d'],
        latent_dimension=vae_parameters['latent_dimension'],
    ).to(device)
    vae_encoder.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'encoder.pth'), map_location=device))
    vae_encoder.eval()

    # for return
    z_mean, property_prediction = [], []
    batch_size = 100

    # reconstruct
    reconstruct_batch_iteration = ceil(len(data_chemprop) / batch_size)
    with torch.no_grad():
        pbar = tqdm(range(reconstruct_batch_iteration), total=reconstruct_batch_iteration, leave=True)
        time.sleep(1.0)
        with pbar as t:
            for batch_iteration in t:
                # manual batch iterations
                start_idx = batch_iteration * batch_size
                stop_idx = (batch_iteration + 1) * batch_size

                if batch_iteration == reconstruct_batch_iteration - 1:
                    stop_idx = len(data_chemprop)

                data_batch = MoleculeDataset([data_chemprop[idx] for idx in range(start_idx, stop_idx)])
                num_molecules_batch = stop_idx - start_idx

                # encode
                latent_points, mus, log_vars = vae_encoder(data_batch)

                # store
                z_mean += mus.cpu().tolist()

    # save
    z_mean_arr = np.array(z_mean)
    # np.save(os.path.join(gpr_dir, 'test_z_mean'), z_mean_arr)
    np.save(os.path.join(gpr_dir, 'valid_z_mean'), z_mean_arr)
