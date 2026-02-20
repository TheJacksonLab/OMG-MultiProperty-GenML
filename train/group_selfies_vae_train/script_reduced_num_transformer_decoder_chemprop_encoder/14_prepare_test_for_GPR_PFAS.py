import os
import torch

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem


if __name__ == '__main__':
    """
    This script prepares the valid & test property data for GPR.
    """
    ####### MODIFY #######
    # set dir for trained GenML
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

    # load data
    data_idx = torch.load(os.path.join(data_dir, 'valid_idx.pth'))
    # data_idx = torch.load(os.path.join(data_dir, 'test_idx.pth'))
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

    # get the number of Fluroine
    def get_num_fluorine(smi: str) -> int:
        """Return the number of fluorine atoms in a molecule given its SMILES string."""
        mol = Chem.MolFromSmiles(smi)
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    methyl_terminated_polymer_list = df_data['methyl_terminated_product'].tolist()
    num_fluorine_train_polymers_arr = np.array([get_num_fluorine(methyl_terminated_polymer) for methyl_terminated_polymer in methyl_terminated_polymer_list]).reshape(-1, 1)

    # targeted properties
    targeted_property = np.hstack((monomer_phi_arr, homo_lumo_gap_arr, chi_water_arr, num_fluorine_train_polymers_arr))

    # save
    gpr_dir = Path(os.path.join(model_check_point_directory, 'gpr'))
    gpr_dir.mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(gpr_dir, 'valid_property_PFAS'), targeted_property)
    # np.save(os.path.join(gpr_dir, 'test_property_PFAS'), targeted_property)
