import os
import sys
import torch
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

PATH_OMG_MULTI_TASK = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML'
sys.path.append(PATH_OMG_MULTI_TASK)
from vae.preprocess_decoder_transformer import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot


if __name__ == '__main__':
    """
    This script prepares data to train Group SELFIES VAE for a Transformer decoder. 
    """
    ############# MODIFY #############
    # load data
    load_directory_name = 'train/data/group_selfies_vae_data/polymerization_step_growth_750k'
    load_directory = os.path.join(PATH_OMG_MULTI_TASK, load_directory_name)
    data_path = os.path.join(load_directory, 'two_reactants.csv')

    # load idx for data splits -> same for Molecule Chef
    train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'train_bag_idx.pth'))
    temp_idx = torch.load(os.path.join(load_directory, 'test_bag_idx.pth'))
    test_size = 0.1
    random_state = 42

    # save directory
    save_directory = load_directory

    property_columns_Boltzmann_average = ['normalized_monomer_phi'] + \
                                         ['HOMO', 'LUMO']
    property_columns_mean = ['chi_parameter_water']  # mean

    normalize_bool = [False] + \
                     [False, False] + \
                     [False]
    ############# MODIFY #############

    # Target properties
    target_property_list = [f'{col}_Boltzmann_average' for col in property_columns_Boltzmann_average] + [f'{col}_mean' for col in property_columns_mean]
    assert len(target_property_list) == len(normalize_bool)

    # for normalized polarizability
    for idx, normalize in enumerate(normalize_bool):
        if normalize:
            target_property_list[idx] = 'normalized_' + target_property_list[idx]

    # load data
    df_polymer = pd.read_csv(data_path)

    # prepare Group SELFIES
    print('Representation: Group SELFIES', flush=True)
    print('Preparing Group SELFIES ...', flush=True)
    encoding_list, encoding_alphabet, largest_molecule_len = get_selfie_and_smiles_encodings_for_dataset(df_polymer, save_vocab_dir=save_directory)

    # one-hot encoding
    print('--> Creating one-hot encoding...', flush=True)
    data, integer_encoded = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)  # (number of molecules, maximum length, number_of_letters)
    print('Finished creating one-hot encoding.', flush=True)

    # save data - due to the random order
    torch.save(data, os.path.join(save_directory, 'data.pth'), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(integer_encoded, os.path.join(save_directory, 'integer_encoded.pth'), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(encoding_list, os.path.join(save_directory, 'encoding_list.pth'))  # not with EOS token.
    torch.save(encoding_alphabet, os.path.join(save_directory, 'encoding_alphabet.pth'))
    torch.save(largest_molecule_len, os.path.join(save_directory, 'largest_molecule_len.pth'))  # one less compared to the integer_encoded.shape[1] due to the EOS token.

    # train : valid : test split - 8 : 1 : 1
    temp_train_idx, test_monomer_bags_idx = train_test_split(temp_idx,
                                                             test_size=int(df_polymer.shape[0] * test_size),
                                                             random_state=random_state)
    temp_train_idx, valid_monomer_bags_idx = train_test_split(temp_train_idx,
                                                              test_size=int(df_polymer.shape[0] * test_size),
                                                              random_state=random_state)
    train_monomer_bags_idx += temp_train_idx

    # save
    torch.save(train_monomer_bags_idx, os.path.join(save_directory, 'train_idx.pth'))
    torch.save(valid_monomer_bags_idx, os.path.join(save_directory, 'valid_idx.pth'))
    torch.save(test_monomer_bags_idx, os.path.join(save_directory, 'test_idx.pth'))
    property_value = df_polymer[target_property_list].to_numpy()  # (number of molecules, number of properties)
    torch.save(property_value, os.path.join(save_directory, 'property_value.pth'))

    # set parameters
    print(data.shape, flush=True)
    len_max_molec = data.shape[1]  # maximum length -> "not" excluding the first iodine
    print(len_max_molec, flush=True)
    len_alphabet = data.shape[2]  # number_of_letters
    print(len_alphabet, flush=True)

    # find idx
    eos_idx = 0
    asterisk_idx = 0

    # find '[EOS]' and '[I]'
    for idx, component in enumerate(encoding_alphabet):
        if component == '[EOS]':
            eos_idx = idx
        if component == '[I]':
            asterisk_idx = idx

    print(f'EOS idx is {eos_idx}', flush=True)
    print(f'Asterisk idx is {asterisk_idx}', flush=True)
