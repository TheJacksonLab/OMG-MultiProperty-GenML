import os
import sys
import time
import numpy as np
import pandas as pd

from math import ceil
from tqdm import tqdm
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.preprocessing import StandardScaler

import torch

from torch.distributions.multivariate_normal import MultivariateNormal

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies')
from group_selfies import GroupGrammar

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML')
from vae.decoder_transformer.torch import SequenceDecoder
from vae.encoder_chemprop.chemprop_encoder import Encoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from utils.functions import terminate_p_smi_with_CH3, encoding_return_scaled_mean_prediction
from vae.utils.evaluation import tokenids_to_vocab

CHEMPROP_PATH = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/vae/encoder_chemprop/chemprop'
sys.path.append(CHEMPROP_PATH)
from chemprop.data import get_data, MoleculeDataset
from chemprop.models import MoleculeModel


if __name__ == '__main__':
    """
    This script performs a Gaussian prior generation to get valid polymers and re-encoded values to be compared with
    chemprop values.         
    
    After running this script, go to chemprop_evidential/re-encoding dir to get chemprop prediction results. 
    
    """
    ####### MODIFY #######
    # load vae parameters
    # 1) linear for 11 properties
    dir_name = 'polymerization_step_growth_750k'
    train_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/group_selfies_vae_train/{dir_name}'
    model_save_directory = os.path.join(
        train_dir,
        'divergence_weight_2.949_latent_dim_104_learning_rate_0.003'
    )
    model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_54')

    # load data
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    data_path = os.path.join(data_dir, 'two_reactants.csv')
    train_idx_path = os.path.join(data_dir, 'train_idx.pth')

    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # save dir
    save_dir = os.path.join(model_check_point_directory, 're-encoding')

    number_of_generation = 3000
    ####### MODIFY #######

    # set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # train_idx
    train_idx = torch.load(train_idx_path, map_location=device)  # used to calculate novelty

    # load vae parameters
    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    eos_idx = vae_parameters['eos_idx']
    dtype = vae_parameters['dtype']

    # create group grammar
    vocab_save_path = os.path.join(data_dir, 'group_selfies_vocab.txt')
    grammar_fragment = GroupGrammar.from_file(vocab_save_path)

    # load data
    df_polymer = pd.read_csv(data_path)

    # preparation training data for scaling
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    train_idx_for_property_scaling = torch.load(os.path.join(data_dir, 'train_idx.pth'))  # whole train_idx for train property scaling

    # preparation training data for scaling
    property_train_for_property_scaling = property_value[train_idx_for_property_scaling]
    property_train_scaler = StandardScaler()
    property_train_scaler = property_train_scaler.fit(property_train_for_property_scaling)

    # load SELFIES data - due to the random order
    encoding_alphabet = torch.load(os.path.join(data_dir, 'encoding_alphabet.pth'), map_location=device)
    largest_molecule_len = torch.load(os.path.join(data_dir, 'largest_molecule_len.pth'), map_location=device)

    # set chemprop encoder
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

    # set property prediction network
    property_predictor = PropertyNetworkPredictionModule(
        latent_dim=vae_parameters['latent_dimension'],
        property_dim=vae_parameters['property_dim'],
        property_network_hidden_dim_list=vae_parameters['property_network_hidden_dim_list'],
        property_linear=vae_parameters['property_linear'],
        dtype=dtype,
        weights=vae_parameters['property_weights']
    ).to(device)

    # set decoder
    vae_decoder = SequenceDecoder(
        embedding_dim=vae_parameters['latent_dimension'],
        decoder_num_layers=vae_parameters['decoder_num_layers'],
        num_attention_heads=vae_parameters['decoder_num_attention_heads'],
        max_length=vae_parameters['max_length'][0],  # tuple(int)
        vocab=vae_parameters['vocab'][0],  # tuple(dict)
        loss_weights=vae_parameters['decoder_loss_weights'][0],  # tuple(None)
        add_latent=vae_parameters['decoder_add_latent'][0],  # tuple(bool)
    ).to(device)

    # load state dictionary
    vae_encoder.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'encoder.pth'), map_location=device))
    vae_decoder.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'decoder.pth'), map_location=device))
    property_predictor.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'property_predictor.pth'), map_location=device))

    # eval
    vae_encoder.eval()
    vae_decoder.eval()
    property_predictor.eval()

    # gaussian prior generation
    multivariate_normal = MultivariateNormal(
        torch.zeros(vae_parameters['latent_dimension'], dtype=dtype, device=device),
        torch.eye(vae_parameters['latent_dimension'], dtype=dtype, device=device)
    )

    # property idx
    monomer_phi_property_idx = 0
    lumo_energy_property_idx = 2
    homo_energy_property_idx = 1
    chi_water_property_idx = 8

    # batch size
    batch_size = 100
    batch_iteration = ceil(number_of_generation / batch_size)

    # generated polymer maximum length is less than original polymers
    max_length_to_decode = largest_molecule_len
    # max_length_to_decode = largest_molecule_len - 1  # excluding the first asterisk.
    len_alphabet = len(encoding_alphabet)
    generated_token_list = list()
    random_prior_list = list()
    linear_property_prediction_list = list()

    # reference property prediction
    latent_dimension = vae_parameters['latent_dimension']
    with torch.no_grad():
        reference_property_prediction = property_predictor(torch.zeros(latent_dimension, device=device).view(1, -1)).cpu().numpy()

    # generate
    with torch.no_grad():
        tqdm.write("Generation ...")
        pbar = tqdm(range(batch_iteration), total=batch_iteration, leave=True)
        time.sleep(1.0)
        with pbar as t:
            for iteration in t:
                # manual batch iterations
                start_idx = iteration * batch_size
                stop_idx = (iteration + 1) * batch_size

                if iteration == batch_iteration - 1:
                    stop_idx = number_of_generation

                # num molecules batch
                num_molecules_batch = stop_idx - start_idx

                # sample gaussian prior
                random_prior = multivariate_normal.sample(sample_shape=torch.Size([num_molecules_batch]))  # (num_molecules_batch, latent_dim)
                random_prior_list.extend(random_prior.cpu().numpy().tolist())

                # property prediction
                linear_property_prediction = property_predictor(random_prior).cpu().numpy()

                # beam search
                beam_size = 1
                predictions = vae_decoder.inference(random_prior, beam_size=beam_size)

                # get decoded tokens
                decoded_tokens = [
                    tokenids_to_vocab(predictions[sample][0].tolist(), vae_decoder.vocab, grammar_fragment, convert_to_smi=False) for sample in range(num_molecules_batch)
                ]

                # extend
                generated_token_list.extend(decoded_tokens)
                linear_property_prediction_list.extend(linear_property_prediction)

    # analysis
    assert len(random_prior_list) == len(generated_token_list)
    train_polymers = df_polymer['product'][train_idx]
    random_prior_arr = np.array(random_prior_list)
    linear_property_prediction_arr = np.array(linear_property_prediction_list)  # scaled
    linear_property_prediction_arr_unscaled = property_train_scaler.inverse_transform(linear_property_prediction_arr)

    # get monomer-level targeted functionalities
    monomer_phi_pred = linear_property_prediction_arr_unscaled[:, monomer_phi_property_idx]
    homo_lumo_gap_pred = linear_property_prediction_arr_unscaled[:, lumo_energy_property_idx] - linear_property_prediction_arr_unscaled[:, homo_energy_property_idx]
    chi_water_pred = linear_property_prediction_arr_unscaled[:, chi_water_property_idx]

    # validity, uniqueness, and novelty check
    generated_polymer_list = []
    for decoded_tokens in generated_token_list:
        # convert to smi
        try:
            decoded_str = Chem.MolToSmiles(grammar_fragment.decoder(decoded_tokens))
        except:  # sometimes fails
            generated_polymer_list.append(None)
            continue

        # replace
        canonical_smiles_generated_molecule = Chem.CanonSmiles(decoded_str.replace("I", "*"))

        # (1) the number of '*' should be 2
        asterisk_cnt = 0
        for char in canonical_smiles_generated_molecule:
            if char == '*':
                asterisk_cnt += 1

        if asterisk_cnt != 2:
            # print(f'The number of asterisks is not 2: {canonical_smiles_generated_molecule}')
            generated_polymer_list.append(None)
            continue

        # only consisting of '**'
        if canonical_smiles_generated_molecule == '**':
            # print(f'The generated molecule is **', flush=True)
            generated_polymer_list.append(None)
            continue

        # (2) '*' should be connected through single bond
        mol = Chem.MolFromSmiles(canonical_smiles_generated_molecule)
        flag = 1
        atoms = mol.GetAtoms()
        bond_order = list()
        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                if atom.GetDegree() != 1:  # asterisk should be the end of molecule
                    flag = 0  # exclude
                    break
                else:
                    bond_order.append(atom.GetExplicitValence())
        assert len(bond_order) == 2

        # check bond order
        if bond_order[0] == 1 and bond_order[1] == 1:
            pass
        else:
            flag = 0

        # exclude error
        if flag == 0:
            print(f"The connectivity of asterisks is not valid: {canonical_smiles_generated_molecule}", flush=True)
            generated_polymer_list.append(None)
            continue

        # pass all criteria
        generated_polymer_list.append(canonical_smiles_generated_molecule)

    # get df
    z_coordinates_name = [f'z_{num + 1}' for num in range(vae_parameters['latent_dimension'])]
    df_generated = pd.DataFrame(columns=['product'] + z_coordinates_name + ['monomer_phi_linear', 'homo_lumo_gap_linear', 'chi_water_linear'])
    df_generated['product'] = generated_polymer_list  # None if polymer is not valid.
    for z_idx, z_name in enumerate(z_coordinates_name):  # z_idx starting from zero
        df_generated[z_name] = random_prior_arr[:, z_idx]
    df_generated['monomer_phi_linear'] = monomer_phi_pred
    df_generated['homo_lumo_gap_linear'] = homo_lumo_gap_pred
    df_generated['chi_water_linear'] = chi_water_pred

    # get valid polymers
    assert df_generated.shape[0] == random_prior_arr.shape[0]
    num_latent_points = random_prior_arr.shape[0]
    df_valid = df_generated[df_generated['product'].apply(lambda x: x is not None)]

    # re-encoding strategy
    methyl_terminated_smi_list = [Chem.MolToSmiles(terminate_p_smi_with_CH3(product_smi)) for product_smi in df_valid['product']]
    data_chemprop = get_data(smi_list=[[methyl_terminated_smi] for methyl_terminated_smi in methyl_terminated_smi_list])
    data_chemprop.reset_features_and_targets()  # reset

    re_encoding_property_prediction_arr, _ = encoding_return_scaled_mean_prediction(
        data_chemprop, df_valid['product'].tolist(), vae_encoder, vae_decoder,
        property_predictor, grammar_fragment, batch_size=100
    )  # (1, num_data, num_property) -> scaled properties
    re_encoding_property_prediction_arr_unscaled = property_train_scaler.inverse_transform(re_encoding_property_prediction_arr[0])

    # append
    df_valid = df_valid.copy()
    df_valid['monomer_phi_re_encoding'] = re_encoding_property_prediction_arr_unscaled[:, monomer_phi_property_idx]
    df_valid['homo_lumo_gap_re_encoding'] = re_encoding_property_prediction_arr_unscaled[:, lumo_energy_property_idx] - re_encoding_property_prediction_arr_unscaled[:, homo_energy_property_idx]
    df_valid['chi_water_re_encoding'] = re_encoding_property_prediction_arr_unscaled[:, chi_water_property_idx]

    # analyze
    # 1) check validity
    number_of_valid_polymers = df_valid.shape[0]
    print(f"Gaussian prior generated valid {number_of_valid_polymers} polymers "
          f"{100 * (number_of_valid_polymers / num_latent_points):.3f}% ",
          flush=True)
    print('====================', flush=True)

    # save valid df
    df_valid.to_csv(os.path.join(save_dir, 'valid_gaussian.csv'), index=False)
