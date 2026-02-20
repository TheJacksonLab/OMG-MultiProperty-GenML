import os
import sys
import time
import pandas as pd

from math import ceil
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch

from torch.distributions.multivariate_normal import MultivariateNormal

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies')
from group_selfies import GroupGrammar

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML')
from vae.decoder_transformer.torch import SequenceDecoder
from utils import filter_valid_p_smiles
from vae.utils.evaluation import tokenids_to_vocab


if __name__ == '__main__':
    """
    This script performs a Gaussian prior generation.         
    """
    ####### MODIFY #######
    # load vae parameters (linear)
    dir_name = 'polymerization_step_growth_750k'
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    train_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/group_selfies_vae_train/{dir_name}'
    model_save_directory = os.path.join(
        train_dir,
        'divergence_weight_2.949_latent_dim_104_learning_rate_0.003'
    )
    model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_54')

    # non-linear - 1
    # dir_name = 'polymerization_step_growth_750k_non_linear'
    # data_dir = f'/home/sk77/PycharmProjects/omg_multi_task/train/data/group_selfies_vae_data/{dir_name}'
    # train_dir = f'/home/sk77/PycharmProjects/omg_multi_task/train/group_selfies_vae_train/{dir_name}'
    # model_save_directory = os.path.join(
    #     train_dir,
    #     'divergence_weight_9.569_latent_dim_64_learning_rate_0.001',
    # )
    # model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_39')

    # load data
    data_path = os.path.join(data_dir, 'two_reactants.csv')
    train_idx_path = os.path.join(data_dir, 'train_idx.pth')

    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    number_of_generation = 30000
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

    # load SELFIES data - due to the random order
    encoding_alphabet = torch.load(os.path.join(data_dir, 'encoding_alphabet.pth'), map_location=device)
    largest_molecule_len = torch.load(os.path.join(data_dir, 'largest_molecule_len.pth'), map_location=device)

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
    vae_decoder.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'decoder.pth'), map_location=device))

    # eval
    vae_decoder.eval()

    # gaussian prior generation
    multivariate_normal = MultivariateNormal(
        torch.zeros(vae_parameters['latent_dimension'], dtype=dtype, device=device),
        torch.eye(vae_parameters['latent_dimension'], dtype=dtype, device=device)
    )

    # batch size
    batch_size = 100
    batch_iteration = ceil(number_of_generation / batch_size)

    # generated polymer maximum length is less than original polymers
    max_length_to_decode = largest_molecule_len
    # max_length_to_decode = largest_molecule_len - 1  # excluding the first asterisk.
    len_alphabet = len(encoding_alphabet)
    generated_polymer_list = list()

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

                # beam search
                beam_size = 1
                predictions = vae_decoder.inference(random_prior, beam_size=beam_size)

                # get decoded tokens
                decoded_tokens = [
                    tokenids_to_vocab(predictions[sample][0].tolist(), vae_decoder.vocab, grammar_fragment, convert_to_smi=False) for sample in range(num_molecules_batch)
                ]

                # extend
                generated_polymer_list.extend(decoded_tokens)

    # analysis
    train_polymers = df_polymer['product'][train_idx]
    filtered_list = filter_valid_p_smiles(generated_polymer_list, grammar_fragment, number_of_generation, train_polymers)
