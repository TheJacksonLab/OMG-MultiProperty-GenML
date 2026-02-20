import os
import sys
import time
import random

import torch
from torch.nn.init import xavier_uniform_, zeros_

import joblib
import numpy as np
import pandas as pd

import optuna

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from multiprocessing import Manager

from joblib import parallel_backend  # "https://github.com/rapidsai/dask-cuda/issues/789"

from pathlib import Path

from sklearn.preprocessing import StandardScaler

PATH_OMG_MULTI_TASK = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML'
sys.path.append(PATH_OMG_MULTI_TASK)
from vae.decoder_transformer.torch import SequenceDecoder
from vae.encoder_chemprop.chemprop_encoder import Encoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.training_optuna_decoder_transformer_encoder_chemprop import train_model
from vae.utils.save_decoder_transformer_encoder_chemprop import VAEParameters

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies')
from group_selfies import GroupGrammar

CHEMPROP_PATH = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/vae/encoder_chemprop/chemprop'
sys.path.append(CHEMPROP_PATH)
from chemprop.data import get_data, MoleculeDataset
from chemprop.models import MoleculeModel

############# MODIFY #############
dir_name = 'polymerization_step_growth_750k'

# property name - to get dimension for property network
property_columns_Boltzmann_average = ['normalized_monomer_phi'] + \
                                         ['HOMO', 'LUMO', 'normalized_polarizability'] + \
                                         ['s1_energy', 'dominant_transition_energy',
                                          'dominant_transition_oscillator_strength', 't1_energy']
property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']  # mean
target_property_list = [f'{col}' for col in property_columns_Boltzmann_average] + [f'{col}' for col in property_columns_mean]
print(f"Property dimension is {len(target_property_list)}", flush=True)
############# MODIFY #############

# dir
data_dir = os.path.join(PATH_OMG_MULTI_TASK, f'train/data/group_selfies_vae_data/{dir_name}')
train_dir = os.path.join(PATH_OMG_MULTI_TASK, f'train/group_selfies_vae_train/{dir_name}')

# model parameters
# 1) encoder
fingerprint_dim = 300  # chemprop MPN dimension
encoder_layer_1d = fingerprint_dim
encoder_layer_2d = fingerprint_dim
encoder_layer_3d = fingerprint_dim

mpn_bias = True
mpn_depth = 3
mpn_dropout_rate = 0.0

# 2) property prediction
property_dim = len(target_property_list)
property_weights = np.ones(shape=(property_dim,))
property_linear = True
property_network_hidden_sizes = [[] for _ in range(property_dim)]  # for linear property model
# property_network_hidden_sizes = [[32, 16, 8, 4, 2] for _ in range(property_dim)]  # for non-linear property model

# 3) decoder
decoder_num_layers = 4
decoder_num_attention_heads = 8
decoder_loss_weights = None  # (tensor): weights to use in cross entropy loss. default: None
decoder_add_latent = True  # (bool): whether to add the latent space embeddings to word embeddings. default: True

# training parameters
num_epochs = 60
batch_size = 32
weight_decay = 1e-4

# GPU
n_gpus = 2
n_trials = 16  # total trials for hyper-parameter tuning
gpu_idx_range = [2, 3]
############# MODIFY #############


class Objective:
    def __init__(self, gpu_queue):
        # Shared queue to manage GPU IDs.
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        # Fetch GPU ID for this trial.
        gpu_id = self.gpu_queue.get()

        # optimization
        with torch.cuda.device(gpu_id):
            # set torch details class
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

            # set hyperparameters
            divergence_weight = trial.suggest_float(name="divergence_weight", low=1e-1, high=10, log=True)
            latent_space_dim_quotient = trial.suggest_int(name="latent_space_dim_quotient", low=8, high=16)
            latent_space_dim = latent_space_dim_quotient * decoder_num_attention_heads  # even dim for positional encodings
            learning_rate = trial.suggest_float(name="learning_rate", low=1e-5, high=1e-2, log=True)

            integer_encoded_train_tensor = torch.tensor(integer_encoded_train, dtype=torch.long).to(device)  # long data type
            integer_encoded_valid_tensor = torch.tensor(integer_encoded_valid, dtype=torch.long).to(device)  # long data type
            y_train_tensor = torch.tensor(property_train_scaled, dtype=dtype).to(device)
            y_valid_tensor = torch.tensor(property_valid_scaled, dtype=dtype).to(device)

            # set VAE parameters
            vae_parameters = VAEParameters(
                save_directory=train_dir,
                eos_idx=eos_idx,
                latent_dimension=latent_space_dim,
                encoder_feature_dim=fingerprint_dim,
                encoder_layer_1d=encoder_layer_1d,
                encoder_layer_2d=encoder_layer_2d,
                encoder_layer_3d=encoder_layer_3d,
                mpn_bias=mpn_bias,
                mpn_depth=mpn_depth,
                mpn_dropout_rate=mpn_dropout_rate,
                decoder_num_layers=decoder_num_layers,
                decoder_num_attention_heads=decoder_num_attention_heads,
                max_length=len_max_molec,
                vocab=vocab,
                decoder_loss_weights=decoder_loss_weights,
                decoder_add_latent=decoder_add_latent,
                property_dim=property_dim,
                property_network_hidden_dim_list=property_network_hidden_sizes,
                property_weights=property_weights,
                property_linear=property_linear,
                dtype=dtype,
            )
            # set chemprop encoder
            chemprop_mpn = MoleculeModel(
                hidden_size=fingerprint_dim,
                bias=mpn_bias,
                depth=mpn_depth,
                device=device,
                dropout_rate=mpn_dropout_rate,
            )
            # set encoder
            encoder = Encoder(
                chemprop_mpn=chemprop_mpn,
                in_dimension=vae_parameters.encoder_feature_dim,
                layer_1d=vae_parameters.encoder_layer_1d,
                layer_2d=vae_parameters.encoder_layer_2d,
                layer_3d=vae_parameters.encoder_layer_3d,
                latent_dimension=vae_parameters.latent_dimension,
            ).to(device)
            total_params = sum(p.numel() for p in encoder.parameters())
            print(total_params)

            # set decoder
            decoder = SequenceDecoder(
                embedding_dim=vae_parameters.latent_dimension,
                decoder_num_layers=vae_parameters.decoder_num_layers,
                num_attention_heads=vae_parameters.decoder_num_attention_heads,
                max_length=vae_parameters.max_length[0],  # tuple(int)
                vocab=vae_parameters.vocab[0],  # tuple(dict)
                loss_weights=None,
                add_latent=True
            ).to(device)

            # model weight initialization - ONMT (3.5.1) adopts - "skip_init"
            for name, module in decoder.named_modules():
                for param_name, param in module.named_parameters():
                    if param_name == "weight" and param.dim() > 1:
                        xavier_uniform_(param)
                    elif param_name == "bias":
                        zeros_(param)
            total_params = sum(p.numel() for p in decoder.parameters())
            print(total_params)

            # set property prediction network
            property_network_module = PropertyNetworkPredictionModule(
                latent_dim=vae_parameters.latent_dimension,
                property_dim=vae_parameters.property_dim,
                property_network_hidden_dim_list=vae_parameters.property_network_hidden_dim_list,
                property_linear=vae_parameters.property_linear,
                dtype=vae_parameters.dtype,
                weights=vae_parameters.property_weights
            ).to(device)
            total_params = sum(p.numel() for p in property_network_module.parameters())
            print(total_params)

            # training
            model_save_directory = os.path.join(
                train_dir,
                f'divergence_weight_{divergence_weight:.3f}_latent_dim_{latent_space_dim}_learning_rate_{learning_rate:.3f}'
            )
            Path(model_save_directory).mkdir(parents=True, exist_ok=True)

            # check point dir
            check_point_dir = Path(os.path.join(model_save_directory, 'check_point'))
            check_point_dir.mkdir(parents=True, exist_ok=True)

            print("start training", flush=True)
            start = time.time()
            reconstruction_loss, divergence_loss, property_loss = train_model(
                vae_encoder=encoder,
                vae_decoder=decoder,
                property_predictor=property_network_module,
                data_train=train_data_chemprop,
                data_valid=valid_data_chemprop,
                integer_encoded_train=integer_encoded_train_tensor,
                integer_encoded_valid=integer_encoded_valid_tensor,
                y_train=y_train_tensor,
                y_valid=y_valid_tensor,
                y_scaler=property_scaler,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr_enc=learning_rate,
                lr_dec=learning_rate,
                lr_property=learning_rate,
                weight_decay=weight_decay,
                mmd_weight=divergence_weight,
                save_directory=model_save_directory,
                check_point_dir=check_point_dir,
                grammar_fragment=grammar_fragment,
                dtype=dtype,
                device=device
            )

            # save parameter
            save_parameter_dict = dict()
            info = vars(vae_parameters)
            for key, value in info.items():
                save_parameter_dict[key] = value

            torch.save(save_parameter_dict, os.path.join(model_save_directory, 'vae_parameters.pth'))

            end = time.time()
            print("total %.3f minutes took for training" % ((end - start) / 60.0), flush=True)

            # return GPU ID to the queue
            time.sleep(random.randint(1, 8))
            self.gpu_queue.put(gpu_id)

            return reconstruction_loss + divergence_loss + property_loss


def tune_hyperparameters():
    """
    Tune hyperparmeters using Optuna
    https://github.com/optuna/optuna/issues/1365
    """
    study = optuna.create_study(
        direction='minimize'
    )

    # parallel optimization
    with Manager() as manager:
        # Initialize the queue by adding available GPU IDs.
        gpu_queue = manager.Queue()
        for i in gpu_idx_range:
            gpu_queue.put(i)

        # debug
        with parallel_backend("dask", n_jobs=n_gpus):
            study.optimize(
                func=Objective(gpu_queue),
                n_trials=n_trials,
                n_jobs=n_gpus
            )

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ", flush=True)
    print("  Number of finished trials: ", len(study.trials), flush=True)
    print("  Number of pruned trials: ", len(pruned_trials), flush=True)
    print("  Number of complete trials: ", len(complete_trials), flush=True)

    print("Best trial:", flush=True)
    trial = study.best_trial

    print("  Value: ", trial.value, flush=True)

    print("  Params: ", flush=True)
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value), flush=True)

    # Show results.
    print(study.trials_dataframe().to_string(), flush=True)


if __name__ == '__main__':
    """
    This script train VAE with Group SELFIES tokens. 
    """
    # set clients for GPU
    # cluster = LocalCUDACluster()
    # client = Client(cluster)

    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    print('Loading Group SELFIES ...', flush=True)
    integer_encoded_arr = torch.load(os.path.join(data_dir, 'integer_encoded.pth'))  # (number of data, number of length)
    encoding_alphabet = torch.load(os.path.join(data_dir, 'encoding_alphabet.pth'))
    train_monomer_bags_idx = torch.load(os.path.join(data_dir, 'train_idx.pth'))
    valid_monomer_bags_idx = torch.load(os.path.join(data_dir, 'valid_idx.pth'))
    test_monomer_bags_idx = torch.load(os.path.join(data_dir, 'test_idx.pth'))
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))

    # create group grammar for inference during training
    vocab_save_path = os.path.join(data_dir, 'group_selfies_vocab.txt')
    grammar_fragment = GroupGrammar.from_file(vocab_save_path)

    # create vocab
    vocab = {alphabet: idx for idx, alphabet in enumerate(encoding_alphabet)}

    # set parameters
    len_max_molec = integer_encoded_arr.shape[1]  # maximum length -> "not" excluding the first asterisk

    # find idx
    eos_idx = 0

    # find '[EOS]'
    for idx, component in enumerate(encoding_alphabet):
        if component == '[EOS]':
            eos_idx = idx

    print(f'EOS idx is {eos_idx}', flush=True)

    # get molecular structures
    df = pd.read_csv(os.path.join(data_dir, 'two_reactants.csv'))
    df_train = df.iloc[train_monomer_bags_idx]
    df_valid = df.iloc[valid_monomer_bags_idx]

    # data loader for chemprop
    print('Loading data for Chemprop ...', flush=True)
    train_data_chemprop = get_data(smi_list=[[smi] for smi in df_train['methyl_terminated_product'].tolist()])
    train_data_chemprop.reset_features_and_targets()  # before doing training

    valid_data_chemprop = get_data(smi_list=[[smi] for smi in df_valid['methyl_terminated_product'].tolist()])
    valid_data_chemprop.reset_features_and_targets()  # before doing training

    integer_encoded_train = integer_encoded_arr[train_monomer_bags_idx]
    integer_encoded_valid = integer_encoded_arr[valid_monomer_bags_idx]
    property_train = property_value[train_monomer_bags_idx]
    property_valid = property_value[valid_monomer_bags_idx]

    # property scaling
    property_scaler = StandardScaler()
    property_train_scaled = property_scaler.fit_transform(property_train)
    property_valid_scaled = property_scaler.transform(property_valid)

    # save property scaler
    joblib.dump(property_scaler, os.path.join(data_dir, 'property_scaler.pkl'))
    tune_hyperparameters()
