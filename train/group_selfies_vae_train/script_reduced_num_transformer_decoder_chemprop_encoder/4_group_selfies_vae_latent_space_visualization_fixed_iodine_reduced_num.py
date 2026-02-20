import os
import sys
import time
import random
import json

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

import torch

from math import ceil
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from typing import Tuple, List
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML')

from vae.decoder_transformer.torch import SequenceDecoder
from vae.encoder_chemprop.chemprop_encoder import Encoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.utils.evaluation import tokenids_to_vocab

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies')
from group_selfies import GroupGrammar

CHEMPROP_PATH = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/vae/encoder_chemprop/chemprop'
sys.path.append(CHEMPROP_PATH)
from chemprop.data import get_data, MoleculeDataset
from chemprop.models import MoleculeModel


def reconstruction(data_chemprop, integer_encoded_tensor, reconstruct_type='Train', batch_size=100) -> Tuple[List, List, List, List]:
    """
    This function performs reconstruction.
    :param data_chemprop: data for the molecules (chemprop)
    :param integer_encoded_tensor: integer encoded tensor.
    :param reconstruct_type: Train, Valid, or Test to print out
    :param batch_size: batch size to be used for reconstruction
    :return: z_mean, property_prediction, answer_polymer, wrong_prediction_polymer
    """
    # for return
    z_mean, property_prediction, answer_polymer, wrong_prediction_polymer = [], [], [], []
    count_of_reconstruction = 0

    # reconstruct
    reconstruct_batch_iteration = ceil(len(data_chemprop) / batch_size)
    with torch.no_grad():
        tqdm.write(f"{reconstruct_type} Reconstruction ...")
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
                integer_encoded_batch = integer_encoded_tensor[start_idx: stop_idx]
                num_molecules_batch = stop_idx - start_idx

                # encode
                latent_points, mus, log_vars = vae_encoder(data_batch)

                # store
                z_mean += mus.cpu().tolist()

                # property_prediction
                property_prediction += property_predictor(mus).cpu().tolist()

                # inference with a trained decoder (beam search)
                beam_size = 1
                predictions = vae_decoder.inference(latent_points, beam_size=beam_size)

                # decoded smiles (converted from tokens)
                decoded_molecules = [
                    tokenids_to_vocab(predictions[sample][0].tolist(), vae_decoder.vocab, grammar_fragment, convert_to_smi=True) for sample in range(num_molecules_batch)
                ]
                answer_molecules = [
                    tokenids_to_vocab(integer_encoded_batch[sample].tolist(), vae_decoder.vocab, grammar_fragment, convert_to_smi=True) for sample in range(num_molecules_batch)
                ]

                # count right & wrong molecules
                for count_idx in range(num_molecules_batch):
                    decoded_molecule = decoded_molecules[count_idx]
                    answer_molecule = answer_molecules[count_idx]
                    if decoded_molecule == answer_molecule:
                        count_of_reconstruction += 1
                    else:
                        wrong_prediction_polymer.append(decoded_molecule)
                        answer_polymer.append(answer_molecule)

        tqdm.write(f'{reconstruct_type} reconstruction rate is {(100 * (count_of_reconstruction / len(data_chemprop))):.3f}%')

        return z_mean, property_prediction, answer_polymer, wrong_prediction_polymer


if __name__ == '__main__':
    """
    This script analyzes the training results from Group SELFIES. Don't need to locate '[I]' at the beginning of generated Group SELFIES.
    This doesn't exclude the first asterisk. 
    """
    ######### MODIFY #########
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # property name
    property_columns_Boltzmann_average = ['normalized_monomer_phi'] + \
                                         ['HOMO', 'LUMO', 'normalized_polarizability'] + \
                                         ['s1_energy', 'dominant_transition_energy',
                                          'dominant_transition_oscillator_strength', 't1_energy']
    property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']  # mean
    property_name = [f'{col}' for col in property_columns_Boltzmann_average] + [f'{col}' for col in property_columns_mean]
    property_label = ['$\Phi_{mon}$'] + \
                     ['$E_{HOMO}$', '$E_{LUMO}$', '$\\alpha$'] + \
                     ['$E_{S_1}$', "$E^{\prime}_{singlet}$", "$f^{\prime}_{osc}$", '$E_{T_1}$'] + \
                     ['$\chi_{water}$', '$\chi_{ethanol}$', '$\chi_{chloroform}$']

    # load vae parameters (linear)
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

    # number of train samples to use
    num_train_samples = 3000
    ######### MODIFY #########

    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    dtype = vae_parameters['dtype']

    # create group grammar
    vocab_save_path = os.path.join(data_dir, 'group_selfies_vocab.txt')
    grammar_fragment = GroupGrammar.from_file(vocab_save_path)

    # load SELFIES data - due to the random order
    encoding_alphabet = torch.load(os.path.join(data_dir, 'encoding_alphabet.pth'))
    integer_encoded_arr = torch.load(os.path.join(data_dir, 'integer_encoded.pth'))  # (number of data, number of length)

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

    # load data
    train_idx_for_property_scaling = torch.load(os.path.join(data_dir, 'train_idx.pth'))  # whole train_idx for train property scaling
    # random.seed(42)
    # train_idx = random.sample(train_idx_for_property_scaling, num_train_samples)  # subset to visualize the latent space
    train_idx = train_idx_for_property_scaling
    torch.save(train_idx, os.path.join(data_dir, f'train_idx_randomly_sampled_for_latent_space_visualization.pth'))  # save
    valid_idx = torch.load(os.path.join(data_dir, 'valid_idx.pth'))
    test_idx = torch.load(os.path.join(data_dir, 'test_idx.pth'))

    # preparation training data
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))

    # load df
    df = pd.read_csv(os.path.join(data_dir, 'two_reactants.csv'))
    df_train = df.iloc[train_idx]
    df_valid = df.iloc[valid_idx]
    df_test = df.iloc[test_idx]

    # data loader for chemprop
    print('Loading data for Chemprop ...', flush=True)

    train_data_chemprop = get_data(smi_list=[[smi] for smi in df_train['methyl_terminated_product'].tolist()])
    train_data_chemprop.reset_features_and_targets()  # reset

    valid_data_chemprop = get_data(smi_list=[[smi] for smi in df_valid['methyl_terminated_product'].tolist()])
    valid_data_chemprop.reset_features_and_targets()  # reset

    test_data_chemprop = get_data(smi_list=[[smi] for smi in df_test['methyl_terminated_product'].tolist()])
    test_data_chemprop.reset_features_and_targets()  # reset

    integer_encoded_train = integer_encoded_arr[train_idx]
    integer_encoded_valid = integer_encoded_arr[valid_idx]
    integer_encoded_test = integer_encoded_arr[test_idx]

    property_train_for_property_scaling = property_value[train_idx_for_property_scaling]
    property_train = property_value[train_idx]
    property_valid = property_value[valid_idx]
    property_test = property_value[test_idx]

    # property scaling
    property_scaler = StandardScaler()
    property_scaler = property_scaler.fit(property_train_for_property_scaling)
    property_train_scaled = property_scaler.transform(property_train)
    property_valid_scaled = property_scaler.transform(property_valid)
    property_test_scaled = property_scaler.transform(property_test)

    # set property prediction network
    property_predictor = PropertyNetworkPredictionModule(
        latent_dim=vae_parameters['latent_dimension'],
        property_dim=vae_parameters['property_dim'],
        property_network_hidden_dim_list=vae_parameters['property_network_hidden_dim_list'],
        property_linear=vae_parameters['property_linear'],
        dtype=dtype,
        weights=vae_parameters['property_weights']
    ).to(device)

    integer_encoded_train_tensor = torch.tensor(integer_encoded_train, dtype=torch.long).to(device)
    integer_encoded_valid_tensor = torch.tensor(integer_encoded_valid, dtype=torch.long).to(device)
    integer_encoded_test_tensor = torch.tensor(integer_encoded_test, dtype=torch.long).to(device)

    # load state dictionary
    vae_encoder.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'encoder.pth'), map_location=device))
    vae_decoder.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'decoder.pth'), map_location=device))
    property_predictor.load_state_dict(torch.load(os.path.join(model_check_point_directory, 'property_predictor.pth'), map_location=device))

    # eval
    vae_encoder.eval()
    vae_decoder.eval()
    property_predictor.eval()

    # check reconstruction
    batch_size = 100
    z_mean_train, train_property_prediction, _, _ = reconstruction(train_data_chemprop, integer_encoded_train_tensor, reconstruct_type='Train', batch_size=batch_size)
    z_mean_valid, valid_property_prediction, _, _ = reconstruction(valid_data_chemprop, integer_encoded_valid_tensor, reconstruct_type='Valid', batch_size=batch_size)
    z_mean_test, test_property_prediction, _, _ = reconstruction(test_data_chemprop, integer_encoded_test_tensor, reconstruct_type='Test', batch_size=batch_size)

    # plot property prediction
    y_train_true = property_train
    y_valid_true = property_valid
    y_test_true = property_test

    y_train_prediction = property_scaler.inverse_transform(train_property_prediction)
    y_valid_prediction = property_scaler.inverse_transform(valid_property_prediction)
    y_test_prediction = property_scaler.inverse_transform(test_property_prediction)

    # get r2 score & plot
    pred_plot_dir = os.path.join(model_check_point_directory, 'property_pred')
    Path(pred_plot_dir).mkdir(parents=True, exist_ok=True)
    for idx in range(vae_parameters['property_dim']):
        # scatter plot
        g = sns.JointGrid()

        # train plot df & scatter plot & kde plots on the marginal axes
        train_color = '#3E6AE5'
        train_true, train_pred = y_train_true[:, idx], y_train_prediction[:, idx]
        train_plot_df = pd.DataFrame({'true': train_true, 'pred': train_pred})
        train_r2_score = r2_score(y_true=train_true, y_pred=train_pred)
        train_rmse = root_mean_squared_error(y_true=train_true, y_pred=train_pred)
        ax = sns.scatterplot(x='true', y='pred', data=train_plot_df, ax=g.ax_joint,
                             edgecolor=None, alpha=0.5, legend=False, size=3.0)
        sns.kdeplot(x='true', data=train_plot_df, ax=g.ax_marg_x, fill=True, color=train_color)
        sns.kdeplot(y='pred', data=train_plot_df, ax=g.ax_marg_y, fill=True, color=train_color)

        # valid plot df
        valid_color = '#219B35'
        valid_true, valid_pred = y_valid_true[:, idx], y_valid_prediction[:, idx]
        valid_plot_df = pd.DataFrame({'true': valid_true, 'pred': valid_pred})
        valid_r2_score = r2_score(y_true=valid_true, y_pred=valid_pred)
        valid_rmse = root_mean_squared_error(y_true=valid_true, y_pred=valid_pred)
        ax = sns.scatterplot(x='true', y='pred', data=valid_plot_df, ax=g.ax_joint,
                             edgecolor=None, alpha=0.5, legend=False, size=3.0)
        sns.kdeplot(x='true', data=valid_plot_df, ax=g.ax_marg_x, fill=True, color=valid_color)
        sns.kdeplot(y='pred', data=valid_plot_df, ax=g.ax_marg_y, fill=True, color=valid_color)

        # test plot df
        test_color = '#ED3B6F'
        test_true, test_pred = y_test_true[:, idx], y_test_prediction[:, idx]
        test_plot_df = pd.DataFrame({'true': test_true, 'pred': test_pred})
        test_r2_score = r2_score(y_true=test_true, y_pred=test_pred)
        test_rmse = root_mean_squared_error(y_true=test_true, y_pred=test_pred)
        ax = sns.scatterplot(x='true', y='pred', data=test_plot_df, ax=g.ax_joint,
                             edgecolor=None, alpha=0.5, legend=False, size=3.0)
        sns.kdeplot(x='true', data=test_plot_df, ax=g.ax_marg_x, fill=True, color=test_color)
        sns.kdeplot(y='pred', data=test_plot_df, ax=g.ax_marg_y, fill=True, color=test_color)

        # tick parameters
        plot_label = property_label[idx]
        g.ax_joint.tick_params(labelsize=16)
        g.set_axis_labels(f'True {plot_label}', f'Prediction {plot_label}', fontsize=16)

        # y=x
        plot_true_min = np.min([np.min(train_true), np.min(valid_true), np.min(test_true)])
        plot_true_max = np.max([np.max(train_true), np.max(valid_true), np.max(test_true)])
        plot_pred_min = np.min([np.min(train_pred), np.min(valid_pred), np.min(test_pred)])
        plot_pred_max = np.max([np.max(train_pred), np.max(valid_pred), np.max(test_pred)])
        plot_min = np.min([plot_true_min, plot_pred_min])
        plot_max = np.max([plot_true_max, plot_pred_max])
        g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], color='#665A5E', linestyle='--', linewidth=1.5)

        # Create custom legend handles with different colors
        train_handle = plt.Line2D([], [], color=train_color, label=f'Train $R^2$: {train_r2_score:.2f}', marker='o', linestyle='None')
        valid_handle = plt.Line2D([], [], color=valid_color, label=f'Valid $R^2$: {valid_r2_score:.2f}', marker='o', linestyle='None')
        test_handle = plt.Line2D([], [], color=test_color, label=f'Test $R^2$: {test_r2_score:.2f}', marker='o', linestyle='None')

        # Add the custom handles to the legend
        g.ax_joint.legend(handles=[train_handle, valid_handle, test_handle], handletextpad=0.1,
                          fontsize=14, loc='upper left')

        # tight layout & save
        fig_save_dir = Path(os.path.join(pred_plot_dir, plot_label))
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f"recovered_property_prediction.png"))
        plt.close()

        # save RMSE
        rmse_dict = {'Train RMSE': train_rmse, 'Valid RMSE': valid_rmse, 'Test RMSE': test_rmse}
        with open(os.path.join(fig_save_dir, 'rmse.json'), 'w') as f:
            json.dump(rmse_dict, f)
            f.close()

        # save arr
        np.save(os.path.join(fig_save_dir, 'train_true_arr_inverse_transformed_randomly_sampled.npy'), train_true)
        np.save(os.path.join(fig_save_dir, 'train_pred_arr_inverse_transformed_randomly_sampled.npy'), train_pred)
        np.save(os.path.join(fig_save_dir, 'valid_true_arr_inverse_transformed.npy'), valid_true)
        np.save(os.path.join(fig_save_dir, 'valid_pred_arr_inverse_transformed.npy'), valid_pred)
        np.save(os.path.join(fig_save_dir, 'test_true_arr_inverse_transformed.npy'), test_true)
        np.save(os.path.join(fig_save_dir, 'test_pred_arr_inverse_transformed.npy'), test_pred)

    # convert to numpy
    z_mean = np.array(z_mean_train)
    property_value = np.array(y_train_prediction)

    # PCA
    pca_dir = os.path.join(model_check_point_directory, 'pca')
    Path(pca_dir).mkdir(parents=True, exist_ok=True)
    n_components = 64
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(z_mean)

    principal_df = pd.DataFrame(
        data=principal_components,
        columns=['PC%d' % (num + 1) for num in range(n_components)]
    )
    print('PCA is done', flush=True)

    # plot explained variance by principal component analysis
    exp_var_pca = pca.explained_variance_ratio_
    plt.bar(range(1, len(exp_var_pca) + 1), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(pca_dir, 'Explained_ratio.png'), dpi=300)
    plt.show()
    plt.close()

    # PCA plot for only one property
    for property_idx, target_property in enumerate(property_name):
        # PC1 and PC2
        g = sns.JointGrid(x='PC1', y='PC2', data=principal_df)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#2F65ED", "#F5EF7E", "#F89A3F"])
        principal_df[f'property_{property_idx}_z_mean'] = property_value[:, property_idx]  # add properties

        # scatter plot
        sns.scatterplot(x='PC1', y='PC2', hue=f'property_{property_idx}_z_mean', data=principal_df, palette=cmap,
                        ax=g.ax_joint,
                        edgecolor=None, alpha=0.75, legend=False, size=3.0)

        # kde plots on the marginal axes
        sns.kdeplot(x='PC1', data=principal_df, ax=g.ax_marg_x, fill=True, color='#4DC3C8')
        sns.kdeplot(y='PC2', data=principal_df, ax=g.ax_marg_y, fill=True, color='#4DC3C8')

        # tick parameters
        g.ax_joint.tick_params(labelsize=16)
        g.set_axis_labels('Principal component 1', 'Principal component 2', fontsize=16)
        g.fig.tight_layout()

        # color bars
        norm = plt.Normalize(principal_df[f'property_{property_idx}_z_mean'].min(),
                             principal_df[f'property_{property_idx}_z_mean'].max())
        scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        # Make space for the colorbar
        g.fig.subplots_adjust(bottom=0.2)
        cax = g.fig.add_axes([0.18, 0.08, 0.6, 0.02])  # l, b, w, h
        cbar = g.fig.colorbar(scatter_map, cax=cax, orientation='horizontal')
        cbar.set_label(label=property_name[property_idx], size=12)
        cbar.ax.tick_params(labelsize=12)
        g.fig.savefig(os.path.join(pca_dir, f'latent_space_1_2_property_{property_idx}.png'), dpi=800)
        plt.close()

    # save mean
    np.save(os.path.join(pca_dir, 'z_mean.npy'), z_mean)
    np.save(os.path.join(pca_dir, 'property_z_mean.npy'), property_value)
