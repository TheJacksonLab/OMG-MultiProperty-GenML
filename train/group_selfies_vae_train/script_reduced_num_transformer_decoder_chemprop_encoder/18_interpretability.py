import os
import sys
import numpy as np
import pandas as pd

import seaborn as sns

from rdkit import Chem
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies')
from group_selfies import GroupGrammar

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML')
from vae.decoder_transformer.torch import SequenceDecoder
from vae.encoder_chemprop.chemprop_encoder import Encoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from utils.functions import terminate_p_smi_with_CH3, encoding_return_scaled_mean_prediction

CHEMPROP_PATH = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/vae/encoder_chemprop/chemprop'
sys.path.append(CHEMPROP_PATH)
from chemprop.data import get_data, MoleculeDataset
from chemprop.models import MoleculeModel


def black_box_function(valid_polymer_list, vae_encoder, vae_decoder, property_predictor, grammar_fragment, return_mus=False):
    """
    Black-box functions to model.
    :params valid_polymer_list: list containing p-smiles. use to predict their properties with encoder, decoder, and property prediction model
    :return predicted properties
    """
    # re-encoding strategy
    methyl_terminated_smi_list = [Chem.MolToSmiles(terminate_p_smi_with_CH3(product_smi)) for product_smi in valid_polymer_list]
    data_chemprop = get_data(smi_list=[[methyl_terminated_smi] for methyl_terminated_smi in methyl_terminated_smi_list])
    data_chemprop.reset_features_and_targets()  # reset

    re_encoding_property_prediction_arr, _ = encoding_return_scaled_mean_prediction(
        data_chemprop, valid_polymer_list, vae_encoder, vae_decoder,
        property_predictor, grammar_fragment, batch_size=100
    )  # (1, num_data, num_property)

    # get reference point
    if return_mus:
        vae_encoder.eval()
        with torch.no_grad():
            latent_points, mus, log_vars = vae_encoder(data_chemprop)
            mus_arr = mus.cpu().numpy()
        return re_encoding_property_prediction_arr, mus_arr

    else:
        return re_encoding_property_prediction_arr


if __name__ == '__main__':
    """
    This script investigates the interpretability of the linearly organized latent space by identifying the trained
    OMG CRUs near the generated polymer.
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
    plot_dir = Path(os.path.join(model_check_point_directory, 'interpretability'))
    plot_dir.mkdir(exist_ok=True, parents=True)

    # load data
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    data_path = os.path.join(data_dir, 'two_reactants.csv')
    train_idx_path = os.path.join(data_dir, 'train_idx.pth')

    # property idx & property name
    monomer_phi_property_idx = 0
    lumo_energy_property_idx = 2
    homo_energy_property_idx = 1
    chi_water_property_idx = 8
    property_name = ['monomer_phi', 'homo_lumo_gap', 'chi_water']

    # polymers to decode (linear) - found molecules for Figure 5
    polymers_to_decode = [
        '*OCCN(CCOC(=O)C(CCCc1ccsc1)C(*)=O)c1ccccn1',  # molecule 1
        '*OCC(C)(COC(=O)C(CCCc1ccsc1)C(*)=O)c1ccccn1',  # molecule 2
        '*OCC(COC(=O)C(CCc1ccccc1)C(*)=O)c1ccc(C(C)(C)C)cc1',  # molecule 3
        '*Oc1cc(C)cc(OC(=O)C(CC)(C(*)=O)c2ccccc2)c1C'  # molecule 4
    ]

    # generated idx to use - this idx corresponds to idx for polymers_to_decode
    reference_polymer_idx = 0  # to calculate optimization score for plot
    assert reference_polymer_idx == 0

    # eigen cut off and navigation vector to use
    eigen_cut_off = 36  # from the previous script (cnt from 7_group_selfies_vae_prepare_data_for_3D_landscape.py)
    use_navigation_vector = False

    # draw option
    scale_factor = 10

    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # optimization metric
    use_std_distance = False  # based on std to define the optimization score
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
    df_train = df_polymer.iloc[train_idx]
    train_z_mean = np.load(os.path.join(model_check_point_directory, 'pca', 'z_mean.npy'))

    # load SELFIES data - due to the random order
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

    # state dict for property prediction network
    state_dict = torch.load(os.path.join(model_check_point_directory, 'property_predictor.pth'), map_location=device)

    # preparation training data for scaling
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    train_idx_for_property_scaling = torch.load(os.path.join(data_dir, 'train_idx.pth'))  # whole train_idx for train property scaling

    # preparation training data for scaling
    property_train_for_property_scaling = property_value[train_idx_for_property_scaling]
    property_train_scaler = StandardScaler()
    property_train_scaler = property_train_scaler.fit(property_train_for_property_scaling)

    # eval
    vae_encoder.eval()
    vae_decoder.eval()
    property_predictor.eval()

    # (scaled) property prediction
    re_encoding_property_prediction_arr_scaled, mu_arr = black_box_function(
        valid_polymer_list=polymers_to_decode,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        property_predictor=property_predictor,
        grammar_fragment=grammar_fragment,
        return_mus=True
    )  # (1, num_data, num_property)

    re_encoding_property_prediction_arr = property_train_scaler.inverse_transform(re_encoding_property_prediction_arr_scaled[0])
    monomer_phi_arr = re_encoding_property_prediction_arr[:, monomer_phi_property_idx]
    lumo_energy_arr = re_encoding_property_prediction_arr[:, lumo_energy_property_idx]
    homo_energy_arr = re_encoding_property_prediction_arr[:, homo_energy_property_idx]
    chi_water_arr = re_encoding_property_prediction_arr[:, chi_water_property_idx]

    # PCA for the trained polymers
    pca_dir = os.path.join(model_check_point_directory, 'pca')
    z_mean_train = np.load(os.path.join(pca_dir, 'z_mean.npy'))
    n_components = 64  # the same dimension as non-linear
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(z_mean_train)

    # pca projection for whole polymers to decode
    PCA_projection_generated_whole = pca.transform(mu_arr)

    # get a delta z vector
    delta_z_train_to_reference = z_mean_train - mu_arr[reference_polymer_idx]
    delta_z_generated_to_reference = mu_arr[1:] - mu_arr[reference_polymer_idx]  # without reference polymer

    # load property data
    property_train = property_value[train_idx]

    # train arr
    monomer_phi_arr_train = df_train['normalized_monomer_phi_Boltzmann_average'].to_numpy()
    homo_lumo_gap_arr_train = df_train['LUMO_Boltzmann_average'].to_numpy() - df_train['HOMO_Boltzmann_average'].to_numpy()
    chi_water_arr_train = df_train['chi_parameter_water_mean'].to_numpy()

    # get standard deviation
    monomer_phi_arr_train_std = np.std(monomer_phi_arr_train)
    homo_lumo_gap_arr_train_std = np.std(homo_lumo_gap_arr_train)
    chi_water_arr_train_std = np.std(chi_water_arr_train)

    ### get directions
    # 1) monomer phi direction
    monomer_phi_direction_vector = state_dict[f'property_network_module.{monomer_phi_property_idx}.weight'].cpu().numpy().squeeze()

    # 2) HOMO-LUMO gap direction
    std_LUMO = property_train_scaler.scale_[lumo_energy_property_idx]
    std_HOMO = property_train_scaler.scale_[homo_energy_property_idx]
    homo_lumo_gap_property_direction_vector = std_LUMO * state_dict[f'property_network_module.{lumo_energy_property_idx}.weight'].cpu().numpy().squeeze() - std_HOMO * state_dict[f'property_network_module.{homo_energy_property_idx}.weight'].cpu().numpy().squeeze()
    homo_lumo_gap_property_direction_vector /= homo_lumo_gap_arr_train_std  # to get a standardized direction

    # 3) Chi water direction
    chi_water_direction_vector = state_dict[f'property_network_module.{chi_water_property_idx}.weight'].cpu().numpy().squeeze()

    # navigation vector for multi-objective optimization (simultaneous increase of Tg, band gap, and chi water)
    ### MODIFY based on a target property optimization ###
    navigation_vector = monomer_phi_direction_vector * (-1) + homo_lumo_gap_property_direction_vector + chi_water_direction_vector
    # navigation_vector = monomer_phi_direction_vector * (-1)
    # navigation_vector = homo_lumo_gap_property_direction_vector
    # navigation_vector = chi_water_direction_vector
    ###

    # calculate a dot product with navigation vector
    dot_product_train_to_navigation_vector = np.dot(delta_z_train_to_reference, navigation_vector)
    dot_product_generated_to_navigation_vector = np.dot(delta_z_generated_to_reference, navigation_vector)

    # get a meaningful subspace relevant to the multi-objective optimization
    if use_navigation_vector:
        inner_product_arr = np.zeros(shape=(eigen_cut_off,))
        for eigen_idx, eigenvector in enumerate(pca.components_[:eigen_cut_off]):
            inner_product = abs(np.dot(navigation_vector, eigenvector))  # absolute values
            inner_product_arr[eigen_idx] = inner_product

        # for visualization
        sorted_eigen_idx = np.argsort(inner_product_arr)
        most_relevant_pca_axis_idx = sorted_eigen_idx[-1]
        next_relevant_pca_axis_idx = sorted_eigen_idx[-2]
        relevant_pc_total = principal_components[:, np.arange(eigen_cut_off)[[most_relevant_pca_axis_idx, next_relevant_pca_axis_idx]]]

    else:  # not use a navigation vector (as a baseline)
        # PC1 and PC2
        most_relevant_pca_axis_idx = 0  # start from 0
        next_relevant_pca_axis_idx = 1  # start from 0
        relevant_pc_total = principal_components[:, [most_relevant_pca_axis_idx, next_relevant_pca_axis_idx]]

    # PCA df
    df_train = df_train.reset_index(drop=True)
    df_PCA = pd.DataFrame(columns=[f'PC{most_relevant_pca_axis_idx + 1}', f'PC{next_relevant_pca_axis_idx + 1}'],
                          data=relevant_pc_total)  # idx from 1
    df_plot = pd.concat([df_train, df_PCA], axis=1)

    # get a reference value
    # df_reference = df_plot[df_plot['product'] == polymers_to_decode[reference_polymer_idx]]
    # monomer_phi_reference = df_reference['normalized_monomer_phi_Boltzmann_average'].values[0]
    # homo_lumo_gap_reference = df_reference['LUMO_Boltzmann_average'].values[0] - df_reference['HOMO_Boltzmann_average'].values[0]
    # chi_water_reference = df_reference['chi_parameter_water_mean'].values[0]

    # get a reference value - to be consistent with BO search
    monomer_phi_reference = monomer_phi_arr[reference_polymer_idx]
    homo_lumo_gap_reference = (lumo_energy_arr - homo_energy_arr)[reference_polymer_idx]
    chi_water_reference = chi_water_arr[0]

    # correlation between optimization score and dot product
    assert reference_polymer_idx == 0
    monomer_phi_generated_whole = monomer_phi_arr[1:]
    homo_lumo_gap_generated_whole = (lumo_energy_arr - homo_energy_arr)[1:]
    chi_water_generated_whole = chi_water_arr[1:]

    # optimization score - based on the reference value
    if not use_std_distance:
        monomer_phi_optimization_score = (-1) * (monomer_phi_arr_train - monomer_phi_reference) / monomer_phi_reference * 100
        homo_lumo_gap_optimization_score = (homo_lumo_gap_arr_train - homo_lumo_gap_reference) / homo_lumo_gap_reference * 100
        chi_water_optimization_score = (chi_water_arr_train - chi_water_reference) / chi_water_reference * 100
        mean_optimization_score = np.array([monomer_phi_optimization_score, homo_lumo_gap_optimization_score, chi_water_optimization_score]).mean(axis=0)

        monomer_phi_optimization_score_generated_whole = (-1) * (monomer_phi_generated_whole - monomer_phi_reference) / monomer_phi_reference * 100
        homo_lumo_gap_optimization_score_generated_whole = (homo_lumo_gap_generated_whole - homo_lumo_gap_reference) / homo_lumo_gap_reference * 100
        chi_water_optimization_score_generated_whole = (chi_water_generated_whole - chi_water_reference) / chi_water_reference * 100

    else:  # std distance
        # optimization score - based on the standard deviation
        monomer_phi_optimization_score = (-1) * (monomer_phi_arr_train - monomer_phi_reference) / monomer_phi_arr_train_std
        homo_lumo_gap_optimization_score = (homo_lumo_gap_arr_train - homo_lumo_gap_reference) / homo_lumo_gap_arr_train_std
        chi_water_optimization_score = (chi_water_arr_train - chi_water_reference) / chi_water_arr_train_std
        mean_optimization_score = np.array([monomer_phi_optimization_score, homo_lumo_gap_optimization_score, chi_water_optimization_score]).mean(axis=0)

        # optimization score for generated polymers based on the standard deviation
        monomer_phi_optimization_score_generated_whole = (-1) * (monomer_phi_generated_whole - monomer_phi_reference) / monomer_phi_arr_train_std
        homo_lumo_gap_optimization_score_generated_whole = (homo_lumo_gap_generated_whole - homo_lumo_gap_reference) / homo_lumo_gap_arr_train_std
        chi_water_optimization_score_generated_whole = (chi_water_generated_whole - chi_water_reference) / chi_water_arr_train_std

    df_plot['mean_optimization_score'] = mean_optimization_score
    df_plot['monomer_phi_optimization_score'] = monomer_phi_optimization_score
    df_plot['homo_lumo_gap_optimization_score'] = homo_lumo_gap_optimization_score
    df_plot['chi_water_optimization_score'] = chi_water_optimization_score
    #
    # ### MODIFY based on a target property optimization ###
    mean_optimization_score_generated_whole = np.array([monomer_phi_optimization_score_generated_whole, homo_lumo_gap_optimization_score_generated_whole, chi_water_optimization_score_generated_whole]).mean(axis=0)
    # mean_optimization_score_generated_whole = monomer_phi_optimization_score_generated_whole
    # mean_optimization_score_generated_whole = homo_lumo_gap_optimization_score_generated_whole
    # mean_optimization_score_generated_whole = chi_water_optimization_score_generated_whole
    ###

    # print('Averaged optimization score for generated polymers')
    # print(mean_optimization_score_generated_whole)
    #
    #
    # ### MODIFY based on a target property optimization ###
    # linear_correlation, p_value = pearsonr(dot_product_train_to_navigation_vector, mean_optimization_score)
    # # linear_correlation, p_value = pearsonr(dot_product_train_to_navigation_vector, monomer_phi_optimization_score)
    # # linear_correlation, p_value = pearsonr(dot_product_train_to_navigation_vector, homo_lumo_gap_optimization_score)
    # # linear_correlation, p_value = pearsonr(dot_product_train_to_navigation_vector, chi_water_optimization_score)
    # ###
    #
    # df_plot['dot_product_train_to_navigation_vector'] = dot_product_train_to_navigation_vector
    # cmap = LinearSegmentedColormap.from_list("", ["#6296C4", "#F5EF7E", "#FE7F2D"])
    # g = sns.JointGrid()
    #
    # ### MODIFY based on a target property optimization ###
    # hue_property = 'mean_optimization_score'
    # # hue_property = 'monomer_phi_optimization_score'
    # # hue_property = 'homo_lumo_gap_optimization_score'
    # # hue_property = 'chi_water_optimization_score'
    # ###
    #
    # sns.scatterplot(x='dot_product_train_to_navigation_vector', y=hue_property, ax=g.ax_joint,
    #                 hue=hue_property, palette=cmap,
    #                 data=df_plot, edgecolor=None, alpha=0.4, legend=False, size=3.0)
    #
    # # Add a dummy handle so the legend shows your correlation text
    # handle = Line2D([], [], linestyle='none', marker='', color='none')
    # leg = g.ax_joint.legend(
    #     [handle],
    #     [f'Linear correlation ($\\rho$): {linear_correlation:.3f}'],
    #     loc='upper left',
    #     frameon=True,
    #     fancybox=True,  # rounded box
    #     handlelength=0.3,  # no space for handle
    #     handletextpad=0,  # no gap between handle and text
    #     borderpad=0.3,  # inner padding of the box (smaller = tighter)
    #     labelspacing=0,  # vertical spacing (only one line here)
    #     borderaxespad=0.8,  # gap between legend box and axes
    #     fontsize=13
    # )
    #
    # g.ax_joint.scatter(
    #     x=dot_product_generated_to_navigation_vector,
    #     y=mean_optimization_score_generated_whole,
    #     facecolor='#FF6B6B',  # interior color
    #     edgecolor='#21150A',  # edge color
    #     s=50.0,
    #     marker='X',
    #     alpha=1.0
    # )
    # g.ax_joint.scatter(
    #     x=[0],
    #     y=[0],
    #     facecolor='#21150A',  # interior color
    #     edgecolor='#21150A',  # edge color
    #     s=100.0,
    #     marker='*',
    #     alpha=1.0
    # )
    # if not use_std_distance:
    #     g.set_axis_labels(f'Alignment score (dot product)',
    #                       f'Optimization score (%)', fontsize=16)
    # else:  # std distance
    #     g.set_axis_labels(f'Alignment score (dot product)',
    #                       f'Optimization score (scaled)', fontsize=16)
    # g.ax_joint.tick_params(labelsize=15)
    # sns.kdeplot(x='dot_product_train_to_navigation_vector', data=df_plot, ax=g.ax_marg_x, fill=True, color='#CC97BD',
    #             alpha=0.1)
    # sns.kdeplot(y=hue_property, data=df_plot, ax=g.ax_marg_y, fill=True, color='#CC97BD',
    #             alpha=0.1)
    # plt.tight_layout()
    # ### MODIFY based on a target property optimization ###
    # plot_name = 'optimization_score_dot_product'
    # # plot_name = 'optimization_score_dot_product_monomer_phi'
    # # plot_name = 'optimization_score_dot_product_homo_lumo_gap'
    # # plot_name = 'optimization_score_dot_product_chi_water'
    # ###
    # if not use_std_distance:
    #     plt.savefig(os.path.join(plot_dir, f'{plot_name}.png'), dpi=300)
    # else:  # use std_distance
    #     plt.savefig(os.path.join(plot_dir, f'{plot_name}_std_distance.png'), dpi=300)
    # plt.close()

    # plot (optimization score) -> 2D projection
    hue_property = 'mean_optimization_score'
    cmap = LinearSegmentedColormap.from_list("", ["#6296C4", "#F5EF7E", "#FE7F2D"])
    g = sns.JointGrid(x=f'PC{most_relevant_pca_axis_idx + 1}', y=f'PC{next_relevant_pca_axis_idx + 1}', data=df_plot)
    sns.scatterplot(x=f'PC{most_relevant_pca_axis_idx + 1}', y=f'PC{next_relevant_pca_axis_idx + 1}',
                    hue=hue_property, data=df_plot, palette=cmap,
                    ax=g.ax_joint,
                    edgecolor=None, alpha=1.00, legend=False, size=3.0)

    # add arrow
    arrow_starting_point = np.array([
        PCA_projection_generated_whole[:, most_relevant_pca_axis_idx][reference_polymer_idx],
        PCA_projection_generated_whole[:, next_relevant_pca_axis_idx][reference_polymer_idx],
    ])
    unit_navigation_vector = navigation_vector / np.linalg.norm(navigation_vector)
    arrow_to_draw = unit_navigation_vector.reshape(1, -1) * scale_factor
    arrow_delta = pca.transform(arrow_to_draw)[:, [most_relevant_pca_axis_idx, next_relevant_pca_axis_idx]][0]

    # draw arrow move arrow to the center of the PCA plot
    arrow_color = '#4DB96B'  # kind a green
    g.ax_joint.arrow(
        x=arrow_starting_point[0],  # x-coordinate of the arrow's start
        y=arrow_starting_point[1],  # y-coordinate of the arrow's start
        dx=arrow_delta[0],  # Change in x (length of the arrow in x-direction)
        dy=arrow_delta[1],  # Change in y (length of the arrow in y-direction)
        head_width=0.5,  # Width of the arrow head
        head_length=0.7,  # Length of the arrow head
        linewidth=2.5,
        fc=arrow_color,  # Face color of the arrow
        ec=arrow_color  # Edge color of the arrow
    )

    # add starting point
    g.ax_joint.scatter(
        x=arrow_starting_point[0],
        y=arrow_starting_point[1],
        facecolor='#21150A',  # interior color
        edgecolor='#21150A',  # edge color
        s=100.0,
        marker='*',
        alpha=1.0
    )

    # kde plots on the marginal axes
    sns.kdeplot(x=f'PC{most_relevant_pca_axis_idx + 1}', data=df_plot, ax=g.ax_marg_x, fill=True, color='#CC97BD',
                alpha=0.1)
    sns.kdeplot(y=f'PC{next_relevant_pca_axis_idx + 1}', data=df_plot, ax=g.ax_marg_y, fill=True, color='#CC97BD',
                alpha=0.1)

    # add points
    print('==== Points to Draw on the PCA plot ====')
    print(PCA_projection_generated_whole[:, most_relevant_pca_axis_idx][1:])  # after the reference polymer
    print(PCA_projection_generated_whole[:, next_relevant_pca_axis_idx][1:])  # after the reference polymer
    g.ax_joint.scatter(
        x=PCA_projection_generated_whole[:, most_relevant_pca_axis_idx][1:],
        y=PCA_projection_generated_whole[:, next_relevant_pca_axis_idx][1:],
        facecolor='#FF6B6B',  # interior color
        edgecolor='#21150A',  # edge color
        s=100.0,
        marker='X',
        alpha=1.0
    )

    g.ax_joint.tick_params(labelsize=15)
    g.ax_joint.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    g.set_axis_labels(f'Principal component {most_relevant_pca_axis_idx + 1}',
                      f'Principal component {next_relevant_pca_axis_idx + 1}', fontsize=16)

    # color bars
    norm = plt.Normalize(df_plot[hue_property].min(), df_plot[hue_property].max())
    scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    g.fig.tight_layout()

    # Make space for the colorbar
    g.fig.subplots_adjust(bottom=0.2)
    cax = g.fig.add_axes([0.20, 0.06, 0.6, 0.02])  # l, b, w, h
    cbar = g.fig.colorbar(scatter_map, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    g.ax_joint.set_aspect('equal', adjustable='box')
    if not use_std_distance:
        # fig_save_name = 'optimization'
        fig_save_name = 'optimization_PC1_PC2'
    else:
        fig_save_name = 'optimization_std_distance'
    g.fig.savefig(
        os.path.join(plot_dir, f'{fig_save_name}.png'),
        dpi=600)
    plt.close()
