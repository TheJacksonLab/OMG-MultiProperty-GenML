import os
import torch
import matplotlib.pyplot as plt

import matplotlib.colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


if __name__ == '__main__':
    """
    This script visualizes the PCA of latent space with a property direction for training samples. 
    1) Monomer Phi index (relevant to Tg)
    2) Band gap (electronic property)
    3) Chi water (solubility).
    
    And use linear direction to further visualize a sub-PCA space for each target property.   
    
    "loc_dict" under draw_2D_latent_space_plot should be modified. 
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

    # NEEDED TO BE CHANGED - from group_selfies_vae_prepare_data_for_3D_landscape.py
    eigen_cut_off = 36  # (done)

    # scale factor to search for a molecule
    scale_factor = 10  # This scale factor is used in drawing a arrow. This corresponds to the length in the Z-space
    epsilon = 3.0  # epsilon for each z value to choose a reference point

    scale_factor_list = [1.0, 2.0]  # sigma for scaled property. scale factor to use while searching for a latent space

    # reference molecule idx to choose
    reference_molecule_loc = 161481

    # to use the following indices
    use_following_indices = True  # False -> use the first idx of the dataframe as the initial trial

    ### [Tg] molecule idx to show
    Tg_molecule_scale_1_loc = 580116
    Tg_molecule_scale_2_loc = 735106

    band_gap_molecule_scale_1_loc = 194086
    band_gap_molecule_scale_2_loc = 203832

    ### [Chi water] molecule idx to show
    chi_water_molecule_scale_1_loc = 753861
    chi_water_molecule_scale_2_loc = 21279

    ####### MODIFY #######

    #### STEP 1) Load relevant data & models
    # load VAE
    device = torch.device('cpu')
    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    dtype = vae_parameters['dtype']

    # state dict for property prediction network
    state_dict = torch.load(os.path.join(model_check_point_directory, 'property_predictor.pth'), map_location=device)

    # load principal df
    pca_results_dir = os.path.join(model_check_point_directory, 'pca')
    train_z_mean = np.load(os.path.join(pca_results_dir, 'z_mean.npy'))  # (num_data, latent_space_dim)
    principal_df = pd.read_csv(os.path.join(pca_results_dir, '3d_landscape', 'principal_df.csv'))

    # reset index
    principal_df.index = principal_df['Unnamed: 0'].to_list()

    #### STEP 2) Draw 3D PCA landscape
    # 1) choose a reference point based on z vector (zero vector)
    latent_space_dim = train_z_mean.shape[1]
    z_coordinates_name = [f'z_{num + 1}_mean' for num in range(latent_space_dim)]

    target_reference_z_point = np.zeros_like(train_z_mean[0])
    target_reference_df = pd.DataFrame(columns=z_coordinates_name, data=target_reference_z_point.reshape(1, -1))

    target_search_bool_arr = np.ones(principal_df.shape[0], dtype=bool)  # initially assume all True for every molecule. (num_molecules,)
    for z_coordinate in z_coordinates_name:
        target_search_bool_arr *= ((principal_df[z_coordinate] - target_reference_df[z_coordinate].item()).abs() <= epsilon).to_numpy()  # update
    principal_df_near_origin = principal_df[target_search_bool_arr].copy()

    # calculate z distance from the reference point
    z_vector_from_reference_arr = principal_df_near_origin[z_coordinates_name].to_numpy() - target_reference_z_point  # (num_molecules, latent_dim)
    z_distance_from_reference_arr = np.linalg.norm(z_vector_from_reference_arr, axis=1)
    principal_df_near_origin['z_distance'] = z_distance_from_reference_arr

    # 2) Add PCA distance from (0, 0, 0) for visualization -> 3D PCA space
    PCA_reference_point = (0, 0, 0)
    principal_df_near_origin['PCA_distance'] = (
            (principal_df_near_origin['PC1'] - PCA_reference_point[0]) ** 2 +
            (principal_df_near_origin['PC2'] - PCA_reference_point[1]) ** 2 +
            (principal_df_near_origin['PC3'] - PCA_reference_point[2]) ** 2
    ).pow(0.5)

    # sort by distance
    principal_df_near_origin = principal_df_near_origin[principal_df_near_origin['total_prediction_error'] <= 0.01]
    principal_df_near_origin = principal_df_near_origin[principal_df_near_origin['z_distance'] <= 10]
    principal_df_near_origin = principal_df_near_origin[principal_df_near_origin['PCA_distance'] <= 3.0]
    principal_df_near_origin = principal_df_near_origin.sort_values(by='PCA_distance', ascending=True)
    # print(principal_df_near_origin[['product', 'z_distance', 'PCA_distance', 'total_prediction_error']].head(100).to_string())
    principal_df_near_origin = principal_df_near_origin.loc[[reference_molecule_loc]]  # choose a molecule with a low prediction error for visualization
    # print(principal_df_near_origin[['fitted_Tg_[K]_true', 'fitted_band_gap_[eV]_true', 'chi_water_true']])
    # print(principal_df_near_origin[['fitted_Tg_[K]_pred', 'fitted_band_gap_[eV]_pred', 'chi_water_pred']])
    assert principal_df_near_origin.shape[0] == 1

    # PCA reference point for 3D
    PCA_name_list = ['PC1', 'PC2', 'PC3']
    PCA_reference_point = principal_df_near_origin[PCA_name_list].to_numpy().flatten()

    # Draw the latent space landscape
    pca_dir_3d_landscape = Path(os.path.join(pca_results_dir, '3d_landscape'))  # save dir
    pca_dir_3d_landscape.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    ### run this when you create 3D PCA plot
    scatter = ax.scatter(
        principal_df['PC1'],
        principal_df['PC2'],
        principal_df['PC3'],
        c='#CC97BD',  # use the same color even for different polymerization mechanisms
        s=2.0,
        alpha=0.005
    )

    # add reference point
    ax.scatter(
        principal_df_near_origin['PC1'],
        principal_df_near_origin['PC2'],
        principal_df_near_origin['PC3'],
        facecolor='#21150A',  # interior color
        edgecolor='#21150A',  # edge color
        s=100.0,
        marker='*',
        alpha=1.0
    )

    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.set_zlabel('Principal component 3')

    # Get rid of colored axes planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # adjust alpha of a grid
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid']['color'] = (0.827, 0.827, 0.827, 0.7)  # RGBA: last value is alpha

    #### STEP 3) Draw arrow on the 3D plot
    monomer_phi_property_idx = 0
    lumo_energy_property_idx = 2
    homo_energy_property_idx = 1
    chi_water_property_idx = 8

    monomer_phi_direction_vector = state_dict[f'property_network_module.{monomer_phi_property_idx}.weight'].cpu().numpy().squeeze()

    ### homo_lumo gap vector needs to be unscaled to get a vector because ...
    # delta_homo_lumo_gap = (LUMO * delta z) * std_lumo - (HOMO * delta_z) * std_homo

    # before unscaling
    # homo_lumo_gap_property_direction_vector = state_dict[f'property_network_module.{lumo_energy_property_idx}.weight'].cpu().numpy().squeeze() - state_dict[f'property_network_module.{homo_energy_property_idx}.weight'].cpu().numpy().squeeze()

    # after unscaling
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    train_idx_for_property_scaling = torch.load(os.path.join(data_dir, 'train_idx.pth'))  # whole train_idx for train property scaling
    property_train_for_property_scaling = property_value[train_idx_for_property_scaling]

    E_LUMO_scaler = StandardScaler()  # property scaling for LUMO Energy
    E_LUMO_scaler = E_LUMO_scaler.fit(property_train_for_property_scaling[:, lumo_energy_property_idx].reshape(-1, 1))  # fit
    std_LUMO = E_LUMO_scaler.scale_[0]

    E_HOMO_scaler = StandardScaler()  # property scaling for HOMO Energy
    E_HOMO_scaler = E_HOMO_scaler.fit(property_train_for_property_scaling[:, homo_energy_property_idx].reshape(-1, 1))  # fit
    std_HOMO = E_HOMO_scaler.scale_[0]

    homo_lumo_gap_property_direction_vector = std_LUMO * state_dict[f'property_network_module.{lumo_energy_property_idx}.weight'].cpu().numpy().squeeze() - std_HOMO * state_dict[f'property_network_module.{homo_energy_property_idx}.weight'].cpu().numpy().squeeze()
    ###

    chi_water_direction_vector = state_dict[f'property_network_module.{chi_water_property_idx}.weight'].cpu().numpy().squeeze()

    monomer_phi_property_direction_unit_vector = monomer_phi_direction_vector / np.linalg.norm(monomer_phi_direction_vector)
    homo_lumo_gap_property_direction_unit_vector = homo_lumo_gap_property_direction_vector / np.linalg.norm(homo_lumo_gap_property_direction_vector)
    chi_water_property_direction_unit_vector = chi_water_direction_vector / np.linalg.norm(chi_water_direction_vector)

    # Tg has a negative correlation to monomer phi
    monomer_phi_property_direction_unit_vector *= -1

    # PCA projection of property direction unit vector
    data_mean = np.mean(train_z_mean, axis=0)
    property_direction_unit_vector_arr = np.vstack((monomer_phi_property_direction_unit_vector, homo_lumo_gap_property_direction_unit_vector, chi_water_property_direction_unit_vector)) - data_mean
    property_direction_unit_vector_arr_scaled = property_direction_unit_vector_arr * scale_factor  # scale

    # PCA
    n_components = latent_space_dim
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(train_z_mean)
    whole_PCA_name_list = ['PC%d' % (num + 1) for num in range(n_components)]

    pca_projection_df_property_direction = pd.DataFrame(data=None, columns=whole_PCA_name_list)
    for pca_idx, pca_name in enumerate(whole_PCA_name_list):
        pca_projection_df_property_direction[pca_name] = property_direction_unit_vector_arr_scaled.dot(pca.components_[pca_idx])

    # get a normal vector
    pca_projection_property_direction = pca_projection_df_property_direction[[f'PC1', f'PC2', f'PC3']].to_numpy()  # (num_properties, PCA_dim)

    # run this when you create 3D PCA plot with arrows
    arrow_end = PCA_reference_point + pca_projection_property_direction
    arrow_color_list = ['#F2255B', '#1CC41C', '#6A83D3']
    assert arrow_end.shape[0] == len(arrow_color_list)
    for arrow_position, arrow_color in zip(arrow_end, arrow_color_list):
        # draw arrow: https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-a-3d-plot
        arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color=arrow_color, shrinkA=0,
                               shrinkB=0)  # mutation_scale -> size of arrows / shrinkA -> length of the line / shrinkB -> ??
        arrow = Arrow3D(
            [PCA_reference_point[0], arrow_position[0]],
            [PCA_reference_point[1], arrow_position[1]],
            [PCA_reference_point[2], arrow_position[2]],
            **arrow_prop_dict
        )
        ax.add_artist(arrow)

    # save
    ax.view_init(azim=-30)  # for better visualization. elev: Elevation angle in the z plane. azim: Azimuth angle in the x-y plane. azim -> default: -60
    plt.savefig(os.path.join(pca_dir_3d_landscape, f'3d_landscape_modified_HOMO_LUMO_direction.png'))
    plt.close()

    #### STEP 3) Find a relevant axis for 2D plot
    def draw_2D_latent_space_plot(eigen_cut_off, latent_space_dim, principal_df, hue_property, cmap, color_bar_name,
                                  save_plot_name, pca_projection_df_property_direction, arrow_color,
                                  property_direction_unit_vector_to_use):
        # cut of PCA axis
        inner_product_arr = np.zeros(shape=(latent_space_dim,))
        for eigen_idx, eigenvector in enumerate(pca.components_[:eigen_cut_off]):
            inner_product = abs(np.dot(property_direction_unit_vector_to_use, eigenvector))  # absolute values
            inner_product_arr[eigen_idx] = inner_product

        # sort
        inner_product_arr_idx = np.argsort(inner_product_arr)
        most_relevant_pca_axis_idx = inner_product_arr_idx[-1] + 1  # plus 1 because the figure "1" index
        next_relevant_pca_axis_idx = inner_product_arr_idx[-2] + 1  # plus 1 because the figure "1" index

        # plot
        g = sns.JointGrid(x=f'PC{most_relevant_pca_axis_idx}', y=f'PC{next_relevant_pca_axis_idx}', data=principal_df)
        sns.scatterplot(x=f'PC{most_relevant_pca_axis_idx}', y=f'PC{next_relevant_pca_axis_idx}',
                        hue=hue_property, data=principal_df, palette=cmap,
                        ax=g.ax_joint,
                        edgecolor=None, alpha=1.00, legend=False, size=3.0)

        # arrow starting point
        PCA_name_list = [f'PC{most_relevant_pca_axis_idx}', f'PC{next_relevant_pca_axis_idx}']
        origin_idx = principal_df_near_origin.index[0]
        arrow_starting_point = (principal_df.loc[origin_idx])[PCA_name_list].to_numpy().flatten()

        # arrow delta
        arrow_delta = pca_projection_df_property_direction[PCA_name_list].to_numpy().flatten()

        # draw arrow move arrow to the center of the PCA plot
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

        # find a molecule to visualize
        loc_dict = {
            'Tg':
                {
                    scale_factor_list[0]: Tg_molecule_scale_1_loc,
                    scale_factor_list[1]: Tg_molecule_scale_2_loc
                },
            'band_gap':
                {
                    scale_factor_list[0]: band_gap_molecule_scale_1_loc,
                    scale_factor_list[1]: band_gap_molecule_scale_2_loc,
                },
            'chi_water':
                {
                    scale_factor_list[0]: chi_water_molecule_scale_1_loc,
                    scale_factor_list[1]: chi_water_molecule_scale_2_loc,
                }
        }
        distance_dict = {
            'Tg': ['phi_prediction_error', 'normalized_monomer_phi_Boltzmann_average', 'monomer_phi_prediction'],
            'band_gap': ['homo_lumo_gap_prediction_error', 'HOMO_LUMO_gap_Boltzmann_average', 'homo_lumo_gap_prediction'],
            'chi_water': ['chi_water_prediction_error', 'chi_parameter_water_mean', 'chi_water_prediction']
        }
        property_dict = {
            'Tg': ['fitted_Tg_[K]_true', 'fitted_Tg_[K]_pred', 'normalized_monomer_phi_Boltzmann_average', 'monomer_phi_prediction', 'fitted_band_gap_[eV]_true', 'fitted_band_gap_[eV]_pred', 'chi_water_true', 'chi_water_pred'],
            'band_gap': ['fitted_band_gap_[eV]_true', 'fitted_band_gap_[eV]_pred', 'HOMO_LUMO_gap_Boltzmann_average', 'homo_lumo_gap_prediction', 'fitted_Tg_[K]_true', 'fitted_Tg_[K]_pred', 'chi_water_true', 'chi_water_pred'],
            'chi_water': ['chi_water_true', 'chi_water_pred', 'chi_parameter_water_mean', 'chi_water_prediction', 'fitted_Tg_[K]_true', 'fitted_Tg_[K]_pred', 'fitted_band_gap_[eV]_true', 'fitted_band_gap_[eV]_pred']
        }
        structure_list = ['product', 'reactant_1', 'reactant_2', 'reaction_id']
        print('===============================================================================')
        print(f'Index of a polymer located at the origin: {principal_df_near_origin.index[0]}')
        print('===============================================================================')
        df_property_to_show = principal_df_near_origin[property_dict[save_plot_name]]
        df_structure_to_show = principal_df_near_origin[structure_list]
        point_to_draw_list = []
        for scale_factor in scale_factor_list:
            print(f'{save_plot_name}, scale factor: {scale_factor}', flush=True)
            # get molecule with a latent space search based on "z-space", "not PCA space"
            target_point_df = principal_df_near_origin[z_coordinates_name] + property_direction_unit_vector_to_use * scale_factor  # molecule located on PCA origin + linear directions. This should be performed z-space to avoid information lost on PCA space. (1, latent_dim)
            principal_df_visualization = principal_df.copy()

            # calculate distance on the z space
            target_point_arr = target_point_df.to_numpy()
            reference_point_arr = principal_df_near_origin[z_coordinates_name].to_numpy()

            target_vector_from_reference = target_point_arr - reference_point_arr
            trial_vector_from_reference = principal_df_visualization[z_coordinates_name].to_numpy() - reference_point_arr

            scale_factor_calculated = np.linalg.norm(target_vector_from_reference)  # the same as scale_factor
            directional_dot_product = trial_vector_from_reference @ target_vector_from_reference.reshape(-1, 1) / scale_factor_calculated

            directional_distance = directional_dot_product.flatten() - scale_factor_calculated  # this is mixed (positive + negative). Should only care positive value to get a distance
            principal_df_visualization['z_directional_distance'] = directional_distance  # append

            # use positive distance
            principal_df_visualization = principal_df_visualization[principal_df_visualization['z_directional_distance'] > 0]

            # calculate a distance from a reference point
            z_distance_arr = principal_df_visualization[z_coordinates_name].to_numpy() - reference_point_arr
            z_distance_arr = np.linalg.norm(z_distance_arr, axis=1)
            principal_df_visualization['z_distance'] = z_distance_arr

            # choose molecules
            principal_df_visualization = principal_df_visualization.sort_values(by='z_directional_distance', ascending=True).head(1000)
            principal_df_visualization = principal_df_visualization.sort_values(by='z_distance', ascending=True)
            print(principal_df_visualization[['product', 'z_directional_distance', 'z_distance'] + distance_dict[save_plot_name]].head(100).to_string())

            if not use_following_indices:  # use this one to find a idx to visualize as a starting point
                principal_df_chosen = principal_df_visualization.iloc[[0]]
            else:
                principal_df_chosen = principal_df_visualization.loc[[loc_dict[save_plot_name][scale_factor]]]  # lowest prediction error (except for the origin) + small PCA distance for visualization
            assert principal_df_chosen.shape[0] == 1
            df_property_to_show = pd.concat([df_property_to_show, principal_df_chosen[property_dict[save_plot_name]]], axis=0)
            df_structure_to_show = pd.concat([df_structure_to_show, principal_df_chosen[structure_list]], axis=0)

            # add points
            point_to_draw_list.append(principal_df_chosen[PCA_name_list].to_numpy().flatten())

        # print
        print(df_property_to_show.to_string())
        print(df_structure_to_show.to_string())

        # add points
        point_to_draw_arr = np.array(point_to_draw_list)
        # print(point_to_draw_arr)
        g.ax_joint.scatter(
            x=point_to_draw_arr[:, 0],
            y=point_to_draw_arr[:, 1],
            facecolor='#CC97BD',  # interior color
            edgecolor='#21150A',  # edge color
            s=100.0,
            marker='X',
            alpha=1.0
        )

        # kde plots on the marginal axes
        sns.kdeplot(x=f'PC{most_relevant_pca_axis_idx}', data=principal_df, ax=g.ax_marg_x, fill=True, color='#CC97BD', alpha=0.1)
        sns.kdeplot(y=f'PC{next_relevant_pca_axis_idx}', data=principal_df, ax=g.ax_marg_y, fill=True, color='#CC97BD', alpha=0.1)

        # tick parameters
        if save_plot_name == 'Tg':
            g.ax_joint.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6])
        else:
            g.ax_joint.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        g.ax_joint.tick_params(labelsize=15)
        g.set_axis_labels(f'Principal component {most_relevant_pca_axis_idx}', f'Principal component {next_relevant_pca_axis_idx}', fontsize=16)

        # color bars
        norm = plt.Normalize(principal_df[hue_property].min(), principal_df[hue_property].max())
        scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        g.fig.tight_layout()

        # Make space for the colorbar
        g.fig.subplots_adjust(bottom=0.2)
        cax = g.fig.add_axes([0.20, 0.06, 0.6, 0.02])  # l, b, w, h
        cbar = g.fig.colorbar(scatter_map, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        g.ax_joint.set_aspect('equal', adjustable='box')
        g.fig.savefig(os.path.join(pca_dir_3d_landscape, f'{save_plot_name}_2D_latent_space_modified_HOMO_LUMO_gap_direction.png'), dpi=600)
        plt.close()

    # 1) Tg
    # Tg_save_plot_name = 'Tg'
    # Tg_color_bar_name = '$T_g$ [K]'
    # Tg_hue_property = 'fitted_Tg_[K]_pred'
    # Tg_arrow_color = '#F2255B'
    # Tg_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#6296C4", "#F5EF7E", "#FE7F2D"])
    # Tg_arrow_ending_point_df = pca_projection_df_property_direction.iloc[[0]]
    # draw_2D_latent_space_plot(eigen_cut_off=eigen_cut_off, latent_space_dim=latent_space_dim, principal_df=principal_df,
    #                           hue_property=Tg_hue_property, cmap=Tg_cmap, color_bar_name=Tg_color_bar_name,
    #                           save_plot_name=Tg_save_plot_name, pca_projection_df_property_direction=Tg_arrow_ending_point_df,
    #                           arrow_color=Tg_arrow_color, property_direction_unit_vector_to_use=monomer_phi_property_direction_unit_vector)
    # print(f'{Tg_save_plot_name} plot is done!\n', flush=True)

    # 2) Band gap
    band_gap_save_plot_name = 'band_gap'
    band_gap_color_bar_name = '$E_{gap}$ [eV]'
    band_gap_hue_property = 'fitted_band_gap_[eV]_pred'
    band_gap_arrow_color = '#1CC41C'
    band_gap_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#6296C4", "#F5EF7E", "#FE7F2D"])
    band_gap_arrow_ending_point_df = pca_projection_df_property_direction.iloc[[1]]
    draw_2D_latent_space_plot(eigen_cut_off=eigen_cut_off, latent_space_dim=latent_space_dim, principal_df=principal_df,
                              hue_property=band_gap_hue_property, cmap=band_gap_cmap, color_bar_name=band_gap_color_bar_name,
                              save_plot_name=band_gap_save_plot_name,
                              pca_projection_df_property_direction=band_gap_arrow_ending_point_df,
                              arrow_color=band_gap_arrow_color,
                              property_direction_unit_vector_to_use=homo_lumo_gap_property_direction_unit_vector)
    print(f'{band_gap_save_plot_name} plot is done!\n', flush=True)

    # 3) Chi water
    # chi_water_save_plot_name = 'chi_water'
    # chi_water_color_bar_name = '$\chi_{water}$ [unitless]'
    # chi_water_hue_property = 'chi_water_pred'
    # chi_water_arrow_color = '#6A83D3'
    # chi_water_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#6296C4", "#F5EF7E", "#FE7F2D"])
    # chi_water_arrow_ending_point_df = pca_projection_df_property_direction.iloc[[2]]
    # draw_2D_latent_space_plot(eigen_cut_off=eigen_cut_off, latent_space_dim=latent_space_dim, principal_df=principal_df,
    #                           hue_property=chi_water_hue_property, cmap=chi_water_cmap,
    #                           color_bar_name=chi_water_color_bar_name,
    #                           save_plot_name=chi_water_save_plot_name,
    #                           pca_projection_df_property_direction=chi_water_arrow_ending_point_df,
    #                           arrow_color=chi_water_arrow_color,
    #                           property_direction_unit_vector_to_use=chi_water_property_direction_unit_vector)
    # print(f'{chi_water_save_plot_name} plot is done!\n', flush=True)
