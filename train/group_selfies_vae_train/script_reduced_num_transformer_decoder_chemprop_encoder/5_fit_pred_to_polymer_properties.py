import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from pathlib import Path
from sklearn.metrics import r2_score

if __name__ == '__main__':
    """
    This script
    (1) Fits the model prediction from latent space visualization to polymer properties
    (2) Plots the results in terms of fitted polymeric properties, rather than monomer-based properties.
    """
    # set dir for trained GenML (linear)
    dir_name = 'polymerization_step_growth_750k'
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    train_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/group_selfies_vae_train/{dir_name}'
    model_save_directory = os.path.join(
        train_dir,
        'divergence_weight_2.949_latent_dim_104_learning_rate_0.003',
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

    # pred plot dir
    pred_plot_dir = os.path.join(model_check_point_directory, 'property_pred')

    # set dir for fitting to polymer properties
    model_fit_dir = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer'

    ### 1) Fit phi monomer
    model_fit_json = os.path.join(model_fit_dir, 'Tg', 'model_fit', 'inverse_params.json')

    def inverse_function(x, a, b, c):  # inverse function
        return a / (x + b) + c
    with open(model_fit_json, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_inverse = lambda x: inverse_function(x, **params)  # define function

    # load arr
    property_label = 'Phi_{mon}'
    plot_label = '$T_{g}$'
    property_arr_dir = os.path.join(pred_plot_dir, property_label)  # inverse_transformed
    valid_pred_arr = np.load(os.path.join(property_arr_dir, 'valid_pred_arr_inverse_transformed.npy'))
    valid_true_arr = np.load(os.path.join(property_arr_dir, 'valid_true_arr_inverse_transformed.npy'))
    test_pred_arr = np.load(os.path.join(property_arr_dir, 'test_pred_arr_inverse_transformed.npy'))
    test_true_arr = np.load(os.path.join(property_arr_dir, 'test_true_arr_inverse_transformed.npy'))

    # fitting with an inverse function
    fitted_valid_pred_arr = fitted_inverse(valid_pred_arr)  # Tg
    fitted_valid_true_arr = fitted_inverse(valid_true_arr)  # Tg
    fitted_test_pred_arr = fitted_inverse(test_pred_arr)  # Tg
    fitted_test_true_arr = fitted_inverse(test_true_arr)  # Tg

    # load df to visualize molecules
    df = pd.read_csv(os.path.join(data_dir, 'two_reactants.csv'))
    valid_idx = torch.load(os.path.join(data_dir, 'valid_idx.pth'))
    test_idx = torch.load(os.path.join(data_dir, 'test_idx.pth'))
    df_valid = df.iloc[valid_idx]
    df_test = df.iloc[test_idx]

    # filter based on the mix & max value of experimental Tg
    df_Tg = pd.read_csv('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer/Tg/normalized_phi_estimation.csv')
    experimental_Tg_arr = df_Tg['experimental_Tg'].to_numpy()

    valid_pass_true = (fitted_valid_true_arr >= experimental_Tg_arr.min()) & (fitted_valid_true_arr <= experimental_Tg_arr.max())
    valid_pass_pred = (fitted_valid_pred_arr >= experimental_Tg_arr.min()) & (fitted_valid_pred_arr <= experimental_Tg_arr.max())
    valid_pass = valid_pass_true & valid_pass_pred
    print(f'[Valid] Conditional probability of True Tg satisfying an experimental range if pred satisfies {valid_pass.sum() / valid_pass_pred.sum()}')  # conditional probability
    fitted_valid_pred_arr = fitted_valid_pred_arr[valid_pass]
    fitted_valid_true_arr = fitted_valid_true_arr[valid_pass]
    df_valid_Tg = df_valid[valid_pass].copy()
    df_valid_Tg['fitted_valid_pred'] = fitted_valid_pred_arr  # append
    df_valid_Tg['fitted_valid_true'] = fitted_valid_true_arr  # append

    test_pass_true = (fitted_test_true_arr >= experimental_Tg_arr.min()) & (fitted_test_true_arr <= experimental_Tg_arr.max())
    test_pass_pred = (fitted_test_pred_arr >= experimental_Tg_arr.min()) & (fitted_test_pred_arr <= experimental_Tg_arr.max())
    test_pass = test_pass_true & test_pass_pred
    print(f'[Test] Conditional probability of True Tg satisfying an experimental range if pred satisfies {test_pass.sum() / test_pass_pred.sum()}')
    fitted_test_pred_arr = fitted_test_pred_arr[test_pass]
    fitted_test_true_arr = fitted_test_true_arr[test_pass]
    df_test_Tg = df_test[test_pass].copy()
    df_test_Tg['fitted_test_pred'] = fitted_test_pred_arr  # append
    df_test_Tg['fitted_test_true'] = fitted_test_true_arr  # append

    # seaborn plot
    g = sns.JointGrid()

    # valid plot df
    valid_color = '#219B35'
    valid_plot_df = pd.DataFrame({'true': fitted_valid_true_arr, 'pred': fitted_valid_pred_arr})
    valid_r2_score = r2_score(y_true=fitted_valid_true_arr, y_pred=fitted_valid_pred_arr)
    ax = sns.scatterplot(x='true', y='pred', data=valid_plot_df, ax=g.ax_joint,
                         edgecolor=None, alpha=0.5, legend=False, size=3.0)
    sns.kdeplot(x='true', data=valid_plot_df, ax=g.ax_marg_x, fill=True, color=valid_color)
    sns.kdeplot(y='pred', data=valid_plot_df, ax=g.ax_marg_y, fill=True, color=valid_color)

    # test plot df
    test_color = '#ED3B6F'
    test_plot_df = pd.DataFrame({'true': fitted_test_true_arr, 'pred': fitted_test_pred_arr})
    test_r2_score = r2_score(y_true=fitted_test_true_arr, y_pred=fitted_test_pred_arr)
    ax = sns.scatterplot(x='true', y='pred', data=test_plot_df, ax=g.ax_joint,
                         edgecolor=None, alpha=0.5, legend=False, size=3.0)
    sns.kdeplot(x='true', data=test_plot_df, ax=g.ax_marg_x, fill=True, color=test_color)
    sns.kdeplot(y='pred', data=test_plot_df, ax=g.ax_marg_y, fill=True, color=test_color)

    # tick parameters
    g.ax_joint.tick_params(labelsize=16)
    g.set_axis_labels(f'{plot_label} from experimental fitting [K]', f'Prediction {plot_label} [K]', fontsize=16)

    # y=x
    plot_true_min = np.min([np.min(fitted_valid_true_arr), np.min(fitted_test_true_arr)])
    plot_true_max = np.max([np.max(fitted_valid_true_arr), np.max(fitted_test_true_arr)])
    plot_pred_min = np.min([np.min(fitted_valid_pred_arr), np.min(fitted_test_pred_arr)])
    plot_pred_max = np.max([np.max(fitted_valid_pred_arr), np.max(fitted_test_pred_arr)])
    plot_min = np.min([plot_true_min, plot_pred_min])
    plot_max = np.max([plot_true_max, plot_pred_max])
    g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], color='#665A5E', linestyle='--', linewidth=1.5)

    # Create custom legend handles with different colors
    valid_handle = plt.Line2D([], [], color=valid_color, label=f'Valid $R^2$: {valid_r2_score:.2f}', marker='o',
                              linestyle='None')
    test_handle = plt.Line2D([], [], color=test_color, label=f'Test $R^2$: {test_r2_score:.2f}', marker='o',
                             linestyle='None')

    # Add the custom handles to the legend
    g.ax_joint.legend(handles=[valid_handle, test_handle], handletextpad=0.1,
                      fontsize=14, loc='upper left')

    # tight layout & save
    fig_save_dir = Path(os.path.join(property_arr_dir, f'fitting_to_{plot_label}'))
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"property_prediction.png"))
    plt.close()

    ### 2) Fit Chi water
    property_label = 'chi_{water}'
    plot_label = '$\chi_{water}$'
    property_arr_dir = os.path.join(pred_plot_dir, property_label)
    valid_pred_arr = np.load(os.path.join(property_arr_dir, 'valid_pred_arr_inverse_transformed.npy'))
    valid_true_arr = np.load(os.path.join(property_arr_dir, 'valid_true_arr_inverse_transformed.npy'))
    test_pred_arr = np.load(os.path.join(property_arr_dir, 'test_pred_arr_inverse_transformed.npy'))
    test_true_arr = np.load(os.path.join(property_arr_dir, 'test_true_arr_inverse_transformed.npy'))

    model_fit_json = os.path.join(model_fit_dir, 'Chi', 'model_fit', 'water_chi.json')

    def linear_function(x, gradient, bias):  # linear function
        return gradient * x + bias
    with open(model_fit_json, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_linear = lambda x: linear_function(x, **params)  # define function

    # fitting with a linear function
    fitted_valid_pred_arr = fitted_linear(valid_pred_arr)
    fitted_valid_true_arr = fitted_linear(valid_true_arr)
    fitted_test_pred_arr = fitted_linear(test_pred_arr)
    fitted_test_true_arr = fitted_linear(test_true_arr)

    df_valid_chi_water = df_valid.copy()
    df_valid_chi_water['fitted_valid_pred'] = fitted_valid_pred_arr  # append
    df_valid_chi_water['fitted_valid_true'] = fitted_valid_true_arr  # append

    df_test_chi_water = df_test.copy()
    df_test_chi_water['fitted_test_pred'] = fitted_test_pred_arr  # append
    df_test_chi_water['fitted_test_true'] = fitted_test_true_arr  # append

    # seaborn plot
    g = sns.JointGrid()

    # valid plot df
    valid_color = '#219B35'
    valid_plot_df = pd.DataFrame({'true': fitted_valid_true_arr, 'pred': fitted_valid_pred_arr})
    valid_r2_score = r2_score(y_true=fitted_valid_true_arr, y_pred=fitted_valid_pred_arr)
    ax = sns.scatterplot(x='true', y='pred', data=valid_plot_df, ax=g.ax_joint,
                         edgecolor=None, alpha=0.5, legend=False, size=3.0)
    sns.kdeplot(x='true', data=valid_plot_df, ax=g.ax_marg_x, fill=True, color=valid_color)
    sns.kdeplot(y='pred', data=valid_plot_df, ax=g.ax_marg_y, fill=True, color=valid_color)

    # test plot df
    test_color = '#ED3B6F'
    test_plot_df = pd.DataFrame({'true': fitted_test_true_arr, 'pred': fitted_test_pred_arr})
    test_r2_score = r2_score(y_true=fitted_test_true_arr, y_pred=fitted_test_pred_arr)
    ax = sns.scatterplot(x='true', y='pred', data=test_plot_df, ax=g.ax_joint,
                         edgecolor=None, alpha=0.5, legend=False, size=3.0)
    sns.kdeplot(x='true', data=test_plot_df, ax=g.ax_marg_x, fill=True, color=test_color)
    sns.kdeplot(y='pred', data=test_plot_df, ax=g.ax_marg_y, fill=True, color=test_color)

    # tick parameters
    g.ax_joint.tick_params(labelsize=16)
    g.set_axis_labels(f'{plot_label} from experimental fitting', f'Prediction {plot_label}', fontsize=16)

    # y=x
    plot_true_min = np.min([np.min(fitted_valid_true_arr), np.min(fitted_test_true_arr)])
    plot_true_max = np.max([np.max(fitted_valid_true_arr), np.max(fitted_test_true_arr)])
    plot_pred_min = np.min([np.min(fitted_valid_pred_arr), np.min(fitted_test_pred_arr)])
    plot_pred_max = np.max([np.max(fitted_valid_pred_arr), np.max(fitted_test_pred_arr)])
    plot_min = np.min([plot_true_min, plot_pred_min])
    plot_max = np.max([plot_true_max, plot_pred_max])
    g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], color='#665A5E', linestyle='--', linewidth=1.5)

    # Create custom legend handles with different colors
    valid_handle = plt.Line2D([], [], color=valid_color, label=f'Valid $R^2$: {valid_r2_score:.2f}', marker='o',
                              linestyle='None')
    test_handle = plt.Line2D([], [], color=test_color, label=f'Test $R^2$: {test_r2_score:.2f}', marker='o',
                             linestyle='None')

    # Add the custom handles to the legend
    g.ax_joint.legend(handles=[valid_handle, test_handle], handletextpad=0.1,
                      fontsize=14, loc='upper left')

    # tight layout & save
    fig_save_dir = Path(os.path.join(property_arr_dir, f'fitting_to_{plot_label}'))
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"property_prediction.png"))
    plt.close()

    # 3) Fit band gap
    property_label_homo = 'E_{HOMO}'
    property_arr_dir = os.path.join(pred_plot_dir, property_label_homo)
    valid_pred_arr_homo = np.load(os.path.join(property_arr_dir, 'valid_pred_arr_inverse_transformed.npy'))
    valid_true_arr_homo = np.load(os.path.join(property_arr_dir, 'valid_true_arr_inverse_transformed.npy'))
    test_pred_arr_homo = np.load(os.path.join(property_arr_dir, 'test_pred_arr_inverse_transformed.npy'))
    test_true_arr_homo = np.load(os.path.join(property_arr_dir, 'test_true_arr_inverse_transformed.npy'))

    property_label_lumo = 'E_{LUMO}'
    property_arr_dir = os.path.join(pred_plot_dir, property_label_lumo)
    valid_pred_arr_lumo = np.load(os.path.join(property_arr_dir, 'valid_pred_arr_inverse_transformed.npy'))
    valid_true_arr_lumo = np.load(os.path.join(property_arr_dir, 'valid_true_arr_inverse_transformed.npy'))
    test_pred_arr_lumo = np.load(os.path.join(property_arr_dir, 'test_pred_arr_inverse_transformed.npy'))
    test_true_arr_lumo = np.load(os.path.join(property_arr_dir, 'test_true_arr_inverse_transformed.npy'))

    property_label = 'HOMO_LUMO_gap'
    plot_label = 'Polymer $E_{gap}$'
    property_arr_dir = Path(os.path.join(pred_plot_dir, property_label))
    property_arr_dir.mkdir(exist_ok=True, parents=True)
    valid_pred_arr = valid_pred_arr_lumo - valid_pred_arr_homo
    valid_true_arr = valid_true_arr_lumo - valid_true_arr_homo
    test_pred_arr = test_pred_arr_lumo - test_pred_arr_homo
    test_true_arr = test_true_arr_lumo - test_true_arr_homo

    model_fit_json = os.path.join(model_fit_dir, 'gap_bulk', 'model_fit', 'linear_params.json')

    def linear_function(x, gradient, bias):  # linear function
        return gradient * x + bias
    with open(model_fit_json, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_linear = lambda x: linear_function(x, **params)  # define function

    # fitting with a linear function
    fitted_valid_pred_arr = fitted_linear(valid_pred_arr)
    fitted_valid_true_arr = fitted_linear(valid_true_arr)
    fitted_test_pred_arr = fitted_linear(test_pred_arr)
    fitted_test_true_arr = fitted_linear(test_true_arr)

    # filter based on the mix & max value of polymer gap
    df_polymer_gap = pd.read_csv('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer/gap_bulk/data/Egb.csv')
    polymer_gap_arr = df_polymer_gap['value'].to_numpy()

    valid_pass_true = (fitted_valid_true_arr >= polymer_gap_arr.min()) & (fitted_valid_true_arr <= polymer_gap_arr.max())
    valid_pass_pred = (fitted_valid_pred_arr >= polymer_gap_arr.min()) & (fitted_valid_pred_arr <= polymer_gap_arr.max())
    valid_pass = valid_pass_true & valid_pass_pred
    print(f'[Valid] Conditional probability of True band gap satisfying an computational range if pred satisfies {valid_pass.sum() / valid_pass_pred.sum()}')  # conditional probability
    fitted_valid_pred_arr = fitted_valid_pred_arr[valid_pass]
    fitted_valid_true_arr = fitted_valid_true_arr[valid_pass]
    df_valid_polymer_gap = df_valid[valid_pass].copy()
    df_valid_polymer_gap['fitted_valid_pred'] = fitted_valid_pred_arr  # append
    df_valid_polymer_gap['fitted_valid_true'] = fitted_valid_true_arr  # append

    test_pass_true = (fitted_test_true_arr >= polymer_gap_arr.min()) & (fitted_test_true_arr <= polymer_gap_arr.max())
    test_pass_pred = (fitted_test_pred_arr >= polymer_gap_arr.min()) & (fitted_test_pred_arr <= polymer_gap_arr.max())
    test_pass = test_pass_true & test_pass_pred
    print(f'[Test] Conditional probability of True band gap satisfying an computational range if pred satisfies {test_pass.sum() / test_pass_pred.sum()}')  # conditional probability
    fitted_test_pred_arr = fitted_test_pred_arr[test_pass]
    fitted_test_true_arr = fitted_test_true_arr[test_pass]
    df_test_polymer_gap = df_test[test_pass].copy()
    df_test_polymer_gap['fitted_test_pred'] = fitted_test_pred_arr  # append
    df_test_polymer_gap['fitted_test_true'] = fitted_test_true_arr  # append

    # seaborn plot
    g = sns.JointGrid()

    # valid plot df
    valid_color = '#219B35'
    valid_plot_df = pd.DataFrame({'true': fitted_valid_true_arr, 'pred': fitted_valid_pred_arr})
    valid_r2_score = r2_score(y_true=fitted_valid_true_arr, y_pred=fitted_valid_pred_arr)
    ax = sns.scatterplot(x='true', y='pred', data=valid_plot_df, ax=g.ax_joint,
                         edgecolor=None, alpha=0.5, legend=False, size=3.0)
    sns.kdeplot(x='true', data=valid_plot_df, ax=g.ax_marg_x, fill=True, color=valid_color)
    sns.kdeplot(y='pred', data=valid_plot_df, ax=g.ax_marg_y, fill=True, color=valid_color)

    # test plot df
    test_color = '#ED3B6F'
    test_plot_df = pd.DataFrame({'true': fitted_test_true_arr, 'pred': fitted_test_pred_arr})
    test_r2_score = r2_score(y_true=fitted_test_true_arr, y_pred=fitted_test_pred_arr)
    ax = sns.scatterplot(x='true', y='pred', data=test_plot_df, ax=g.ax_joint,
                         edgecolor=None, alpha=0.5, legend=False, size=3.0)
    sns.kdeplot(x='true', data=test_plot_df, ax=g.ax_marg_x, fill=True, color=test_color)
    sns.kdeplot(y='pred', data=test_plot_df, ax=g.ax_marg_y, fill=True, color=test_color)

    # tick parameters
    g.ax_joint.tick_params(labelsize=16)
    g.set_axis_labels(f'{plot_label} from fitting to DFT [eV]', f'Prediction {plot_label}', fontsize=16)

    # y=x
    plot_true_min = np.min([np.min(fitted_valid_true_arr), np.min(fitted_test_true_arr)])
    plot_true_max = np.max([np.max(fitted_valid_true_arr), np.max(fitted_test_true_arr)])
    plot_pred_min = np.min([np.min(fitted_valid_pred_arr), np.min(fitted_test_pred_arr)])
    plot_pred_max = np.max([np.max(fitted_valid_pred_arr), np.max(fitted_test_pred_arr)])
    plot_min = np.min([plot_true_min, plot_pred_min])
    plot_max = np.max([plot_true_max, plot_pred_max])
    g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], color='#665A5E', linestyle='--', linewidth=1.5)

    # Create custom legend handles with different colors
    valid_handle = plt.Line2D([], [], color=valid_color, label=f'Valid $R^2$: {valid_r2_score:.2f}', marker='o',
                              linestyle='None')
    test_handle = plt.Line2D([], [], color=test_color, label=f'Test $R^2$: {test_r2_score:.2f}', marker='o',
                             linestyle='None')

    # Add the custom handles to the legend
    g.ax_joint.legend(handles=[valid_handle, test_handle], handletextpad=0.1,
                      fontsize=14, loc='upper left')

    # tight layout & save
    fig_save_dir = Path(os.path.join(property_arr_dir, f'fitting_to_{plot_label}'))
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"property_prediction.png"))
    plt.close()

    # visualize
    # df_test_sort = df_test_polymer_gap.sort_values('fitted_test_true', ascending=True)
    # print(df_test_sort[['product', 'fitted_test_true']].head(10).to_string())
    # print(df_test_sort[['product', 'fitted_test_true']].tail(10).to_string())
