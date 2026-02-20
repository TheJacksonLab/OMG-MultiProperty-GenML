import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

if __name__ == '__main__':
    """
    This script plots a figure for Bayesian optimization over a hypercube. Run this script after running "analyze_test_bo_hypercube.py".      
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

    # non-linear
    # dir_name = 'polymerization_step_growth_750k_non_linear'
    # train_dir = f'/home/sk77/PycharmProjects/omg_multi_task/train/group_selfies_vae_train/{dir_name}'
    # model_save_directory = os.path.join(
    #     train_dir,
    #     'divergence_weight_9.569_latent_dim_64_learning_rate_0.001',
    # )
    # model_check_point_directory = os.path.join(model_save_directory, 'check_point', 'epoch_39')

    # load data
    data_dir = f'/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/data/group_selfies_vae_data/{dir_name}'
    data_path = os.path.join(data_dir, 'two_reactants.csv')
    train_idx_path = os.path.join(data_dir, 'train_idx.pth')

    # property name
    property_name = ['monomer_phi', 'homo_lumo_gap', 'chi_water']

    # monomer property optimization direction
    optimization_direction = np.array([-1, 1, 1])  # decrease / increase / increase
    n_iter = 5  # Bayesian optimization
    num_polymer_per_iter = 20
    num_generated_polymers = num_polymer_per_iter * n_iter

    # dir to load the initial training data
    k = 2000
    gpr_dir = os.path.join(model_check_point_directory, 'gpr')
    initial_train_dir = os.path.join(gpr_dir, f'k_means_num_{k}')
    bo_dir = os.path.join(gpr_dir, 'bo', f'k_means_num_{k}')

    # hypercube length to explore
    hypercube_length_list = [0.2, 0.3, 0.4, 0.5]  # scaled
    ####### MODIFY #######

    # save dir for optimization direction
    optimization_save_name = 'optimization_'
    for direction in optimization_direction:
        optimization_save_name += f'{int(direction)}_'
    optimization_save_name = optimization_save_name[:-1]
    save_bo_optimization = os.path.join(bo_dir, optimization_save_name)

    # loop
    list_df_valid_novel_unique_opt = []
    percent_relative_improvement_mean_list_opt, percent_relative_improvement_min_list_opt, percent_relative_improvement_max_list_opt = [], [], []  # satisfying multi-objective optimization
    tanimoto_mean_list_opt, tanimoto_min_list_opt, tanimoto_max_list_opt = [], [], []
    for hypercube_length in hypercube_length_list:
        hypercube_dir = os.path.join(save_bo_optimization, f'hypercube_length_{hypercube_length}')

        # load
        df_valid_novel_unique_opt = pd.read_csv(os.path.join(hypercube_dir, 'valid_novel_unique_opt.csv'))

        # percent_relative_improvement
        percent_relative_improvement_arr = df_valid_novel_unique_opt['mean_relative_improvement_percent'].to_numpy()
        percent_relative_improvement_mean_list_opt.append(percent_relative_improvement_arr.mean())
        percent_relative_improvement_min_list_opt.append(percent_relative_improvement_arr.min())
        percent_relative_improvement_max_list_opt.append(percent_relative_improvement_arr.max())

        # Tanimoto similarity
        tanimoto_arr = df_valid_novel_unique_opt['Tanimoto_methyl_terminated_p_mol_and_reference'].to_numpy()
        tanimoto_mean_list_opt.append(tanimoto_arr.mean())
        tanimoto_min_list_opt.append(tanimoto_arr.min())
        tanimoto_max_list_opt.append(tanimoto_arr.max())

        # append
        list_df_valid_novel_unique_opt.append(df_valid_novel_unique_opt)

    # plot dir
    plot_dir = Path(os.path.join(save_bo_optimization, 'plot'))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # plot - all discovered polymers
    plt.figure(figsize=(6, 3.5))
    plt.plot(hypercube_length_list, percent_relative_improvement_mean_list_opt, color='#47B649', label='Mean', marker='o')  # green
    plt.fill_between(hypercube_length_list, percent_relative_improvement_min_list_opt, percent_relative_improvement_max_list_opt,
                     color='#47B649', alpha=0.1, label='Min & Max')
    plt.xlabel('Search length (scaled)', fontsize=14)
    plt.ylabel('Optimization score (%)', fontsize=14)  # mean value
    plt.yticks([0, 5, 10, 15, 20, 25, 30], fontsize=14)
    plt.xticks(hypercube_length_list, fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'opt_optimization_score_mean_trend_fill_between.svg'))
    plt.savefig(os.path.join(plot_dir, f'opt_optimization_score_mean_trend_fill_between.png'))
    plt.close()

    # plot Tanimoto similarity - only polymers satisfying optimization criteria
    plt.figure(figsize=(6, 3.5))
    plt.plot(hypercube_length_list, tanimoto_mean_list_opt, color='#4183C4', marker='o', label='Mean')  # blue
    plt.fill_between(hypercube_length_list, tanimoto_min_list_opt, tanimoto_max_list_opt,
                     color='#4183C4', alpha=0.1, label='Min & Max')
    plt.xlabel('Search length (scaled)', fontsize=14)
    plt.ylabel('Tanimoto similarity', fontsize=14)  # mean value
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=14)
    plt.xticks(hypercube_length_list, fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'opt_tanimoto_trend_fill_between.svg'))
    plt.savefig(os.path.join(plot_dir, f'opt_tanimoto_trend_fill_between.png'))
    plt.close()
