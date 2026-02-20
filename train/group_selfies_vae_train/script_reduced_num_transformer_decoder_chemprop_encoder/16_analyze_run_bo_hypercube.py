import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from pathlib import Path

if __name__ == '__main__':
    """
    This script analyzes Bayesian optimization over a hypercube for PFAS.         
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

    # property name
    property_name = ['monomer_phi', 'homo_lumo_gap', 'chi_water', 'num_fluorine']

    # monomer property optimization direction
    optimization_direction = np.array([-1, 1, 1, -1])  # decrease / increase / increase / decrease
    n_iter = 5  # Bayesian optimization
    num_polymer_per_iter = 20
    num_generated_polymers = num_polymer_per_iter * n_iter

    # dir to load the initial training data
    k = 2000
    gpr_dir = os.path.join(model_check_point_directory, 'gpr')
    initial_train_dir = os.path.join(gpr_dir, f'k_means_num_{k}_PFAS')
    bo_dir = os.path.join(gpr_dir, 'bo', f'k_means_num_{k}_PFAS_2')

    # hypercube length to explore
    hypercube_length_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # scaled

    # save from the bo script - reference polymer
    # reference_polymer = '*OCCOC(=O)c1ccc(C(*)=O)c(COC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)c1'  # PFAS 1
    reference_polymer = '*OCCOC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OCCOC(=O)Nc1ccc(Cc2ccc(NC(*)=O)cc2)cc1'  # PFAS 2
    ####### MODIFY #######

    # save dir for optimization direction
    optimization_save_name = 'optimization_'
    for direction in optimization_direction:
        optimization_save_name += f'{int(direction)}_'
    optimization_save_name = optimization_save_name[:-1]
    save_bo_optimization = os.path.join(bo_dir, optimization_save_name)

    # load property
    reference_polymer_property = np.load(os.path.join(save_bo_optimization, 'reference_re_encoding_property_prediction_arr.npy'))
    methyl_terminated_reference_polymer_mol = terminate_p_smi_with_CH3(reference_polymer)  # to calculate Tanimoto similarity later
    fp_methyl_terminated_reference_polymer_mol = AllChem.GetMorganFingerprintAsBitVect(
        methyl_terminated_reference_polymer_mol, radius=3, nBits=1024
    )  # for the reference chemistry

    # train polymers to get novel polymers
    df_polymer = pd.read_csv(data_path)
    train_idx = torch.load(train_idx_path)
    train_polymers = df_polymer['product'][train_idx]

    # loop
    list_df_valid_novel_unique, list_df_valid_novel_unique_opt = [], []
    percent_relative_improvement_mean_list, percent_relative_improvement_max_list, percent_relative_improvement_min_list = [], [], []
    percent_relative_improvement_mean_list_opt, percent_relative_improvement_max_list_opt, percent_relative_improvement_min_list_opt = [], [], []  # satisfying multi-objective optimization
    tanimoto_mean_list, tanimoto_max_list, tanimoto_min_list = [], [], []
    tanimoto_mean_list_opt, tanimoto_max_list_opt, tanimoto_min_list_opt = [], [], []
    for hypercube_length in hypercube_length_list:
        print(f'========== Hypercube length {hypercube_length} ==========')
        hypercube_dir = os.path.join(save_bo_optimization, f'hypercube_length_{hypercube_length}')
        discovered_polymers = pd.DataFrame()
        for bo_iter in range(1, n_iter + 1):
            bo_iteration_dir = os.path.join(hypercube_dir, f'bo_iteration_{bo_iter}')
            discovered_polymers_iteration = pd.read_csv(os.path.join(bo_iteration_dir, 'discovered_polymer.csv'))  # valid polymers
            discovered_polymers = pd.concat([discovered_polymers, discovered_polymers_iteration], axis=0)

        # drop duplicates
        df_valid_unique = discovered_polymers.drop_duplicates(subset='polymer')

        # get novel polymers
        df_valid_novel_unique = df_valid_unique[~df_valid_unique['polymer'].isin(train_polymers)]
        print(f'== The number of novel polymers : {df_valid_novel_unique.shape[0]} ==')

        # 1) calculate Optimization score
        generated_properties = df_valid_novel_unique[property_name].to_numpy()
        percent_relative_improvement = 100 * (generated_properties - reference_polymer_property) / np.abs(reference_polymer_property)
        multi_objective_bool = np.all(percent_relative_improvement > 0, axis=1)  # whether they satisfy multi-objective optimization
        print(f'== ({multi_objective_bool.sum()} / {df_valid_novel_unique.shape[0]}) satisfying multi-objective optimization criteria ==')

        # every found polymer
        percent_relative_improvement = percent_relative_improvement.mean(axis=1)  # mean value
        df_valid_novel_unique = df_valid_novel_unique.copy()
        df_valid_novel_unique['mean_relative_improvement_percent'] = percent_relative_improvement

        # only polymers satisfying multi-objective optimization
        percent_relative_improvement_opt = percent_relative_improvement[multi_objective_bool]
        df_valid_novel_unique_opt = df_valid_novel_unique[multi_objective_bool]
        df_valid_novel_unique_opt = df_valid_novel_unique_opt.copy()
        df_valid_novel_unique_opt['mean_relative_improvement_percent'] = percent_relative_improvement_opt

        # 2) calculate Tanimoto similarity to the reference polymer
        methyl_terminated_p_mol_valid_novel_unique = df_valid_novel_unique['polymer'].apply(lambda x: terminate_p_smi_with_CH3(x)).to_list()
        fp_methyl_terminated_p_mol_valid_novel_unique = [
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in methyl_terminated_p_mol_valid_novel_unique
        ]
        Tanimoto_methyl_terminated_p_mol_and_reference = np.array([
            TanimotoSimilarity(fp_methyl_terminated_reference_polymer_mol, fp) for fp in fp_methyl_terminated_p_mol_valid_novel_unique
        ])
        df_valid_novel_unique = df_valid_novel_unique.copy()
        df_valid_novel_unique['Tanimoto_methyl_terminated_p_mol_and_reference'] = Tanimoto_methyl_terminated_p_mol_and_reference

        df_valid_novel_unique_opt = df_valid_novel_unique_opt.copy()
        Tanimoto_methyl_terminated_p_mol_and_reference_multi_objective = Tanimoto_methyl_terminated_p_mol_and_reference[multi_objective_bool]
        df_valid_novel_unique_opt['Tanimoto_methyl_terminated_p_mol_and_reference'] = Tanimoto_methyl_terminated_p_mol_and_reference_multi_objective
        print(df_valid_novel_unique_opt)

        # append to the list
        if len(percent_relative_improvement) != 0:
            percent_relative_improvement_mean_list.append(percent_relative_improvement.mean())
            percent_relative_improvement_max_list.append(percent_relative_improvement.max())
            percent_relative_improvement_min_list.append(percent_relative_improvement.min())
        else:  # append all zero
            percent_relative_improvement_mean_list.append(0)
            percent_relative_improvement_max_list.append(0)
            percent_relative_improvement_min_list.append(0)

        if len(percent_relative_improvement_opt) != 0:
            percent_relative_improvement_mean_list_opt.append(percent_relative_improvement_opt.mean())
            percent_relative_improvement_max_list_opt.append(percent_relative_improvement_opt.max())
            percent_relative_improvement_min_list_opt.append(percent_relative_improvement_opt.min())
        else:  # append all zero
            percent_relative_improvement_mean_list_opt.append(0)
            percent_relative_improvement_max_list_opt.append(0)
            percent_relative_improvement_min_list_opt.append(0)

        if len(Tanimoto_methyl_terminated_p_mol_and_reference) != 0:
            tanimoto_mean_list.append(Tanimoto_methyl_terminated_p_mol_and_reference.mean())
            tanimoto_max_list.append(Tanimoto_methyl_terminated_p_mol_and_reference.max())
            tanimoto_min_list.append(Tanimoto_methyl_terminated_p_mol_and_reference.min())
        else:  # append all zero
            tanimoto_mean_list.append(0)
            tanimoto_max_list.append(0)
            tanimoto_min_list.append(0)

        if len(Tanimoto_methyl_terminated_p_mol_and_reference_multi_objective) != 0:
            tanimoto_mean_list_opt.append(Tanimoto_methyl_terminated_p_mol_and_reference_multi_objective.mean())
            tanimoto_max_list_opt.append(Tanimoto_methyl_terminated_p_mol_and_reference_multi_objective.max())
            tanimoto_min_list_opt.append(Tanimoto_methyl_terminated_p_mol_and_reference_multi_objective.min())
        else:  # append all zero
            tanimoto_mean_list_opt.append(0)
            tanimoto_max_list_opt.append(0)
            tanimoto_min_list_opt.append(0)

        list_df_valid_novel_unique.append(df_valid_novel_unique)
        list_df_valid_novel_unique_opt.append(df_valid_novel_unique_opt)

        # save
        df_valid_novel_unique_opt.to_csv(os.path.join(hypercube_dir, 'valid_novel_unique_opt.csv'), index=False)

    # plot dir
    plot_dir = Path(os.path.join(save_bo_optimization, 'plot'))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # plot - all discovered polymers
    plt.figure(figsize=(6, 6))
    plt.plot(hypercube_length_list, percent_relative_improvement_max_list, color='m', linestyle='--', marker='o')
    plt.plot(hypercube_length_list, percent_relative_improvement_mean_list, color='m', marker='o')
    plt.plot(hypercube_length_list, percent_relative_improvement_min_list, color='m', linestyle='--', marker='o')

    # annotate number of polymers discovered
    for hypercube_length, percent_relative_improvement_mean, df in zip(hypercube_length_list, percent_relative_improvement_mean_list, list_df_valid_novel_unique):
        txt = f'{df.shape[0]}'
        plt.annotate(txt, (hypercube_length, percent_relative_improvement_mean + 2), fontsize=10)
    plt.title(f'{num_generated_polymers} generated polymers', fontsize=16)
    plt.xlabel('Hypercube length (scaled)', fontsize=16)
    plt.ylabel('Optimization score (%)', fontsize=16)  # mean value
    plt.yticks([-40, -30, -20, -10, 0, 10, 20], fontsize=14)
    plt.xticks(hypercube_length_list, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'discovered_optimization_score_mean_trend.png'))
    plt.close()

    # plot - only polymers satisfying optimization criteria
    plt.figure(figsize=(6, 6))
    plt.title(f'{num_generated_polymers} generated polymers', fontsize=16)
    plt.plot(hypercube_length_list, percent_relative_improvement_max_list_opt, color='m', linestyle='--', marker='o')
    plt.plot(hypercube_length_list, percent_relative_improvement_mean_list_opt, color='m', marker='o')
    plt.plot(hypercube_length_list, percent_relative_improvement_min_list_opt, color='m', linestyle='--', marker='o')

    # annotate number of polymers discovered
    for hypercube_length, percent_relative_improvement_mean, df in zip(hypercube_length_list, percent_relative_improvement_mean_list_opt, list_df_valid_novel_unique_opt):
        txt = f'{df.shape[0]}'
        plt.annotate(txt, (hypercube_length, percent_relative_improvement_mean + 1), fontsize=10)
    plt.xlabel('Hypercube length (scaled)', fontsize=16)
    plt.ylabel('Optimization score (%)', fontsize=16)  # mean value
    plt.yticks([0, 10, 20, 30], fontsize=14)
    plt.xticks(hypercube_length_list, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'opt_optimization_score_mean_trend.png'))
    plt.close()

    # plot Tanimoto similarity - all discovered polymers
    plt.figure(figsize=(6, 6))
    plt.plot(hypercube_length_list, tanimoto_max_list, color='m', linestyle='--', marker='o')
    plt.plot(hypercube_length_list, tanimoto_mean_list, color='m', marker='o')
    plt.plot(hypercube_length_list, tanimoto_min_list, color='m', linestyle='--', marker='o')

    # annotate number of polymers discovered
    for hypercube_length, tanimoto_mean, df in zip(hypercube_length_list, tanimoto_mean_list, list_df_valid_novel_unique):
        txt = f'{df.shape[0]}'
        plt.annotate(txt, (hypercube_length, tanimoto_mean + 0.02), fontsize=10)
    plt.title(f'{num_generated_polymers} generated polymers', fontsize=16)
    plt.xlabel('Hypercube length (scaled)', fontsize=16)
    plt.ylabel('Tanimoto similarity', fontsize=16)  # mean value
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.xticks(hypercube_length_list, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'discovered_tanimoto_trend.png'))
    plt.close()

    # plot Tanimoto similarity - only polymers satisfying optimization criteria
    plt.figure(figsize=(6, 6))
    plt.plot(hypercube_length_list, tanimoto_max_list_opt, color='m', linestyle='--', marker='o')
    plt.plot(hypercube_length_list, tanimoto_mean_list_opt, color='m', marker='o')
    plt.plot(hypercube_length_list, tanimoto_min_list_opt, color='m', linestyle='--', marker='o')

    # annotate number of polymers discovered
    for hypercube_length, tanimoto_mean, df in zip(hypercube_length_list, tanimoto_mean_list_opt, list_df_valid_novel_unique_opt):
        txt = f'{df.shape[0]}'
        plt.annotate(txt, (hypercube_length, tanimoto_mean + 0.02), fontsize=10)
    plt.title(f'{num_generated_polymers} generated polymers', fontsize=16)
    plt.xlabel('Hypercube length (scaled)', fontsize=16)
    plt.ylabel('Tanimoto similarity', fontsize=16)  # mean value
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.xticks(hypercube_length_list, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'opt_tanimoto_trend.png'))
    plt.close()
