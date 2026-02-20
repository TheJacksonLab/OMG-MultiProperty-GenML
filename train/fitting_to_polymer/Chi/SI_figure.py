import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import pearsonr


if __name__ == '__main__':
    # experimental chi
    df_experimental = pd.read_csv('./data/reference_chi_canon.csv')

    # add prediction values to df_experimental
    df_pred = pd.read_csv('./chi_results/chi_prediction_0_025_solute_toluene_cosmo_solvent_cosmo_cosmo.csv')
    df_experimental['pred_chi'] = df_pred['chi_parameter_mean'].to_numpy()
    df_experimental['std_chi'] = df_pred['chi_parameter_std'].to_numpy()

    # condition on a volume fraction to avoid a critical regime
    df_experimental = df_experimental[df_experimental['polymer_volume_fraction'] >= 0.2]

    # df_experimental = df_experimental[df_experimental['solvent'] == 'water']
    # print(df_experimental)

    # prepare data to plot
    experimental_chi_arr = df_experimental['experimental_chi'].to_numpy()
    pred_chi_arr = df_experimental['pred_chi'].to_numpy()
    std_chi_arr = df_experimental['std_chi'].to_numpy()
    solvent_arr = df_experimental['solvent'].to_numpy(dtype='str')

    # plot
    solvent_list = ['water']
    color_list = ['#4357AD']
    assert len(solvent_list) == len(color_list)
    for solvent_choice, color in zip(solvent_list, color_list):
        plt.figure(figsize=(3.25, 3.25), dpi=300)
        pred_chi_arr_solvent = pred_chi_arr[solvent_arr == solvent_choice]
        experimental_chi_arr_solvent = experimental_chi_arr[solvent_arr == solvent_choice]

        # linear correlation
        pearsonr_object = pearsonr(x=pred_chi_arr_solvent, y=experimental_chi_arr_solvent)
        linear_correlation = pearsonr_object.statistic

        plt.scatter(pred_chi_arr_solvent, experimental_chi_arr_solvent, color=color, s=20.0, alpha=0.5,
                    label=f'Linear correlation $\\rho$ = {linear_correlation:.3f}')
        plt.legend(fontsize=7, loc='lower right')
        plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], fontsize=10)
        plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], fontsize=10)
        plt.xlabel('$\chi_{water}$ (COSMO-SAC) [unitless]', fontsize=10)
        plt.ylabel('Experimental $\chi_{water}$ [unitless]', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'./plot/SI_chi_pred_{solvent_choice}.png')
        plt.savefig(f'./plot/SI_chi_pred_{solvent_choice}.svg', format='svg', dpi=1200)
        plt.close()

        # load linear fit
        model_fit_dir = '/home/sk77/PycharmProjects/omg_multi_task/train/fitting_to_polymer'
        model_fit_json = os.path.join(model_fit_dir, 'Chi', 'model_fit', 'water_chi.json')
        def linear_function(x, gradient, bias):  # linear function
            return gradient * x + bias
        with open(model_fit_json, "r") as f:  # load json
            params = json.load(f)
            f.close()
        plt.figure(figsize=(3.25, 3.25), dpi=300)
        fitted_linear = lambda x: linear_function(x, **params)  # define function
        chi_exp_predicted = fitted_linear(pred_chi_arr_solvent)
        r2_linear_fit = r2_score(y_true=experimental_chi_arr_solvent, y_pred=chi_exp_predicted)
        plot_min = min(chi_exp_predicted.min(), experimental_chi_arr_solvent.min())  # y=x plot
        plot_max = max(chi_exp_predicted.max(), experimental_chi_arr_solvent.max())  # y=x plot
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'grey', linestyle='--')
        plt.scatter(chi_exp_predicted, experimental_chi_arr_solvent, color=color, s=20.0,
                    label=f'Linear fitting ($R^2$={r2_linear_fit:.3f})', alpha=0.5)
        plt.legend(fontsize=8, loc='upper left')
        plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], fontsize=10)
        plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], fontsize=10)
        plt.xlabel('Prediction $\chi_{water}$ [unitless]', fontsize=10)
        plt.ylabel('Experimental $\chi_{water}$ [unitless]', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'./plot/SI_prediction_chi_pred_{solvent_choice}.png')
        plt.savefig(f'./plot/SI_prediction_chi_pred_{solvent_choice}.svg', format='svg', dpi=1200)
        plt.close()
