import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

if __name__ == '__main__':
    ###### MODIFY ######
    polymer_property = 'Egb'
    ML_property = 'HOMO_LUMO_gap'
    ML_plot_name = 'HOMO-LUMO gap [eV]'
    SAVE_DIR = './data'
    ###### MODIFY ######

    # pred
    df_pred_path = os.path.join(SAVE_DIR, 'pred', 'AL3_test_pred_total_merged.csv')
    df_pred = pd.read_csv(df_pred_path)

    # calculate gap
    df_pred['HOMO_LUMO_gap'] = df_pred['LUMO_Boltzmann_average'].to_numpy() - df_pred['HOMO_Boltzmann_average'].to_numpy()

    # polymer data
    df_polymer_path = os.path.join(SAVE_DIR, f'{polymer_property}.csv')
    df_polymer = pd.read_csv(df_polymer_path)

    # plot
    polymer_arr = df_polymer['value'].to_numpy()
    ML_monomer_arr = df_pred[ML_property].to_numpy()

    # linear correlation
    pearsonr_object = pearsonr(x=polymer_arr, y=ML_monomer_arr)
    linear_correlation = pearsonr_object.statistic

    color = '#EF8354'
    plt.figure(figsize=(3.25, 3.25), dpi=300)
    plt.scatter(ML_monomer_arr, polymer_arr, color=color, s=10.0, alpha=0.5,
                label=f'Linear correlation $\\rho$ = {linear_correlation:.3f}')
    plt.xlabel(f'Monomer {ML_plot_name}', fontsize=10)  # ML prediction
    plt.ylabel(f'Polymer $E_g$ [eV]', fontsize=10)  # bulk
    plt.xticks([0, 2, 4, 6, 8, 10], fontsize=10)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=10)
    plt.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('./plot', f'SI_{polymer_property}.png'))
    plt.savefig(os.path.join('./plot', f'SI_{polymer_property}.svg'), format='svg', dpi=1200)
    plt.close()

    # load linear fit
    model_fit_dir = '/home/sk77/PycharmProjects/omg_multi_task/train/fitting_to_polymer'
    model_fit_json = os.path.join(model_fit_dir, 'gap_bulk', 'model_fit', 'linear_params.json')
    def linear_function(x, gradient, bias):  # linear function
        return gradient * x + bias
    with open(model_fit_json, "r") as f:  # load json
        params = json.load(f)
        f.close()
    fitted_linear = lambda x: linear_function(x, **params)  # define function

    # linear fit
    plt.figure(figsize=(3.25, 3.25), dpi=300)

    linear_predicted_arr = fitted_linear(ML_monomer_arr)
    r2_linear_fit = r2_score(y_true=polymer_arr, y_pred=linear_predicted_arr)
    plt.scatter(linear_predicted_arr, polymer_arr, color=color, label=f'Prediction ($R^2$={r2_linear_fit:.3f})',
                alpha=0.5, s=10.0)

    # min max
    plot_min = np.min([polymer_arr.min(), linear_predicted_arr.min()])
    plot_max = np.max([polymer_arr.max(), linear_predicted_arr.max()])
    plt.plot([plot_min, plot_max], [plot_min, plot_max], color='grey', linestyle='--', alpha=0.5)
    plt.xticks([0, 2, 4, 6, 8, 10], fontsize=10)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=10)
    plt.xlabel(f'Prediction $E_g$ [eV]', fontsize=10)
    plt.ylabel(f'Polymer $E_g$ [eV]', fontsize=10)
    plt.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('./plot', f'SI_prediction_{polymer_property}.png'))
    plt.savefig(os.path.join('./plot', f'SI_prediction_{polymer_property}.svg'), format='svg', dpi=1200)
    plt.close()
