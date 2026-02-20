import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# Define the inverse function
def inverse_function(x, a, b, c):
    return a / (x + b) + c


if __name__ == '__main__':
    df = pd.read_csv('./normalized_phi_estimation.csv')  # (315, 4) -> ['canon_p_smi', 'experimental_Tg', 'polymer_phi', 'polymer_backbone_phi']

    # correlation
    experimental_Tg_arr = df['experimental_Tg'].to_numpy()
    normalized_monomer_phi_arr = df['normalized_monomer_phi'].to_numpy()

    # plot - Tg versus polymer phi
    color = '#586A6A'
    pearson_r = pearsonr(x=normalized_monomer_phi_arr, y=experimental_Tg_arr)
    linear_correlation = pearson_r.statistic
    plt.figure(figsize=(3.25, 3.25), dpi=300)
    plt.scatter(normalized_monomer_phi_arr, experimental_Tg_arr, color=color, alpha=0.5, s=10.0,
                label=f'Linear correlation $\\rho$ = {linear_correlation:.3f}')
    plt.xlabel('$\Phi_\mathrm{mon}$ [unitless]', fontsize=10)
    plt.ylabel('Experimental $T_g$ [K]', fontsize=10)
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10)
    plt.yticks([200, 300, 400, 500, 600, 700], fontsize=10)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('./plot/SI_correlation_normalized_monomer_phi.png')
    plt.savefig('./plot/SI_correlation_normalized_monomer_phi.svg', format='svg', dpi=1200)
    plt.close()

    # To load a model
    def inverse_function(x, a, b, c):
        return a / (x + b) + c
    with open('./model_fit/inverse_params.json', "r") as f:
        params = json.load(f)
    fitted_inverse = lambda x: inverse_function(x, **params)

    # plot a prediction to see how accurate inverse fitting is
    pred_Tg = fitted_inverse(normalized_monomer_phi_arr)

    # plot of predicted Tg
    plt.figure(figsize=(3.25, 3.25), dpi=300)
    x, y = pred_Tg, experimental_Tg_arr
    r2 = r2_score(y_true=y, y_pred=x)
    plt.scatter(x, y, color=color, alpha=0.5, marker='o', s=10.0, label=f'Prediction $R^2$ = {r2:0.3f}')

    # y=x
    plot_min = min(x.min(), y.min())
    plot_max = max(x.max(), y.max())
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'grey', linestyle='--')

    plt.xlabel('Prediction $T_g$ [K]', fontsize=10)
    plt.ylabel('Experimental $T_g$ [K]', fontsize=10)
    # plt.title('Prediction of $T_g$ using $\Phi_\mathrm{mon}$', fontsize=10)
    plt.xticks([200, 300, 400, 500, 600, 700], fontsize=10)
    plt.yticks([200, 300, 400, 500, 600, 700], fontsize=10)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('./plot/SI_Tg_normalized_phi_inverse_fit.png')
    plt.savefig('./plot/SI_Tg_normalized_phi_inverse_fit.svg', format='svg', dpi=1200)
    plt.close()
