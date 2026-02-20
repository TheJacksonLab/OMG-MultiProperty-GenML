import os
import json
import pandas as pd

if __name__ == '__main__':
    """
    This script fits given monomer-level properties to polymer properties.
    """
    ######## MODIFY ########
    # monomer properties multiplied by optimization direction from BoTorch
    # SMI: *OCC=C1C(C)=CC(COC(=O)C(CCCCc2ccccc2)C(*)=O)=C1C
    # predicted_monomer_phi = -0.32432914 * (-1)
    # predicted_homo_lumo_gap = 3.1242347 * (1)
    # predicted_chi_water = 1.4344972 * (1)

    # SMI: *OCC(C=Cc1ccccc1)COC(=O)C(CCC1C=Cc2ccccc21)C(*)=O
    # predicted_monomer_phi = -0.27188691 * (-1)
    # predicted_homo_lumo_gap = 3.2951105 * (1)
    # predicted_chi_water = 1.3535551 * (1)

    # SMI: *OCC(C)(COC(=O)C(CCCc1ccsc1)C(*)=O)c1ccccn1 - search length 0.2
    # predicted_monomer_phi = -0.296135 * (-1)
    # predicted_homo_lumo_gap = 3.4828005 * (1)
    # predicted_chi_water = 1.3367898 * (1)

    # SMI: *OCC(COC(=O)C(CCc1ccccc1)C(*)=O)c1ccc(C(C)(C)C)cc1 - search length 0.3
    # predicted_monomer_phi = -0.28924674 * (-1)
    # predicted_homo_lumo_gap = 3.5943375 * (1)
    # predicted_chi_water = 1.4236093 * (1)

    # SMI: *Oc1cc(C)cc(OC(=O)C(CC)(C(*)=O)c2ccccc2)c1C - search length 0.4
    # predicted_monomer_phi = -0.2110603 * (-1)
    # predicted_homo_lumo_gap = 3.3786888 * (1)
    # predicted_chi_water = 1.3717616 * (1)

    # PFAS 1 (reference)
    # predicted_monomer_phi = -0.26696256 * (-1)
    # predicted_homo_lumo_gap = 2.77699351 * (1)
    # predicted_chi_water = 1.38700807 * (1)

    # generated PFAS 1 replacement - SMI: *Oc1cc(C(F)(F)F)c(OC(=O)C(CCC2=CC=CC=CC(C)(C)C2)C(*)=O)cc1F
    # predicted_monomer_phi = -0.264355331659317 * (-1)
    # predicted_homo_lumo_gap = 2.8132264614105225 * (1)
    # predicted_chi_water = 1.4603437185287476 * (1)

    # generated PFAS 1 replacement - SMI: *OCC1(C)CC(C)(C(C)C)C1OC(=O)C(Cc1ccccc1)C(*)=O
    # predicted_monomer_phi = -0.2345505952835083 * (-1)
    # predicted_homo_lumo_gap = 3.69116735458374 * (1)
    # predicted_chi_water = 1.4602943658828735 * (1)

    # generated PFAS 1 replacement: SMI: *OCC1CC2(C)CC(OC(=O)C3CCCCC3(C)C(*)=O)CC2(c2ccccc2)C1
    # predicted_monomer_phi = -0.1815603077411651 * (-1)
    # predicted_homo_lumo_gap = 3.8424534797668457 * (1)
    # predicted_chi_water = 1.4890081882476809 * (1)

    # PFAS 2 (reference): SMI: *OCCOC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OCCOC(=O)Nc1ccc(Cc2ccc(NC(*)=O)cc2)cc1
    # predicted_monomer_phi = -0.26316482 * (-1)
    # predicted_homo_lumo_gap = 3.77131414 * (1)
    # predicted_chi_water = 1.20379698 * (1)

    # generated PFAS 2 replacement: SMI: *OCC(C)(C)C(C)(COC(=O)Nc1ccccc1Cc1ccccc1NC(*)=O)C(F)(F)F
    # predicted_monomer_phi = -0.2502015829086303 * (-1)
    # predicted_homo_lumo_gap = 3.933802604675293 * (1)
    # predicted_chi_water = 1.2779347896575928 * (1)

    # generated PFAS 2 replacement: SMI: *OC(c1ccc(C)cc1)C(C)(COC(=O)NC12CCCC(C(*)=O)(C1)C2)C(C)C
    # predicted_monomer_phi = -0.2127707749605178 * (-1)
    # predicted_homo_lumo_gap = 3.873133420944214 * (1)
    # predicted_chi_water = 1.373024821281433 * (1)

    # generated PFAS 2 replacement: SMI: *OC(C(=O)OCC(C)C)C(OC(=O)C12CCCC(C(*)=O)(C1)C2)C(=O)OCc1ccccc1
    # predicted_monomer_phi = -0.2391656041145324 * (-1)
    # predicted_homo_lumo_gap = 3.947654962539673 * (1)
    # predicted_chi_water = 1.310472846031189 * (1)

    # Search length 0.2 - (1): SMI: *OCCN(CCOC(=O)C(CCCc1ccccc1)C(*)=O)c1ccccn1
    predicted_monomer_phi = -0.33547062 * (-1)
    predicted_homo_lumo_gap = 2.9436283 * (1)
    predicted_chi_water = 1.3490932 * (1)
    ######## MODIFY ########

    # set dir for fitting to polymer properties
    print('================')
    model_fit_dir = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer'

    # experimental Tg for min / max filtering
    df_Tg = pd.read_csv('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer/Tg/normalized_phi_estimation.csv')
    experimental_Tg_arr = df_Tg['experimental_Tg'].to_numpy()
    Tg_min, Tg_max = experimental_Tg_arr.min(), experimental_Tg_arr.max()

    # load model to fit monomer phi to Tg for later
    model_fit_json_monomer_phi = os.path.join(model_fit_dir, 'Tg', 'model_fit', 'inverse_params.json')

    def inverse_function(x, a, b, c):  # inverse function
        return a / (x + b) + c

    with open(model_fit_json_monomer_phi, "r") as f:  # load json
        monomer_phi_params = json.load(f)
        f.close()
    monomer_phi_fitted_inverse = lambda x: inverse_function(x, **monomer_phi_params)  # define function
    fitted_experimental_Tg = monomer_phi_fitted_inverse(predicted_monomer_phi)
    assert fitted_experimental_Tg > Tg_min
    assert fitted_experimental_Tg < Tg_max
    print(f"Experimental glass transition temperature (K): {fitted_experimental_Tg:.2f}")

    # 2) Band gap
    # filter based on the min & max value of computational band gap
    df_polymer_gap = pd.read_csv('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/train/fitting_to_polymer/gap_bulk/data/Egb.csv')
    polymer_gap_arr = df_polymer_gap['value'].to_numpy()
    band_gap_min, band_gap_max = polymer_gap_arr.min(), polymer_gap_arr.max()

    # fit homo lumo gap to computational band gap for later
    model_fit_json_band_gap = os.path.join(model_fit_dir, 'gap_bulk', 'model_fit', 'linear_params.json')

    def linear_function(x, gradient, bias):  # linear function
        return gradient * x + bias

    with open(model_fit_json_band_gap, "r") as f:  # load json
        band_gap_params = json.load(f)
        f.close()
    fitted_linear_band_gap = lambda x: linear_function(x, **band_gap_params)  # define function
    fitted_polymer_band_gap = fitted_linear_band_gap(predicted_homo_lumo_gap)
    assert fitted_polymer_band_gap > band_gap_min
    assert fitted_polymer_band_gap < band_gap_max
    print(f"Polymer band gap (eV): {fitted_polymer_band_gap:.2f}")

    # 3) Chi water - "not" filter based on the mix & max value of chi water (experiment) -> too small data size
    # fit chi_water to experimental chi_water
    model_fit_json_chi_water = os.path.join(model_fit_dir, 'Chi', 'model_fit', 'water_chi.json')
    with open(model_fit_json_chi_water, "r") as f:  # load json
        chi_water_params = json.load(f)
        f.close()
    fitted_linear_chi_water = lambda x: linear_function(x, **chi_water_params)  # define function
    fitted_experimental_chi_water = fitted_linear_chi_water(predicted_chi_water)
    print(f"Experimental Chi water: {fitted_experimental_chi_water:.2f}")
    print('================')
