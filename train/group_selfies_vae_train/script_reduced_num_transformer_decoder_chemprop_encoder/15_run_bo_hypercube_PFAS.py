import os
import sys
import time
import numpy as np
import pandas as pd

from math import ceil
from tqdm import tqdm
from rdkit import Chem
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch

import warnings
from botorch.exceptions import BadInitialCandidatesWarning
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll

from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood

import matplotlib.pyplot as plt

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies')
from group_selfies import GroupGrammar

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML')
from vae.decoder_transformer.torch import SequenceDecoder
from vae.encoder_chemprop.chemprop_encoder import Encoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from utils.functions import terminate_p_smi_with_CH3, encoding_return_scaled_mean_prediction
from vae.utils.evaluation import tokenids_to_vocab

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


def generated_polymers(chosen_latent_space_points_arr, vae_decoder, grammar_fragment, device, dtype):
    """
    This function generates polymers from chosen latent space points.
    Note that not all of these polymers can be decoded to valid polymers.
    :params chosen_latent_space_points: np.ndarray. chosen latent space points to decode
    :return valid_latent_space_points, valid_polymers_list
    """
    # batch size
    number_of_generation = chosen_latent_space_points_arr.shape[0]
    batch_size = 100
    batch_iteration = ceil(number_of_generation / batch_size)

    # generated polymer maximum length is less than original polymers
    generated_token_list = list()

    # eval
    vae_decoder.eval()

    # generate
    with torch.no_grad():
        tqdm.write("Generation ...")
        pbar = tqdm(range(batch_iteration), total=batch_iteration, leave=True)
        time.sleep(1.0)
        with pbar as t:
            for iteration in t:
                # manual batch iterations
                start_idx = iteration * batch_size
                stop_idx = (iteration + 1) * batch_size

                if iteration == batch_iteration - 1:
                    stop_idx = number_of_generation

                # num molecules batch
                num_molecules_batch = stop_idx - start_idx

                # convert to tensor
                batch_latent_space_points_arr = chosen_latent_space_points_arr[start_idx:stop_idx]
                batch_latent_space_points_tensor = torch.tensor(batch_latent_space_points_arr, device=device, dtype=dtype)

                # beam search
                beam_size = 1
                predictions = vae_decoder.inference(batch_latent_space_points_tensor, beam_size=beam_size)

                # get decoded tokens
                decoded_tokens = [
                    tokenids_to_vocab(predictions[sample][0].tolist(), vae_decoder.vocab, grammar_fragment, convert_to_smi=False) for sample in range(num_molecules_batch)
                ]

                # extend
                generated_token_list.extend(decoded_tokens)

    # analysis
    assert number_of_generation == len(generated_token_list)

    # check valid polymers
    generated_polymer_list = []
    for decoded_tokens in generated_token_list:
        # convert to smi
        try:
            decoded_str = Chem.MolToSmiles(grammar_fragment.decoder(decoded_tokens))
        except:  # sometimes fails
            generated_polymer_list.append(None)
            continue

        # replace
        canonical_smiles_generated_molecule = Chem.CanonSmiles(decoded_str.replace("I", "*"))

        # (1) the number of '*' should be 2
        asterisk_cnt = 0
        for char in canonical_smiles_generated_molecule:
            if char == '*':
                asterisk_cnt += 1

        if asterisk_cnt != 2:
            # print(f'The number of asterisks is not 2: {canonical_smiles_generated_molecule}')
            generated_polymer_list.append(None)
            continue

        # only consisting of '**'
        if canonical_smiles_generated_molecule == '**':
            # print(f'The generated molecule is **', flush=True)
            generated_polymer_list.append(None)
            continue

        # (2) '*' should be connected through single bond
        mol = Chem.MolFromSmiles(canonical_smiles_generated_molecule)
        flag = 1
        atoms = mol.GetAtoms()
        bond_order = list()
        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                if atom.GetDegree() != 1:  # asterisk should be the end of molecule
                    flag = 0  # exclude
                    break
                else:
                    bond_order.append(atom.GetExplicitValence())
        assert len(bond_order) == 2

        # check bond order
        if bond_order[0] == 1 and bond_order[1] == 1:
            pass
        else:
            flag = 0

        # exclude error
        if flag == 0:
            print(f"The connectivity of asterisks is not valid: {canonical_smiles_generated_molecule}", flush=True)
            generated_polymer_list.append(None)
            continue

        # pass all criteria
        generated_polymer_list.append(canonical_smiles_generated_molecule)

    # get idx & polymers
    chemical_valid_idx = [idx for idx in range(number_of_generation) if generated_polymer_list[idx] is not None]
    valid_polymer_list = [polymer for polymer in generated_polymer_list if polymer is not None]
    assert len(chemical_valid_idx) == len(valid_polymer_list)

    print(f"The number of valid polymers is {len(chemical_valid_idx)} / {number_of_generation}")

    return chosen_latent_space_points_arr[chemical_valid_idx], valid_polymer_list


# fit GP model
def fit_model(train_z_arr, predict_y_arr, device):
    """
    This function fits GP.
    :return model_list_gp, z_scaler, y_scaler
    """
    num_properties = predict_y_arr.shape[1]

    # scale
    z_scaler = MinMaxScaler()  # to be in the unit cube
    y_scaler = StandardScaler()  # standard scaler
    train_z_arr_scaled = z_scaler.fit_transform(train_z_arr)
    predict_y_arr_scaled = y_scaler.fit_transform(predict_y_arr)

    train_z_tensor = torch.tensor(train_z_arr_scaled, device=device)
    predict_y_tensor = torch.tensor(predict_y_arr_scaled, device=device)
    models = [SingleTaskGP(train_z_tensor, predict_y_tensor[:, idx].view(-1, 1)) for idx in range(num_properties)]
    for model in models:
        model.train()
    model_list_gp = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list_gp.likelihood, model_list_gp)
    fit_gpytorch_mll(mll)

    return model_list_gp, z_scaler, y_scaler


def optimize_qnehvi_and_get_observation(model, ref_point, train_z_tensor, qnehvi_sampler, inequality_constraints):
    """
    Optimizes the qEHVI acquisition function, and returns a new candidate and observation.
    """
    # Acquisition function: qNEHVI (for multi-objective)
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,  # ModelListGP. takes in scaled z and returns scaled y
        ref_point=ref_point,  # scaled in the outcome space
        X_baseline=train_z_tensor,
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=qnehvi_sampler,
    )

    # TODO - Tune parameters + direction for optimization / inequality condition for the hypercube
    # Maybe Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO) (?)
    # q : 1. 10 -> doable -> 100 -> so much memory usage
    # Optimize acquisition function
    new_z, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([torch.zeros(latent_dim), torch.ones(latent_dim)]),  # (2, latent_dim)
        q=bo_batch_size,
        num_restarts=num_restarts,  # The number of starting points for multistart acquisition function optimization.
        raw_samples=raw_samples,  # The number of samples for initialization
        sequential=True,  # uses sequential optimization
        inequality_constraints=inequality_constraints,
    )

    # inverse-transform
    new_z_for_vae_arr = initial_z_scaler.inverse_transform(new_z.detach().cpu().numpy())

    # observe new values
    new_z_valid_arr, valid_polymer_list = generated_polymers(
        chosen_latent_space_points_arr=new_z_for_vae_arr,
        vae_decoder=vae_decoder,
        grammar_fragment=grammar_fragment, device=device, dtype=dtype
    )
    # (scaled) property prediction
    re_encoding_property_prediction_arr_scaled = black_box_function(
        valid_polymer_list=valid_polymer_list,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        property_predictor=property_predictor,
        grammar_fragment=grammar_fragment
    )  # (1, num_data, num_property)
    re_encoding_property_prediction_arr = property_train_scaler.inverse_transform(re_encoding_property_prediction_arr_scaled[0])  # (num_data, num_property)
    monomer_phi = re_encoding_property_prediction_arr[:, monomer_phi_property_idx].reshape(-1, 1)
    lumo_energy = re_encoding_property_prediction_arr[:, lumo_energy_property_idx].reshape(-1, 1)
    homo_energy = re_encoding_property_prediction_arr[:, homo_energy_property_idx].reshape(-1, 1)
    chi_water = re_encoding_property_prediction_arr[:, chi_water_property_idx].reshape(-1, 1)

    num_fluorine = np.array([get_num_fluorine(p_smi) for p_smi in valid_polymer_list]).reshape(-1, 1)  # unscaled

    targeted_re_encoding_property_prediction_arr = np.hstack([monomer_phi, lumo_energy - homo_energy, chi_water, num_fluorine])
    targeted_re_encoding_property_prediction_arr *= optimization_direction  # BoTorch assumes maximization

    return new_z_valid_arr, targeted_re_encoding_property_prediction_arr, valid_polymer_list


if __name__ == '__main__':
    """
    This script implements Bayesian optimization for PFAS replacement. 
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

    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # property idx & property name
    monomer_phi_property_idx = 0
    lumo_energy_property_idx = 2
    homo_energy_property_idx = 1
    chi_water_property_idx = 8
    property_name = ['monomer_phi', 'homo_lumo_gap', 'chi_water', 'num_fluorine']

    # monomer property optimization direction
    optimization_direction = np.array([-1, 1, 1, -1])  # decrease / increase / increase / decrease

    # dir to load the initial training data
    k = 2000
    gpr_dir = os.path.join(model_check_point_directory, 'gpr')
    initial_train_dir = os.path.join(gpr_dir, f'k_means_num_{k}_PFAS')
    bo_dir = os.path.join(gpr_dir, 'bo', f'k_means_num_{k}_PFAS_2')

    ### 1) set the reference PFAS polymer - side-chain fluorinated polymers
    # reference_polymer = '*OCCOC(=O)c1ccc(C(*)=O)c(COC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)c1'

    ### 2) set the reference PFAS polymer - Fluoropolymer - a carbon atom backbone with fluorine atoms
    reference_polymer = '*OCCOC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OCCOC(=O)Nc1ccc(Cc2ccc(NC(*)=O)cc2)cc1'

    # BoTorch setting
    bo_batch_size = 20  # batch per iteration
    n_iter = 5  # iter for bo
    mc_samples = 128  # approximate the acquisition function
    num_restarts = 10
    raw_samples = 512

    # hypercube length to explore
    hypercube_length_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # scaled
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

    # preparation training data for scaling
    property_value = torch.load(os.path.join(data_dir, 'property_value.pth'))
    train_idx_for_property_scaling = torch.load(os.path.join(data_dir, 'train_idx.pth'))  # whole train_idx for train property scaling
    property_train_for_property_scaling = property_value[train_idx_for_property_scaling]
    property_train_scaler = StandardScaler()
    property_train_scaler = property_train_scaler.fit(property_train_for_property_scaling)

    # eval
    vae_encoder.eval()
    vae_decoder.eval()
    property_predictor.eval()

    ### get the property of the reference polymer for the replacement - maintaining properties at the same time replacing Fluroine.
    # (scaled) property prediction for the reference point
    reference_re_encoding_property_prediction_arr_scaled, reference_mu_arr = black_box_function(
        valid_polymer_list=[reference_polymer],
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        property_predictor=property_predictor,
        grammar_fragment=grammar_fragment,
        return_mus=True
    )  # (1, num_data, num_property)
    reference_re_encoding_property_prediction_arr = property_train_scaler.inverse_transform(reference_re_encoding_property_prediction_arr_scaled[0])
    monomer_phi = reference_re_encoding_property_prediction_arr[0][monomer_phi_property_idx]
    lumo_energy = reference_re_encoding_property_prediction_arr[0][lumo_energy_property_idx]
    homo_energy = reference_re_encoding_property_prediction_arr[0][homo_energy_property_idx]
    chi_water = reference_re_encoding_property_prediction_arr[0][chi_water_property_idx]

    # get the number of Fluroine
    def get_num_fluorine(smi: str) -> int:
        """Return the number of fluorine atoms in a molecule given its SMILES string."""
        mol = Chem.MolFromSmiles(smi)
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    num_fluorine = get_num_fluorine(reference_polymer)

    targeted_reference_re_encoding_property_prediction_arr = np.array([
        monomer_phi, lumo_energy - homo_energy, chi_water, num_fluorine
    ])  # unscaled
    targeted_reference_re_encoding_property_prediction_arr *= optimization_direction  # BoTorch assumes maximization

    # prepare an initial training data - unscaled
    initial_train_z = np.load(os.path.join(initial_train_dir, 'sampled_z_mean.npy'))
    initial_train_y = np.load(os.path.join(initial_train_dir, 'sampled_properties.npy'))  # (num_data, 4)

    # valid data - unscaled
    test_z = np.load(os.path.join(gpr_dir, 'valid_z_mean.npy'))[:100]
    test_y = np.load(os.path.join(gpr_dir, 'valid_property_PFAS.npy'))[:100]  # (num_data, 4)
    latent_dim = initial_train_z.shape[1]

    # modify initial train y based on the optimization direction - BoTorch assumes "maximization"
    initial_train_y *= optimization_direction
    test_y *= optimization_direction

    # save dir for optimization direction
    optimization_save_name = 'optimization_'
    for direction in optimization_direction:
        optimization_save_name += f'{int(direction)}_'
    optimization_save_name = optimization_save_name[:-1]
    save_bo_optimization = Path(os.path.join(bo_dir, optimization_save_name))
    save_bo_optimization.mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(save_bo_optimization, 'reference_re_encoding_property_prediction_arr'), targeted_reference_re_encoding_property_prediction_arr)

    # Apply BO for hypercube
    for hypercube_length in hypercube_length_list:
        print(f'======= Hypercube length (scaled): {hypercube_length} =======')

        # prepare initial surrogate models
        model_list_gp, initial_z_scaler, initial_y_scaler = fit_model(train_z_arr=initial_train_z, predict_y_arr=initial_train_y, device=device)

        # save the initial model
        hypercube_dir = os.path.join(save_bo_optimization, f'hypercube_length_{hypercube_length}')
        initial_bo_dir = Path(os.path.join(hypercube_dir, 'initial_train'))
        initial_bo_dir.mkdir(exist_ok=True, parents=True)

        # initial training results - cpu
        cpu_device = 'cpu'
        with torch.no_grad():
            model_list_gp = ModelListGP(*[m.to(cpu_device) for m in model_list_gp.models])  # to cpu
            means, stds = [], []
            test_z_scaled = initial_z_scaler.transform(test_z)
            test_z_tensor = torch.tensor(test_z_scaled, device=cpu_device)  # to cpu
            for sub_model in model_list_gp.models:
                sub_model.eval()
                posterior = sub_model.posterior(test_z_tensor)
                means.append(posterior.mean)
                stds.append(posterior.variance.sqrt())

            # Stack results: shape (n_outputs, n_test, 1)
            means = torch.stack(means).numpy()  # (n_outputs, n_test, 1)
            stds = torch.stack(stds).numpy()  # (n_outputs, n_test, 1)

            # un-scale
            mean_arr = initial_y_scaler.inverse_transform(means[:, :, 0].T)  # (n_test, n_outputs)
            std_arr = stds[:, :, 0].T * initial_y_scaler.scale_  # (n_test, n_outputs)

            # plot
            num_properties = mean_arr.shape[1]
            for idx in range(num_properties):
                mean_pred = mean_arr[:, idx]
                std_pred = std_arr[:, idx]

                plt.figure(figsize=(6, 6))
                plt.errorbar(test_y[:, idx], mean_pred, yerr=std_pred, fmt='o')
                plt.plot([mean_pred.min(), mean_pred.max()], [mean_pred.min(), mean_pred.max()], ls='--', color='gray')
                plt.xlabel('Test')
                plt.ylabel('Pred')
                plt.savefig(os.path.join(initial_bo_dir, f'{property_name[idx]}.png'))
                plt.close()

        # scale to property for GP & BO
        targeted_reference_re_encoding_property_prediction_arr_scaled = initial_y_scaler.transform(
            targeted_reference_re_encoding_property_prediction_arr.reshape(1, -1)
        ).flatten()

        # scale z for GP & BO
        reference_mu_arr_scaled = initial_z_scaler.transform(reference_mu_arr).flatten()
        assert np.all((reference_mu_arr_scaled >= 0) & (reference_mu_arr_scaled <= 1))  # check the training regime

        # make inequality constraints base on hypercube length and z-scaler
        inequality_constraints = []
        lower_bound_arr_scaled = reference_mu_arr_scaled - hypercube_length / 2  # set the boundary
        upper_bound_arr_scaled = reference_mu_arr_scaled + hypercube_length / 2  # set the boundary
        # check if all exceeds [0, 1]
        if np.all(lower_bound_arr_scaled < 0) & np.all(upper_bound_arr_scaled > 1):
            print(f'This is the final hypercube length! {hypercube_length}')
        # clamp to [0, 1]
        lower_bound_to_use = np.clip(lower_bound_arr_scaled, 0, 1)
        upper_bound_to_use = np.clip(upper_bound_arr_scaled, 0, 1)
        # construct BoTorch linear inequality constraints (A x ≥ b)
        for dim in range(latent_dim):
            inequality_constraints.append(
                (torch.tensor([dim]), torch.tensor([1.0]), torch.tensor([lower_bound_to_use[dim]]))
            )
            inequality_constraints.append(
                (torch.tensor([dim]), torch.tensor([-1.0]), torch.tensor([-upper_bound_to_use[dim]]))
            )

        # Bayesian Optimization
        model_list_gp = ModelListGP(*[m.to(cpu_device) for m in model_list_gp.models])  # to cpu
        initial_train_z_scaled = initial_z_scaler.transform(initial_train_z)
        observed_z_tensor = torch.tensor(initial_train_z_scaled, device=cpu_device)  # [0, 1] scaling
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))  # define the qNEI acquisition modules using a QMC sampler

        # to compute hypervolume
        hvs_list = list()
        ref_point = torch.tensor(targeted_reference_re_encoding_property_prediction_arr_scaled, device=cpu_device, dtype=torch.float64)  # float64 for BO
        Y_to_compute_volume = torch.tensor(targeted_reference_re_encoding_property_prediction_arr_scaled.reshape(1, -1), device=cpu_device, dtype=torch.float64)  # float64 for BO
        observed_y_tensor_scaled = torch.tensor(initial_y_scaler.transform(initial_train_y), device=cpu_device)

        for i in range(n_iter):
            new_z_valid_arr, targeted_re_encoding_property_prediction_arr, found_valid_polymer_list = optimize_qnehvi_and_get_observation(
                model=model_list_gp, ref_point=ref_point, train_z_tensor=observed_z_tensor, qnehvi_sampler=qnehvi_sampler,
                inequality_constraints=inequality_constraints
            )

            # scale with the initial scaler
            new_z_valid_arr_scaled = initial_z_scaler.transform(new_z_valid_arr)
            targeted_re_encoding_property_prediction_arr_scaled = initial_y_scaler.transform(targeted_re_encoding_property_prediction_arr)
            targeted_re_encoding_property_prediction_tensor_scaled = torch.tensor(targeted_re_encoding_property_prediction_arr_scaled, device=cpu_device, dtype=torch.float64)  # for BO & GP

            with torch.no_grad():
                # compute hypervolume
                Y_to_compute_volume = torch.cat([Y_to_compute_volume, targeted_re_encoding_property_prediction_tensor_scaled], dim=0)
                bd = DominatedPartitioning(ref_point=ref_point, Y=Y_to_compute_volume)
                volume = bd.compute_hypervolume().item()
                print(f"The BO iter {i + 1} hypervolume determined by the Pareto front: {volume}")  # only consider the reference point
                hvs_list.append(volume)

                # cat y tensor
                observed_y_tensor_scaled = torch.cat([observed_y_tensor_scaled, targeted_re_encoding_property_prediction_tensor_scaled], dim=0)

                # cat z tensor - update training points
                new_z_valid_tensor = torch.tensor(new_z_valid_arr_scaled, device=cpu_device, dtype=torch.float64)  # for GP
                observed_z_tensor = torch.cat([observed_z_tensor, new_z_valid_tensor], dim=0)

                # model prediction
                model_list_gp = ModelListGP(*[m.to(cpu_device) for m in model_list_gp.models])  # to cpu
                means, stds = [], []
                for sub_model in model_list_gp.models:
                    sub_model.eval()
                    posterior = sub_model.posterior(new_z_valid_tensor)
                    means.append(posterior.mean)
                    stds.append(posterior.variance.sqrt())

                # Stack results: shape (n_outputs, n_test, 1)
                means = torch.stack(means).numpy()  # (n_outputs, n_test, 1)
                stds = torch.stack(stds).numpy()  # (n_outputs, n_test, 1)

                # un-scale
                mean_arr = initial_y_scaler.inverse_transform(means[:, :, 0].T)  # (n_test, n_outputs)
                std_arr = stds[:, :, 0].T * initial_y_scaler.scale_  # (n_test, n_outputs)

                # plot
                bo_iter_dir = Path(os.path.join(hypercube_dir, f'bo_iteration_{i + 1}'))
                bo_iter_dir.mkdir(exist_ok=True, parents=True)
                num_properties = mean_arr.shape[1]
                for idx in range(num_properties):
                    mean_pred = mean_arr[:, idx]
                    std_pred = std_arr[:, idx]

                    plt.figure(figsize=(6, 6))
                    plt.errorbar(targeted_re_encoding_property_prediction_arr[:, idx], mean_pred, yerr=std_pred, fmt='o')
                    plt.plot([mean_pred.min(), mean_pred.max()], [mean_pred.min(), mean_pred.max()], ls='--', color='gray')
                    plt.xlabel('Test')
                    plt.ylabel('Pred')
                    plt.tight_layout()
                    plt.savefig(os.path.join(bo_iter_dir, f'{property_name[idx]}.png'))
                    plt.close()

            # re-train the model
            models = [SingleTaskGP(observed_z_tensor, observed_y_tensor_scaled[:, idx].view(-1, 1)) for idx in range(num_properties)]
            for model in models:
                model.train()
                model.to(cpu_device)  # to cpu
            model_list_gp = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model_list_gp.likelihood, model_list_gp)
            fit_gpytorch_mll(mll)

            # save generated polymers
            z_coordinates_name = [f'z_{dim + 1}' for dim in range(latent_dim)]
            df_save = pd.DataFrame(columns=['polymer'] + property_name + z_coordinates_name)
            df_save['polymer'] = found_valid_polymer_list
            for property_idx in range(num_properties):
                df_save[property_name[property_idx]] = targeted_re_encoding_property_prediction_arr[:, property_idx]
            df_save[z_coordinates_name] = new_z_valid_arr
            df_save.to_csv(os.path.join(bo_iter_dir, 'discovered_polymer.csv'), index=False)
