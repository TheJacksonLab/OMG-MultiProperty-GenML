import os
import numpy as np

from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from typing import Iterable, Set

import sys
PATH_TO_GROUP_SELFIES = '/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/group-selfies'
sys.path.append(PATH_TO_GROUP_SELFIES)
from group_selfies import fragment_mols, GroupGrammar, Group


def split_selfies(selfies: str):

    left_idx = selfies.find("[")

    while 0 <= left_idx < len(selfies):
        right_idx = selfies.find("]", left_idx + 1)
        if right_idx == -1:
            raise ValueError("malformed SELFIES string, hanging '[' bracket")

        next_symbol = selfies[left_idx: right_idx + 1]
        yield next_symbol

        left_idx = right_idx + 1
        if selfies[left_idx: left_idx + 1] == ".":
            yield "."
            left_idx += 1


def get_alphabet_from_selfies(selfies_iter: Iterable[str]) -> Set[str]:
    """Constructs an alphabet from an iterable of SELFIES strings.

    The returned alphabet is the set of all symbols that appear in the
    SELFIES strings from the input iterable, minus the dot ``.`` symbol.

    :param selfies_iter: an iterable of SELFIES strings.
    :return: an alphabet of SELFIES symbols, built from the input iterable.

    :Example:

    >>> import selfies as sf
    >>> selfies_list = ["[C][F][O]", "[C].[O]", "[F][F]"]
    >>> alphabet = sf.get_alphabet_from_selfies(selfies_list)
    >>> sorted(list(alphabet))
    ['[C]', '[F]', '[O]']
    """

    alphabet = set()
    for s in selfies_iter:
        for symbol in split_selfies(s):
            alphabet.add(symbol)
    alphabet.discard(".")
    return alphabet


def len_selfies(selfies: str) -> int:  # okay for OMG linear homopolymers
    """Returns the number of symbols in a given SELFIES string.

    :param selfies: a SELFIES string.
    :return: the symbol length of the SELFIES string.

    :Example:

    >>> import selfies as sf
    >>> sf.len_selfies("[C][=C][F].[C]")
    5
    """

    return selfies.count("[") + selfies.count(".")

def get_selfie_and_smiles_encodings_for_dataset(df, save_vocab_dir, data_augment=False):
    """
    input:
        csv file with molecules. The molecules should contain asterisks to specify the repeating points.
        To be compatible with Group SELFIES, the asterisks are replaced with iodine.

    output:
        1) group SELFIES encoding (list)
        2) group SELFIES tokens (list)
        3) Longest Group SELFIES strings (int)
    """
    # 1) preparing polymerization functional groups
    polymerization_function_group_dict = defaultdict()
    polymerization_function_group_dict['ester'] = Group('ester', '*1OC(*1)=O', priority=1)  # creating a group. Used canon smiles just in case.

    # more step growth polymers
    polymerization_function_group_dict['amide'] = Group('amide', 'O=C(*1)N*1', priority=1)  # amide
    polymerization_function_group_dict['ureathane'] = Group('ureathane', 'O=C(N*1)O*1', priority=1)  # ureathane
    polymerization_function_group_dict['urea'] = Group('urea', 'O=C(N*1)N*1', priority=1)  # urea

    # 2) replace asterisks with iodine
    p_smi_list = np.asanyarray(df['product'].tolist())
    # p_smi_iodine_list = [Chem.CanonSmiles(p_smi.replace("*", "I")) for p_smi in p_smi_list]
    p_smi_iodine_list = [p_smi.replace("*", "I") for p_smi in p_smi_list]  # no canon-smi for the first iodine.

    # 3) extract a set of reasonable groups using fragmentation
    fragment_method = 'none'  # mmpa or default or none
    if not fragment_method == 'none':
        print(f'Extracting group fragments using {fragment_method}', flush=True)
        fragments = fragment_mols(m_set=p_smi_iodine_list, convert=True, method=fragment_method)
        # convert=True -> apply MolFromSmiles
        # method='default' (cleaves based on a ring structure) or 'mmpa' (cleaves based on a matched molecular pair)
        vocab_fragment = dict([(f'frag{idx}', Group(f'frag{idx}', frag)) for idx, frag in enumerate(fragments)])
    else:  # none
        vocab_fragment = dict()

    # 4) add functional groups - commented if there is no group vocab.
    for key, value in polymerization_function_group_dict.items():
        vocab_fragment[key] = value

    # 5) create the group grammar
    grammar_fragment = GroupGrammar(vocab=vocab_fragment)
    vocab_save_path = os.path.join(save_vocab_dir, 'group_selfies_vocab.txt')
    grammar_fragment.to_file(vocab_save_path)  # save vocab

    # 6) encode
    def g_encoder(smi):  # not canon smi for the first iodine.
        encoded = grammar_fragment.full_encoder(Chem.MolFromSmiles(smi))  # starting with [I] for canon SMILES. Not starting with [I] for non-canon SMILES.
        decoded = Chem.MolToSmiles(grammar_fragment.decoder(encoded))  # canon smiles
        assert Chem.CanonSmiles(smi) == decoded, (smi, decoded, len(grammar_fragment.vocab))  # condition, error_message
        return encoded

    print('Encoding... ', end='', flush=True)
    encoded_group_selfies_list = [g_encoder(p_smi_iodine) for p_smi_iodine in p_smi_iodine_list]
    print('Done encoding', flush=True)

    # check
    if not data_augment:  # canon SMILES
        decoded_group_selfies_list = [Chem.MolToSmiles(grammar_fragment.decoder(encoded_group_selfies)) for encoded_group_selfies in encoded_group_selfies_list]
        decoded_group_selfies_asterisk_list = [Chem.CanonSmiles(decoded_smi_iodine.replace("I", "*")) for decoded_smi_iodine in decoded_group_selfies_list]
        compare = np.array(p_smi_list) == np.array(decoded_group_selfies_asterisk_list)
        print('Check decode...', np.all(compare))  # true

    # 7) getting Group SELFIES tokens
    all_group_selfies_symbols = get_alphabet_from_selfies(encoded_group_selfies_list)  # set
    all_group_selfies_symbols.add('[EOS]')  # end of sentence
    all_group_selfies_symbols.add('[PAD]')  # PAD tokens
    all_group_selfies_symbols.add('[UNK]')  # Unknown tokens
    all_group_selfies_symbols.add('[SOS]')  # Start of sentence tokens
    group_selfies_alphabet = list(all_group_selfies_symbols)
    largest_group_selfies_len = max(
        len_selfies(encoded_group_selfies) for encoded_group_selfies in encoded_group_selfies_list
    )

    return encoded_group_selfies_list, group_selfies_alphabet, largest_group_selfies_len


def selfies_to_hot(selfie, largest_selfie_len, alphabet, symbol_to_int):
    """
    Go from a single selfies string to a one-hot encoding.
    Note that this function adds [EOS] token to Group SELFIES.
    """
    selfies_len = len_selfies(selfie)
    largest_selfie_len_with_EOS = largest_selfie_len + 1  # with [EOS] token
    if selfies_len < largest_selfie_len_with_EOS:  # add [EOS] if the length of SELFIES is less than maximum.
        selfie += '[EOS]'
        selfie += '[PAD]' * (largest_selfie_len_with_EOS - 1 - selfies_len)  # pad with [PAD] if the length of SELFIES is less than maximum
    else:
        raise ValueError("This is not possible!")

    # integer encode
    symbol_list = split_selfies(selfie)  # generator
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded, dtype=np.uint8)


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """
    Convert a list of selfies strings to a one-hot encoding.
    * CAUTION: 'alphabet' has a different order for each trial because this involves 'set' (stochastic) to get symbols.
    * CAUTION for the number of tokens. I'm now using np.int16
    """
    # dict
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    hot_list, integer_encoded_list = [], []
    for s in selfies_list:
        # onehot_encoded -> (maximum length, number_of_letters)
        try:
            # one-hot encoding
            integer_encoded, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet, symbol_to_int)
            hot_list.append(onehot_encoded)
            integer_encoded_list.append(integer_encoded)

        except:
            print('Cannot find tokens for this Group SELFIES!', s)
            exit()
    return np.array(hot_list, dtype=np.uint8), np.array(integer_encoded_list, dtype=np.int16)
