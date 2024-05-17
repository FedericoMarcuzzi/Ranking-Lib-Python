import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rename_dict_key(input_dict, key, key_alias):
    clean_dict = {key : value for key, value in input_dict.items()}
    key_to_remove = key_alias[np.argmax([k in clean_dict for k in clean_dict])]
    if key_to_remove in clean_dict:
        clean_dict[key] = clean_dict.pop(key_to_remove)
    return clean_dict