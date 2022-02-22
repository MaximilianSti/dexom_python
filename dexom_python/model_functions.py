
import six
import pandas as pd
import numpy as np
from pathlib import Path
from cobra.io import load_json_model, read_sbml_model, load_matlab_model


def read_model(modelfile):
    fileformat = Path(modelfile).suffix
    model = None
    if fileformat == ".sbml" or fileformat == ".xml":
        model = read_sbml_model(modelfile)
    elif fileformat == '.json':
        model = load_json_model(modelfile)
    elif fileformat == ".mat":
        model = load_matlab_model(modelfile)
    elif fileformat == "":
        print("Wrong model path")
    else:
        print("Only SBML, JSON, and Matlab formats are supported for the models")

    try:
        model.solver = 'cplex'
    except:
        print("cplex is not available or not properly installed")

    return model


def clean_model(model, reaction_weights=None, full=False):
    """
    removes variables and constraints added to the model.solver during imat

    Parameters
    ----------
    model: cobra.Model
        a model that has previously been passed to imat
    reaction_weights: dict
        the same reaction weights used for the imat
    full: bool
        the same bool used for the imat calculation
    """
    if full:
        for rxn in model.reactions:
            rid = rxn.id
            if "x_"+rid in model.solver.variables:
                model.solver.remove(model.solver.variables["x_"+rid])
                model.solver.remove(model.solver.variables["xf_"+rid])
                model.solver.remove(model.solver.variables["xr_"+rid])
                model.solver.remove(model.solver.constraints["xr_"+rid+"_upper"])
                model.solver.remove(model.solver.constraints["xr_"+rid+"_lower"])
                model.solver.remove(model.solver.constraints["xf_"+rid+"_upper"])
                model.solver.remove(model.solver.constraints["xf_"+rid+"_lower"])
    else:
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0. and "rh_"+rid+"_pos" in model.solver.variables:
                model.solver.remove(model.solver.variables["rh_"+rid+"_pos"])
                model.solver.remove(model.solver.variables["rh_"+rid+"_neg"])
                model.solver.remove(model.solver.constraints["rh_"+rid+"_pos_bound"])
                model.solver.remove(model.solver.constraints["rh_"+rid+"_neg_bound"])
            elif weight < 0. and "rl_"+rid in model.solver.variables:
                model.solver.remove(model.solver.variables["rl_"+rid])
                model.solver.remove(model.solver.constraints["rl_"+rid+"_upper"])
                model.solver.remove(model.solver.constraints["rl_"+rid+"_lower"])


def check_model_options(model, timelimit=None, feasibility=None, mipgaptol=None):

    if timelimit:
        model.solver.configuration.timeout = timelimit
    if feasibility:
        model.tolerance = feasibility
    if mipgaptol:
        model.solver.problem.parameters.mip.tolerances.mipgap.set(mipgaptol)
    model.solver.configuration.presolve = True
    return model


def get_all_reactions_from_model(model, save=True, shuffle=False, out_path=""):
    """

    Parameters
    ----------
    model: a cobrapy model
    save: bool
        by default, exports the reactions in a csv format
    shuffle: bool
        set to True to shuffle the order of the reactions
    out_path: str
        output path
    Returns
    -------
    A list of all reactions in the model
    """
    rxn_list = [r.id for r in model.reactions]
    if save:
        pd.Series(rxn_list).to_csv(out_path + model.id + "_reactions.csv", header=False, index=False)
    if shuffle:
        np.random.shuffle(rxn_list)
        pd.Series(rxn_list).to_csv(out_path + model.id + "_reactions_shuffled.csv", header=False, index=False)
    return rxn_list


def get_subsystems_from_model(model, save=True, out_path=""):
    """
    Creates a list of all subsystems of a model and their associated reactions
    Parameters
    ----------
    model: a cobrapy model
    save: bool

    Returns
    -------
    rxn_sub: a DataFrame with reaction names as index and subsystem name as column
    sub_list: a list of subsystems
    """

    rxn_sub = {}
    sub_list = []
    i = 0
    for rxn in model.reactions:
        rxn_sub[i] = (rxn.id, rxn.subsystem)
        i += 1
        if rxn.subsystem not in sub_list:
            sub_list.append(rxn.subsystem)
    if sub_list[-1] == "":
        sub_list.pop()
    rxn_sub = pd.DataFrame.from_dict(rxn_sub, orient="index", columns=["ID", "subsystem"])
    if save:
        rxn_sub.to_csv(out_path+model.id+"_reactions_subsystems.csv")
        with open(out_path+model.id+"_subsystems_list.txt", "w+") as file:
            file.write(";".join(sub_list))
    return rxn_sub, sub_list


def save_reaction_weights(reaction_weights, filename):
    """
    Parameters
    ----------
    reaction_weights: dict
        a dictionary where keys = reaction IDs and values = weights
    filename: str
    Returns
    -------
    the reaction_weights dict as a pandas DataFrame
    """
    df = pd.DataFrame(reaction_weights.items(), columns=["reactions", "weights"])
    df.to_csv(filename)
    df.index = df["reactions"]
    return df["weights"]


def load_reaction_weights(filename, rxn_names="reactions", weight_names="weights"):
    """
    loads reaction weights from a .csv file
    Parameters
    ----------
    filename: str
        the path + name of a .csv file containing reaction weights
    rxn_names: str
        the name of the column containing the reaction names
    weight_names: str
        the name of the column containing the weights

    Returns
    -------
    a dict of reaction weights
    """
    df = pd.read_csv(filename)
    df.index = df[rxn_names]
    reaction_weights = df[weight_names].to_dict()
    return {k: float(v) for k, v in reaction_weights.items() if float(v) == float(v)}

