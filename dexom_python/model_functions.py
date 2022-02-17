
import six
import pandas as pd
import numpy as np
import re
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from pathlib import Path
import argparse
from symengine import sympify, Add, Mul, Max, Min


def replace_MulMax_AddMin(expression):
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (replace_MulMax_AddMin(arg) for arg in expression.args)
        if expression.__class__ == Mul:
            return Max(*replaced_args)
        elif expression.__class__ == Add:
            return Min(*replaced_args)
        else:
            return expression.func(*replaced_args)


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


def recon2_gpr(model, gene_weights, save=True, filename="recon2_weights"):
    """
    Applies the GPR rules from the recon2 or recon2.2 model for creating reaction weights

    Parameters
    ----------
    model: a cobrapy model
    gene_weights: a dictionary containing gene IDs & weights
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction_weights: dict where keys = reaction IDs and values = weights
    """
    reaction_weights = {}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
            expr_split = [s.replace(':', '_') if ':' in s else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set([s.replace(':', '_') for s in rxngenes if ':' in s])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0:
                    new_weights[g] = -v - 1e-15
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace('or', '*').replace('and', '+')
            # weight = sympify(expression).xreplace({Mul: Max}).xreplace({Add: Min})
            weight = replace_MulMax_AddMin(sympify(expression))
            reaction_weights[rxn.id] = weight.subs(new_weights)
            if weight + 1e-15 in negweights:
                weight = -weight - 1e-15
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0
    if save:
        save_reaction_weights(reaction_weights, filename+".csv")
    return reaction_weights


def recon1_gpr(model, gene_weights, save=True, filename="recon1_weights"):
    """
    Applies the GPR rules from the recon1 model
    Parameters
    ----------
    model: a cobrapy model
    gene_file: the path to a csv file containing gene scores
    gene_weights: a dictionary containing gene IDs & weights
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction weights: dict
    """
    reaction_weights = {}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
            expr_split = ["g_"+s[:-4] if '_' in s else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set(["g_"+s[:-4] for s in rxngenes if '_' in s])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0:
                    new_weights[g] = -v - 1e-15
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace('or', '*').replace('and', '+')
            # weight = sympify(expression).xreplace({Mul: Max}).xreplace({Add: Min})
            weight = replace_MulMax_AddMin(sympify(expression))
            reaction_weights[rxn.id] = weight.subs(new_weights)
            if weight + 1e-15 in negweights:
                weight = -weight - 1e-15
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0
    if save:
        save_reaction_weights(reaction_weights, filename+".csv")
    return reaction_weights


def iMM1865_gpr(model, gene_weights, save=True, filename="iMM1865_weights"):
    """
    Applies the GPR rules from the iMM1865 model
    Parameters
    ----------
    model: a cobrapy model
    gene_file: the path to a csv file containing gene scores
    gene_weights: a dictionary containing gene IDs & weights
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction weights: dict
    """
    reaction_weights = {}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.split()
            expr_split = ["g_"+s if s.isdigit() else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set(["g_"+s for s in rxngenes if s.isdigit()])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0:
                    new_weights[g] = -v - 1e-15
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace('or', '*').replace('and', '+')
            # weight = sympify(expression).xreplace({Mul: Max}).xreplace({Add: Min})
            weight = replace_MulMax_AddMin(sympify(expression))
            reaction_weights[rxn.id] = weight.subs(new_weights)
            if weight + 1e-15 in negweights:
                weight = -weight - 1e-15
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0
    if save:
        save_reaction_weights(reaction_weights, filename+".csv")
    return reaction_weights


def human1_gpr(model, gene_weights, save=True, filename="human1_weights"):
    """
    Applies the GPR rules from the human-GEM model for creating reaction weights

    Parameters
    ----------
    model: a cobrapy model
    gene_file: the path to a csv file containing gene scores
    gene_weights: a dictionary containing gene IDs & weights
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction_weights: dict where keys = reaction IDs and values = weights
    """
    reaction_weights = {}
    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
            gen_list = set(rxn.genes)  #set([s for s in rxngenes if 'ENSG' in s])
            new_weights = {g.id: gene_weights.get(g.id, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0:
                    new_weights[g] = -v - 1e-15
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace('or', '*').replace('and', '+')
            # weight = sympify(expression).xreplace({Mul: Max}).xreplace({Add: Min})
            weight = replace_MulMax_AddMin(sympify(expression)).subs(new_weights)
            if weight + 1e-15 in negweights:
                weight = -weight - 1e-15
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0.
    if save:
        save_reaction_weights(reaction_weights, filename+".csv")
    return reaction_weights


if __name__ == "__main__":

    description = "Applies GPR rules to transform gene weights into reaction weights"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="GEM in json, sbml or matlab format")
    parser.add_argument("-n", "--modelname", help="supported: human1, recon1, recon2, iMM1865")
    parser.add_argument("-g", "--gene_file", help="csv file containing gene identifiers and scores")
    parser.add_argument("-o", "--output", default="reaction_weights",
                        help="Path to which the reaction_weights .csv file is saved")
    parser.add_argument("--gene_ID", default="ID", help="column containing the gene identifiers")
    parser.add_argument("--gene_score", default="t", help="column containing the gene scores")
    args = parser.parse_args()

    model = read_model(args.model)
    model_gpr = {'human1': human1_gpr, 'recon1': recon1_gpr, 'recon2': recon2_gpr, 'iMM1865': iMM1865_gpr}

    genes = pd.read_csv(args.gene_file)
    gene_weights = pd.Series(genes[args.gene_score], index=genes[args.gene_ID])
    # gene_weights = {idx: np.max(gene_weights.loc[idx][args.gene_score]) for idx in gene_weights.index}
    # current behavior: all genes with several different weights are removed
    for x in set(gene_weights.index):
        if type(gene_weights[x]) != np.float64:
            if len(gene_weights[x].value_counts()) > 1:
                gene_weights.pop(x)
    gene_weights = gene_weights.to_dict()

    if args.modelname not in model_list.keys():
        print("Unsupported model. The currently supported models are: human1, recon1, recon2, iMM1865")
    else:
        reaction_weights = model_gpr[args.modelname](model=model, gene_weights=gene_weights, filename=args.output,
                                                      save=True)
