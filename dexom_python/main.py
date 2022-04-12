
import pandas as pd
from warnings import warn
import numpy as np
from cobra.io import load_json_model
from dexom_python.model_functions import load_reaction_weights, save_reaction_weights, read_model, check_model_options
from dexom_python.gpr_rules import apply_gpr
from dexom_python.result_functions import write_solution, read_solution
from dexom_python.pathway_enrichment import Fischer_groups, plot_Fisher_pathways
from dexom_python.imat import imat
from dexom_python.enum_functions.enumeration import write_rxn_enum_script, write_batch_script_divenum
from dexom_python.enum_functions.icut import icut
from dexom_python.enum_functions.rxn_enum import rxn_enum
from dexom_python.enum_functions.diversity_enum import diversity_enum
import matplotlib.pyplot as plt


def margin(tol, eps, thr):
    print("case where y+ = 1, y- = 0")
    rh_pos_rev = -1000 + (1000 + eps) * (1-tol)
    rh_pos_irr = (1-tol)*eps

    print("worst lower bound for reversible reactions:", rh_pos_rev)
    if rh_pos_rev < 0:
        warn("Negative lower bound, the tolerance & epsilon values are too close together", UserWarning)
    print("worst lower bound for irreversible reactions:", rh_pos_irr)

    print("case where y+ = 0, y- = 1")
    rh_neg = 1000 - (1000 + eps) * (1-tol)
    print("worst upper bound for reverse flux:", rh_neg)
    if rh_pos_rev < 0:
        warn("Positive upper bound, the tolerance & epsilon values are too close together", UserWarning)

    print("case where x = 1")
    rl_up = (tol)*1000
    rl_lo_irr = (tol)*-1000
    print("worst upper bound:", rl_up)
    print("worst lower bound for irreversible reactions:", rl_lo_irr)

    if rl_up >= thr or rl_lo_irr <= -thr:
        warn("The detection threshold is below the solver margin of error", UserWarning)

    return 0


if __name__ == '__main__':
    # for testing DEXOM on a toy example
    #
    # model = read_model("toy_models/small4M.json")
    # reaction_weights = load_reaction_weights("toy_models/small4M_weights.csv")
    #
    # eps = 1e-2  # threshold of activity for highly expressed reactions
    # thr = 1e-5  # threshold of activity for all reactions
    # obj_tol = 1e-3  # variance allowed for the objective_value
    # tlim = 600  # time limit (in seconds) for the imat model.optimisation() call
    # tol = 1e-8  # feasibility tolerance for the solver
    # mipgap = 1e-3  # mip gap tolerance for the solver
    # maxiter = 10  # maximum number of iterations
    # dist_anneal = 0.9  # diversity-enumeration parameter
    #
    # imat_solution = imat(model=model, reaction_weights=reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim,
    #                      feasibility=tol, mipgaptol=mipgap)
    # write_solution(model, solution=imat_solution, threshold=thr, filename="toy_models/small4M_imatsol.csv")
    #
    # rxn_sol = rxn_enum(model=model, rxn_list=[], prev_sol=imat_solution, reaction_weights=reaction_weights, eps=eps,
    #          thr=thr, tlim=tlim, feas=tol, mipgap=mipgap, obj_tol=obj_tol)
    # pd.DataFrame(rxn_sol.unique_binary).to_csv("toy_models/small4M_rxnenum_solutions.csv")
    #
    # div_sol = diversity_enum(model=model, prev_sol=imat_solution, reaction_weights=reaction_weights, eps=eps, thr=thr,
    #                          obj_tol=obj_tol, maxiter=maxiter, out_path="toy_models/small4M_divenum")

    eps = 1e-3  # threshold of activity for highly expressed reactions
    obj_tol = 1e-3  # variance allowed for the objective_value
    tlim = 1200  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-8  # feasibility tolerance for the solver
    mipgap = 1e-3  # mip gap tolerance for the solver
    maxiter = 100  # maximum number of iterations
    thr = 1e-5  # threshold of activity for all reactions
    verb = 1

    # margin(tol, eps, thr)

    # reaction_weights = load_reaction_weights("zebrafish_weights/zebrafish_weights_DMSO2.csv")
    # model = read_model("zebrafish/Zebrafish-GEM.json")
    # check_model_options(model, timelimit=tlim, feasibility=tol, mipgaptol=mipgap, verbosity=verb)
    #
    #
    # imatsol = "enum_TPPhigh/TPPhigh_imatsol.csv"
    # directory = "enum_TCSlow/"
    # modelfile = "zebrafish/Zebrafish-GEM.json"
    # weightfile = "zebrafish_weights/zebrafish_weights_TCShigh.csv"
    # # reactionlist = "zebrafish/Zebrafish1_reactions_shuffled.csv"
    # username = "mstingl"
    # rxnsols = "rxn_enum"
    # objtol = obj_tol

    # newmodel = read_model("tests/model/example_r13m10.json")
    # reaction_weights = load_reaction_weights("tests/model/example_r13m10_weights.csv")
    # newsol = imat(newmodel, reaction_weights, 1, 1e-5)
    # testsols = icut(newmodel, newsol, reaction_weights, eps=1, thr=1e-5, maxiter=20)

    # from dexom_python.dexom_cluster_results import analyze_dexom_cluster_results
    # a = analyze_dexom_cluster_results(directory, directory, 3, 50)
    # from dexom_python.result_functions import plot_pca
    # plot_pca(directory+"all_divenum_sols.csv", directory+"all_rxnenum_sols.csv", directory+"all_pca")

    #
    # write_rxn_enum_script(directory, modelfile, weightfile, reactionlist, imatsol, username, eps=eps, thr=thr,
    #                       tol=tol, iters=50, maxiters=500)
    #
    #
    # write_batch_script_divenum(directory, username, modelfile, weightfile, rxnsols, objtol, eps=1e-3, thr=1e-5,
    #                            tol=1e-8, filenums=50, iters=10, t=1500)

    sols = []
    names = []
    molecules = ["TBT", "TCS", "PFOA", "DDE", "TPP", "BPA"]
    doses = ["low", "high"]
    extra = ["DMSO1", "DMSO2", "partial_consensus"]

    for m in molecules:
        for d in doses:
            a = pd.read_csv('enum_' + m + d + '/all_dexom_sols.csv', index_col=0)
            sols.append(a)
            names.append(m + d)
    for e in extra:
        a = pd.read_csv('enum_' + e + '/all_dexom_sols.csv', index_col=0)
        sols.append(a)
        names.append(e)

    X = pd.concat(sols, ignore_index=True)
    unique = X.drop_duplicates(ignore_index=True)

    model = read_model("zebrafish/Zebrafish-GEM.json")
    unique.columns = [r.id for r in model.reactions]
    frequency = unique.sum()
    reactions = list(frequency[frequency==len(unique)].index)
