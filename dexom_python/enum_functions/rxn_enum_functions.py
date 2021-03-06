import argparse
import pandas as pd
import numpy as np
from dexom_python.imat_functions import imat
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options
from dexom_python.result_functions import read_solution, write_solution
from dexom_python.enum_functions.enumeration import create_enum_variables
from warnings import warn


class RxnEnumSolution(object):
    def __init__(self,
                 all_solutions, unique_solutions, all_binary, unique_binary, all_reactions=None, unique_reactions=None):
        self.all_solutions = all_solutions
        self.unique_solutions = unique_solutions
        self.all_binary = all_binary
        self.unique_binary = unique_binary
        self.all_reactions = all_reactions
        self.unique_reactions = unique_reactions


def rxn_enum(model, reaction_weights, prev_sol, rxn_list=[], eps=1., thr=1e-1, obj_tol=1e-2):
    """
    Reaction enumeration method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: imat Solution object
        an imat solution used as a starting point
    eps: float
        activation threshold in imat
    thr: float
        detection threshold of activated reactions
    tlim: int
        time limit for imat
    tol: float
        tolerance for imat
    obj_tol: float
        variance allowed in the objective_values of the solutions
    Returns
    -------
    solution: RxnEnumSolution object
    """
    tol = model.solver.configuration.tolerances.feasibility
    prev_sol_bin = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
    optimal_objective_value = prev_sol.objective_value - prev_sol.objective_value * obj_tol

    all_solutions = [prev_sol]
    all_solutions_binary = [prev_sol_bin]
    unique_solutions = [prev_sol]
    unique_solutions_binary = [prev_sol_bin]
    all_reactions = []  # for each solution, save which reaction was activated/inactived by the algorithm
    unique_reactions = []

    if not rxn_list:
        rxns = list(model.reactions)
        rxn_list = [r.id for r in rxns]
    for rid in rxn_list:
        if rid not in model.reactions:
            warn('The following reaction ID was not found in the model: %s' % rid)
        else:
            idx = np.where(prev_sol.fluxes.index == rid)[0][0]
            with model as model_temp:
                if rid in model.reactions:
                    rxn = model_temp.reactions.get_by_id(rid)
                    # for active fluxes, check inactivation
                    if prev_sol_bin[idx] == 1:
                        rxn.bounds = (0., 0.)
                    # for inactive fluxes, check activation
                    else:
                        upper_bound_temp = rxn.upper_bound
                        # for inactive reversible fluxes, check activation in backwards direction
                        if rxn.lower_bound < 0.:
                            try:
                                rxn.upper_bound = -thr
                                temp_sol = imat(model_temp, reaction_weights, epsilon=eps, threshold=thr)
                                temp_sol_bin = (np.abs(temp_sol.fluxes) >= thr-tol).values.astype(int)
                                if temp_sol.objective_value >= optimal_objective_value:
                                    all_solutions.append(temp_sol)
                                    all_solutions_binary.append(temp_sol_bin)
                                    if not np.any(np.all(temp_sol_bin == unique_solutions_binary, axis=1)):
                                        unique_solutions.append(temp_sol)
                                        unique_solutions_binary.append(temp_sol_bin)
                                        unique_reactions.append(rid+'_backwards')
                            except:
                                print('An error occurred with reaction %s_reverse. '
                                      'Check feasibility of the model when this reaction is irreversible.' % rid)
                            finally:
                                rxn.upper_bound = upper_bound_temp
                        # for all inactive fluxes, check activation in forwards direction
                        if rxn.upper_bound >= thr:
                            rxn.lower_bound = thr
                        else:
                            rxn.lower_bound = rxn.upper_bound
                    # for all fluxes: compute solution with new bounds
                    try:
                        temp_sol = imat(model_temp, reaction_weights, epsilon=eps, threshold=thr)
                        temp_sol_bin = (np.abs(temp_sol.fluxes) >= thr-tol).values.astype(int)
                        if temp_sol.objective_value >= optimal_objective_value:
                            all_solutions.append(temp_sol)
                            all_solutions_binary.append(temp_sol_bin)
                            all_reactions.append(rid)
                            if not np.any(np.all(temp_sol_bin == unique_solutions_binary, axis=1)):
                                unique_solutions.append(temp_sol)
                                unique_solutions_binary.append(temp_sol_bin)
                                unique_reactions.append(rid)
                    except:
                        if prev_sol_bin[idx] == 1:
                            print('An error occurred with reaction %s. '
                                  'Check feasibility of the model when this reaction is blocked' % rid)
                        else:
                            print('An error occurred with reaction %s. '
                                  'Check feasibility of the model when this reaction is irreversible' % rid)
    solution = RxnEnumSolution(all_solutions, unique_solutions, all_solutions_binary, unique_solutions_binary,
                               all_reactions, unique_reactions)
    return solution


if __name__ == '__main__':
    description = 'Performs the reaction enumeration algorithm on a specified list of reactions'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, matlab, or json format')
    parser.add_argument('-l', '--reaction_list', default=None, help='csv list of reactions to enumerate - if empty, '
                                                                    'will use all reactions in the model')
    parser.add_argument('--range', default='_',
                        help='range of reactions to use from the list, in the format "integer_integer", 0-indexed')
    parser.add_argument('-r', '--reaction_weights', default=None,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-p', '--prev_sol', default=None, help='initial imat solution in .txt format')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-2,
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=1e-5, help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=None, help='Solver time limit')
    parser.add_argument('--tol', type=float, default=1e-6, help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=1e-3, help='Solver MIP gap tolerance')
    parser.add_argument('--obj_tol', type=float, default=1e-3,
                        help='objective value tolerance, as a fraction of the original value')
    parser.add_argument('-o', '--output', default='rxn_enum', help='Path of output files, without format')
    parser.add_argument('--save', action='store_true', help='Use this flag to save each solution individually')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)

    reaction_weights = {}
    if args.reaction_weights is not None:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    rxn_list = []
    if args.reaction_list is not None:
        df = pd.read_csv(args.reaction_list, header=None)
        reactions = [x for x in df.unstack().values]
    else:
        reactions = [r.id for r in model.reactions]

    rxn_range = args.range.split('_')
    if rxn_range[0] == '':
        start = 0
    else:
        start = int(rxn_range[0])
    if rxn_range[1] == '':
        rxn_list = reactions[start:]
    elif int(rxn_range[1]) >= len(reactions):
        rxn_list = reactions[start:]
    else:
        rxn_list = reactions[start:int(rxn_range[1])]

    if args.prev_sol is not None:
        initial_solution, initial_binary = read_solution(args.prev_sol, model)
        model = create_enum_variables(model, reaction_weights, eps=args.epsilon, thr=args.threshold, full=False)
    else:
        initial_solution = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold)

    solution = rxn_enum(model=model, rxn_list=rxn_list, prev_sol=initial_solution, reaction_weights=reaction_weights,
                        eps=args.epsilon, thr=args.threshold, obj_tol=args.obj_tol)

    uniques = pd.DataFrame(solution.unique_binary)
    uniques.to_csv(args.output+'_solutions.csv')

    if args.save:
        for i in range(1, len(solution.unique_solutions)):
            write_solution(model, solution.unique_solutions[i], args.threshold, args.output+'_solution_'+str(i)+'.csv')
