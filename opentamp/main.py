import argparse, traceback
from opentamp.errors_exceptions import OpenTAMPException
from opentamp.pma import pr_graph
from opentamp.pma.hl_solver import HLSolver
from opentamp.pma.ll_solver_OSQP import LLSolverOSQP

"""
Entry-level script. Calls pr_graph.p_mod_abs() to plan, then runs the plans in
simulation using the chosen viewer.
"""

cache = {}
def parse_file_to_dict(f_name):
    d = {}
    if f_name in cache:
        return cache[f_name].copy()
    with open(f_name, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if len(line.split(":", 1)) != 2: import ipdb; ipdb.set_trace()
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
        f.close()
    cache[f_name] = d
    return d.copy()

def main(domain_file, problem_file):
    try:
        domain_config = parse_file_to_dict(domain_file)
        problem_config = parse_file_to_dict(problem_file)
        # solvers_config = parse_file_to_dict(solvers_file)
        hl_solver = HLSolver(domain_config)
        ll_solver = LLSolverOSQP()
        plan, msg = pr_graph.p_mod_abs(hl_solver, ll_solver, domain_config, problem_config)
        if plan:
            print("Executing plan!")
            plan.execute()
        else:
            print(msg)
    except OpenTAMPException as e:
        print("Caught an exception in OpenTAMP:")
        traceback.print_exc()
        print("Terminating...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenTAMP.")
    parser.add_argument("domain_file",
                        help="Path to the domain file to use. All domain settings should be specified in this file.")
    parser.add_argument("problem_file",
                        help="Path to the problem file to use. All problem settings should be specified in this file. Spawned by a generate_*_prob.py script.")
    #parser.add_argument("solvers_file",
    #                    help="Path to the file naming the solvers to use. The HLSolver and LLSolver to use should be specified here.")
    args = parser.parse_args()
    main(args.domain_file, args.problem_file) #, args.solvers_file)
