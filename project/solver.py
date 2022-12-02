import time
import tqdm
from typing import List

import numpy as np
import osqp
import scipy

from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver as SolverBase
from sco_py.expr import CompExpr, EqExpr
import sco_py.sco_osqp.osqp_utils as osqp_utils

from opentamp.pma.backtrack_ll_solver_OSQP import MAX_PRIORITY
from opentamp.pma.robot_solver import RobotSolverOSQP as RobotSolverOSQPBase


# @profile
def osqp_optimize(
    osqp_vars: List[osqp_utils.OSQPVar],
    _sco_vars: List[osqp_utils.Variable],
    osqp_quad_objs: List[osqp_utils.OSQPQuadraticObj],
    osqp_lin_objs: List[osqp_utils.OSQPLinearObj],
    osqp_lin_cnt_exprs: List[osqp_utils.OSQPLinearConstraint],
    eps_abs: float = osqp_utils.DEFAULT_EPS_ABS,
    eps_rel: float = osqp_utils.DEFAULT_EPS_REL,
    max_iter: int = osqp_utils.DEFAULT_MAX_ITER,
    rho: float = osqp_utils.DEFAULT_RHO,
    adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
    sigma: float = osqp_utils.DEFAULT_SIGMA,
    verbose: bool = False,
):
    """
    Calls the OSQP optimizer on the current QP approximation with a given
    penalty coefficient.
    """

    # First, we need to setup the problem as described here: https://osqp.org/docs/solver/index.html
    # Specifically, we need to start by constructing the x vector that contains all the
    # OSQPVars that are part of the QP. This will take the form of a mapping from OSQPVar to
    # index within the x vector.
    var_to_index_dict = {}
    osqp_var_list = list(osqp_vars)
    # Make sure to sort this list to get a canonical ordering of variables to make
    # matrix construction easier
    osqp_var_list.sort()
    for idx, osqp_var in enumerate(osqp_var_list):
        var_to_index_dict[osqp_var] = idx
    num_osqp_vars = len(osqp_vars)

    # Construct the q-vector by looping through all the linear objectives
    q_vec = np.zeros(num_osqp_vars)
    for lin_obj in osqp_lin_objs:
        q_vec[var_to_index_dict[lin_obj.osqp_var]] += lin_obj.coeff

    # Next, construct the P-matrix by looping through all quadratic objectives

    # Since P must be upper-triangular, the shape must be (num_osqp_vars, num_osqp_vars)
    P_mat = np.zeros((num_osqp_vars, num_osqp_vars))
    for quad_obj in osqp_quad_objs:
        for i in range(quad_obj.coeffs.shape[0]):
            idx2 = var_to_index_dict[quad_obj.osqp_vars1[i]]
            idx1 = var_to_index_dict[quad_obj.osqp_vars2[i]]
            if idx1 > idx2:
                P_mat[idx2, idx1] += 0.5 * quad_obj.coeffs[i]
            elif idx1 < idx2:
                P_mat[idx1, idx2] += 0.5 * quad_obj.coeffs[i]
            else:
                P_mat[idx1, idx2] += quad_obj.coeffs[i]

    # Next, setup the A-matrix and l and u vectors
    A_mat = np.zeros((num_osqp_vars + len(osqp_lin_cnt_exprs), num_osqp_vars))
    l_vec = np.zeros(num_osqp_vars + len(osqp_lin_cnt_exprs))
    u_vec = np.zeros(num_osqp_vars + len(osqp_lin_cnt_exprs))

    # First add all the linear constraints
    # However, note that this isn't entirely straightforward: some
    row_num = 0
    for lin_constraint in osqp_lin_cnt_exprs:
        l_vec[row_num] = lin_constraint.lb
        u_vec[row_num] = lin_constraint.ub
        for i in range(lin_constraint.coeffs.shape[0]):
            # if var_to_index_dict[lin_constraint.osqp_vars[i]] == 193:
            #     import ipdb; ipdb.set_trace()
            A_mat[
                row_num, var_to_index_dict[lin_constraint.osqp_vars[i]]
            ] = lin_constraint.coeffs[i]
        row_num += 1

    # Then, add the trust regions for every variable as constraints
    for osqp_var in osqp_vars:
        A_mat[row_num, var_to_index_dict[osqp_var]] = 1.0
        l_vec[row_num] = osqp_var.get_lower_bound()
        u_vec[row_num] = osqp_var.get_upper_bound()
        row_num += 1

    # Finally, construct the matrices and call the OSQP Solver!
    P_mat_sparse = scipy.sparse.csc_matrix(P_mat)
    A_mat_sparse = scipy.sparse.csc_matrix(A_mat)

    m = osqp.OSQP()

    m.setup(
        P=P_mat_sparse,
        q=q_vec,
        A=A_mat_sparse,
        rho=rho,
        sigma=sigma,
        l=l_vec,
        u=u_vec,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        delta=1e-07,
        # polish=True,
        polish=False,
        adaptive_rho=adaptive_rho,
        warm_start=True,
        verbose=False,
        max_iter=max_iter,
    )

    solve_res = m.solve()

    if solve_res.info.status_val == -2 and verbose:
        print(
            "ERROR! OSQP Solver hit max iteration limit. Either reduce your tolerances or increase the max iterations!"
        )

    return (solve_res, var_to_index_dict)


def penalty_sqp_optimize(
    prob,
    add_convexified_terms=False,
    osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
    osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
    osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,
    rho: float = osqp_utils.DEFAULT_RHO,
    adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
    sigma: float = osqp_utils.DEFAULT_SIGMA,
    verbose=False,
):
    """
    Calls the OSQP optimizer on the current QP approximation with a given
    penalty coefficient. Note that add_convexified_terms is a convenience
    boolean useful to toggle whether or not self._osqp_penalty_exprs and
    self._osqp_penalty_cnts are included in the optimization problem
    """

    lin_objs = prob._osqp_lin_objs
    cnt_exprs = prob._osqp_lin_cnt_exprs[:]

    if add_convexified_terms:
        lin_objs += prob._osqp_penalty_exprs
        for penalty_cnt_list in prob._osqp_penalty_cnts:
            cnt_exprs.extend(penalty_cnt_list)

    solve_res, var_to_index_dict = osqp_optimize(
        prob._osqp_vars,
        prob._vars,
        prob._osqp_quad_objs,
        lin_objs,
        cnt_exprs,
        osqp_eps_abs,
        osqp_eps_rel,
        osqp_max_iter,
        rho=rho,
        adaptive_rho=adaptive_rho,
        sigma=sigma,
        verbose=verbose,
    )

    # If the solve failed, just return False
    if solve_res.info.status_val not in [1, 2]:
        return False

    # If the solve succeeded, update all the variables with these new values, then
    # run he callback before returning true
    osqp_utils.update_osqp_vars(var_to_index_dict, solve_res.x)
    prob._update_vars()
    prob._callback()  # TODO: Modify to get the visualizer in a better spot.
    return True


class Solver(SolverBase):
    def solve(
        self,
        prob,
        method=None,
        tol=None,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ):
        """
        Returns whether solve succeeded.

        Given a sco (sequential convex optimization) problem instance, solve
        using specified method to find a solution. If the specified method
        doesn't exist, an exception is thrown.
        """
        if tol is not None:
            self.min_trust_region_size = tol
            self.min_approx_improve = tol
            self.cnt_tolerance = tol

        if method == "penalty_sqp":
            return self._penalty_sqp(
                prob,
                verbose=verbose,
                osqp_eps_abs=osqp_eps_abs,
                osqp_eps_rel=osqp_eps_rel,
                osqp_max_iter=osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                sigma=sigma,
            )
        elif method == "metropolis_hastings":
            return self._metropolis_hastings(
                prob,
                verbose=verbose,
                osqp_eps_abs=osqp_eps_abs,
                osqp_eps_rel=osqp_eps_rel,
                osqp_max_iter=osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                sigma=sigma,
            )
        elif method == "gradient_descent":
            return self._gradient_descent(
                prob,
                verbose=verbose,
                osqp_eps_abs=osqp_eps_abs,
                osqp_eps_rel=osqp_eps_rel,
                osqp_max_iter=osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                sigma=sigma,
            )

        elif method == "mala":
            return self._mala(
                prob,
                verbose=verbose,
                osqp_eps_abs=osqp_eps_abs,
                osqp_eps_rel=osqp_eps_rel,
                osqp_max_iter=osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                sigma=sigma,
            )
        else:
            raise Exception("This method is not supported.")

    def _set_prob_vars(self, prob, x, var_to_index_dict):
        osqp_utils.update_osqp_vars(var_to_index_dict, x)
        prob._update_vars()

    # @profile
    def _metropolis_hastings(
        self,
        prob,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,  # 100_000
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ):
        """
        Run Metropolis-Hastings.
        Returns true if a feasible solution is returned.
        """

        var_to_index_dict = {osqp_var: i for i, osqp_var in enumerate(prob._osqp_vars)}
        burn_in = 5_000

        def logpdf(x, mu, sigma=1):
            return (
                -0.5 * np.log(2 * np.pi) * x.size
                - np.log(sigma)
                - np.sum((x - mu) ** 2) / (2 * sigma**2)
            )

        # penalty_coeff = self.initial_penalty_coeff
        penalty_coeff = 10
        for i in range(self.max_merit_coeff_increases):

            x = np.zeros(len(prob._osqp_vars))
            self._set_prob_vars(prob, x, var_to_index_dict)
            val = -prob.get_value(penalty_coeff, vectorize=False)
            # grad = self._grad(prob, penalty_coeff)
            for j in tqdm.trange(osqp_max_iter):
                # Sample from proposal distribution and get objective value
                x_proposed = np.random.normal(loc=x)
                self._set_prob_vars(prob, x_proposed, var_to_index_dict)
                val_proposed = -prob.get_value(penalty_coeff, vectorize=False)

                # define acceptance probability
                log_p_proposed = val_proposed
                log_p_sampled = val
                log_q_proposed = logpdf(x_proposed, mu=x)
                log_q_sampled = logpdf(x, mu=x_proposed)
                log_A = log_p_proposed + log_q_sampled - log_p_sampled - log_q_proposed

                # Accept or reject and do appropriate setting of variables
                if np.random.uniform() < min(1, np.exp(log_A)):
                    x = x_proposed
                else:
                    self._set_prob_vars(prob, x, var_to_index_dict)

                # If the solve succeeded
                if i > burn_in and prob.get_max_cnt_violation() < self.cnt_tolerance:
                    prob._callback()  # TODO: Modify to get the visualizer in a better spot.
                    return True

            penalty_coeff *= self.merit_coeff_increase_ratio

        return False

    def _grad(self, prob, penalty_coeff, var_to_index_dict):
        gradient = np.zeros(len(prob._osqp_vars))

        # Differentiate cost function
        for bound_expr in prob._quad_obj_exprs + prob._nonquad_obj_exprs:
            x = bound_expr.var.get_value()
            expr_grad = bound_expr.expr.grad(x)

            for var, g in zip(bound_expr.var._osqp_vars.squeeze(), expr_grad):
                gradient[var_to_index_dict[var]] += g

        for bound_expr in prob._nonlin_cnt_exprs:
            x = bound_expr.var.get_value()
            # No gradient if constraint is satisfied
            if bound_expr.expr.eval(x):
                continue

            expr_y = bound_expr.expr.expr.eval(x)
            expr_grad = bound_expr.expr.expr.grad(x)

            if isinstance(bound_expr.expr, EqExpr):
                expr_grad = np.sign(expr_y - bound_expr.expr.val) * expr_grad
            else:
                raise NotImplementedError()

            expr_grad = penalty_coeff * expr_grad.sum(axis=0)
            for var, g in zip(bound_expr.var._osqp_vars.squeeze(1), expr_grad):
                gradient[var_to_index_dict[var]] += g

        # if prob.get_max_cnt_violation() > self.cnt_tolerance:
        #    breakpoint()
        #    cnt_vio = self._compute_cnt_violation(bound_expr)
        #    value += penalty_coeff * np.sum(cnt_vio)
        #    var_grads[bound_expr] += penalty_coeff * bound_expr.expr.grad(bound_expr.var.get_value())

        return gradient

    def _gradient_descent(
        self,
        prob,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,  # 100_000
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ):
        var_to_index_dict = {osqp_var: i for i, osqp_var in enumerate(prob._osqp_vars)}
        lr = 1e-2

        penalty_coeff = 10

        x = np.zeros(len(prob._osqp_vars))
        self._set_prob_vars(prob, x, var_to_index_dict)
        val = -prob.get_value(penalty_coeff, vectorize=False)
        grad = self._grad(prob, penalty_coeff, var_to_index_dict)

        for i in tqdm.trange(osqp_max_iter):
            if i % 50 == 0:
                print(prob.get_max_cnt_violation(), self.cnt_tolerance, val)

            x = x - lr * grad
            self._set_prob_vars(prob, x, var_to_index_dict)
            val = -prob.get_value(penalty_coeff, vectorize=False)
            grad = self._grad(prob, penalty_coeff, var_to_index_dict)

            # If the solve succeeded
            if prob.get_max_cnt_violation() < self.cnt_tolerance:
                prob._callback()  # TODO: Modify to get the visualizer in a better spot.
                return True

    def _mala(
        self,
        prob,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,  # 100_000
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ):
        var_to_index_dict = {osqp_var: i for i, osqp_var in enumerate(prob._osqp_vars)}
        burn_in = 500
        tau = 1e-2

        penalty_coeff = 10
        accepted = 0

        x = np.zeros(len(prob._osqp_vars))
        self._set_prob_vars(prob, x, var_to_index_dict)
        val = -prob.get_value(penalty_coeff, vectorize=False)
        grad = self._grad(prob, penalty_coeff, var_to_index_dict)

        for i in tqdm.trange(osqp_max_iter):
            if i % 50 == 0:
                print(prob.get_max_cnt_violation(), self.cnt_tolerance, accepted)

            mean = x - tau * grad
            std = np.sqrt(2 * tau)
            z = mean + std * np.random.randn(*x.shape)

            self._set_prob_vars(prob, z, var_to_index_dict)
            new_val = -prob.get_value(penalty_coeff, vectorize=False)
            new_grad = self._grad(prob, penalty_coeff, var_to_index_dict)

            # define acceptance probability
            num = new_val - np.sum((x - z + tau * new_grad) ** 2) / (4 * tau)
            den = val - np.sum((z - x + tau * grad) ** 2) / (4 * tau)
            log_A = num - den

            # accept or reject
            if np.random.uniform() <= min(1, np.exp(log_A)):
                x = z
                val = new_val
                grad = new_grad
                accepted += 1
            else:
                self._set_prob_vars(prob, x, var_to_index_dict)

            # If the solve succeeded
            if i > burn_in and prob.get_max_cnt_violation() < self.cnt_tolerance:
                prob._callback()  # TODO: Modify to get the visualizer in a better spot.
                return True

    # @profile
    def _penalty_sqp(
        self,
        prob,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ):
        """
        Return true is the penalty sqp method succeeds.
        Uses Penalty Sequential Quadratic Programming to solve the problem
        instance.
        """
        start = time.time()
        trust_region_size = self.initial_trust_region_size
        penalty_coeff = self.initial_penalty_coeff

        if not prob.find_closest_feasible_point():
            return False

        for i in range(self.max_merit_coeff_increases):
            success = self._min_merit_fn(
                prob,
                penalty_coeff,
                trust_region_size,
                verbose=verbose,
                osqp_eps_abs=osqp_eps_abs,
                osqp_eps_rel=osqp_eps_rel,
                osqp_max_iter=osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                sigma=sigma,
            )
            if verbose:
                print("\n")

            if prob.get_max_cnt_violation() > self.cnt_tolerance:
                penalty_coeff = penalty_coeff * self.merit_coeff_increase_ratio
                trust_region_size = self.initial_trust_region_size
            else:
                end = time.time()
                if verbose:
                    print("sqp time: ", end - start)
                return success
        end = time.time()
        if verbose:
            print("sqp time: ", end - start)
        return False

    # @profile
    def _min_merit_fn(
        self,
        prob,
        penalty_coeff,
        trust_region_size,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ):
        """
        Minimize merit function for penalty sqp.
        Returns true if the merit function is minimized successfully.
        """
        sqp_iter = 1

        while True:
            if verbose:
                print(("  sqp_iter: {0}".format(sqp_iter)))

            prob.convexify()
            prob.update_obj(penalty_coeff)
            merit = prob.get_value(penalty_coeff)
            merit_vec = prob.get_value(penalty_coeff, True)
            prob.save()

            while True:
                if verbose:
                    print(("    trust region size: {0}".format(trust_region_size)))
                prob.add_trust_region(trust_region_size)
                _ = penalty_sqp_optimize(
                    prob,
                    osqp_eps_abs=osqp_eps_abs,
                    osqp_eps_rel=osqp_eps_rel,
                    osqp_max_iter=osqp_max_iter,
                    rho=rho,
                    adaptive_rho=adaptive_rho,
                    sigma=sigma,
                    verbose=verbose,
                )
                model_merit = prob.get_approx_value(penalty_coeff)
                model_merit_vec = prob.get_approx_value(penalty_coeff, True)
                new_merit = prob.get_value(penalty_coeff)

                approx_merit_improve = merit - model_merit
                if not approx_merit_improve:
                    approx_merit_improve += 1e-12

                # we converge if one of the violated constraint groups
                # is below the minimum improvement
                approx_improve_vec = merit_vec - model_merit_vec
                violated = merit_vec > self.cnt_tolerance
                if approx_improve_vec.shape == (0,):
                    approx_improve_vec = np.array([approx_merit_improve])
                    violated = approx_improve_vec > -np.inf

                exact_merit_improve = merit - new_merit

                merit_improve_ratio = exact_merit_improve / approx_merit_improve

                if verbose:
                    print(
                        (
                            "      merit: {0}. model_merit: {1}. new_merit: {2}".format(
                                merit, model_merit, new_merit
                            )
                        )
                    )
                    print(
                        (
                            "      approx_merit_improve: {0}. exact_merit_improve: {1}. merit_improve_ratio: {2}".format(
                                approx_merit_improve,
                                exact_merit_improve,
                                merit_improve_ratio,
                            )
                        )
                    )

                if self._bad_model(approx_merit_improve):
                    if verbose:
                        print(
                            (
                                "Approximate merit function got worse ({0})".format(
                                    approx_merit_improve
                                )
                            )
                        )
                        print(
                            "Either convexification is wrong to zeroth order, or you're in numerical trouble."
                        )
                    prob.restore()
                    return False

                if self._y_converged(approx_merit_improve):
                    if verbose:
                        print("Converged: y tolerance")
                    prob.restore()
                    return True

                # we converge if one of the violated constraint groups
                # is below the minimum improvement and none of its overlapping
                # groups are making progress
                prob.nonconverged_groups = []
                for gid, idx in prob.gid2ind.items():
                    if (
                        violated[idx]
                        and approx_improve_vec[idx] < self.min_approx_improve
                    ):
                        overlap_improve = False
                        for gid2 in prob._cnt_groups_overlap[gid]:
                            if (
                                approx_improve_vec[prob.gid2ind[gid2]]
                                > self.min_approx_improve
                            ):
                                overlap_improve = True
                                break
                        if overlap_improve:
                            continue
                        prob.nonconverged_groups.append(gid)
                if len(prob.nonconverged_groups) > 0:
                    if verbose:
                        print("Converged: y tolerance")
                    prob.restore()
                    # store the failed groups into the prob

                    for i, g in enumerate(sorted(prob._cnt_groups.keys())):
                        if violated[i] and self._y_converged(approx_improve_vec[i]):
                            prob.nonconverged_groups.append(g)
                    return True

                if self._shrink_trust_region(exact_merit_improve, merit_improve_ratio):
                    prob.restore()
                    if verbose:
                        print("Shrinking trust region")
                    trust_region_size = trust_region_size * self.trust_shrink_ratio
                else:
                    if verbose:
                        print("Growing trust region")
                    trust_region_size = trust_region_size * self.trust_expand_ratio
                    break  # from trust region loop

                if self._x_converged(trust_region_size):
                    if verbose:
                        print("Converged: x tolerance")
                    return True

            sqp_iter = sqp_iter + 1


class RobotSolverOSQP(RobotSolverOSQPBase):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self._save_init_kwargs(dict(method=method))

    # @profile
    def _solve_opt_prob(
        self,
        plan,
        priority,
        callback=None,
        init=True,
        active_ts=None,
        verbose=False,
        resample=False,
        smoothing=False,
        init_traj=[],
        debug=False,
    ):
        if callback is not None:
            viewer = callback()
        self.plan = plan
        if active_ts == None:
            active_ts = (0, plan.horizon - 1)

        plan.save_free_attrs()  # Copies the current free_attrs
        self._prob = Prob(callback=callback)
        self._spawn_parameter_to_ll_mapping(plan, active_ts)

        obj_bexprs = []
        for param, values in self.fixed_objs:
            obj_bexprs.extend(
                self._get_fixed_obj(
                    param,
                    values,
                    "min-vel",
                    active_ts=(
                        active_ts[1] - active_ts[0],
                        active_ts[1] - active_ts[0],
                    ),
                )
            )

        if len(init_traj):
            obj_bexprs.extend(
                self._get_fixed_transfer_obj(
                    plan, "min-vel", init_traj, active_ts=active_ts
                )
            )

        initial_trust_region_size = self.initial_trust_region_size
        end_t = active_ts[1] - active_ts[0]

        if resample:
            tol = 1e-3
            """
            When Optimization fails, resample new values for certain timesteps
            of the trajectory and solver as initialization
            """
            ## this is an objective that places
            ## a high value on matching the resampled values
            # failed_preds = plan.get_failed_preds(active_ts = (active_ts[0]+1, active_ts[1]-1), priority=priority, tol = tol)
            # failed_preds = plan.get_failed_preds(active_ts = (active_ts[0], active_ts[1]-1), priority=priority, tol = tol)
            failed_preds = plan.get_failed_preds(
                active_ts=(active_ts[0], active_ts[1]), priority=priority, tol=tol
            )
            rs_obj = self._resample(plan, failed_preds)
            # _get_transfer_obj returns the expression saying the current trajectory should be close to it's previous trajectory.
            obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
            # obj_bexprs.extend(self._get_transfer_obj(plan, self.transfer_norm))

            self._add_all_timesteps_of_actions(
                plan,
                priority=priority,
                add_nonlin=True,
                active_ts=active_ts,
                verbose=verbose,
            )
            obj_bexprs.extend(rs_obj)
            self._add_obj_bexprs(obj_bexprs)
            initial_trust_region_size = 1e3

        else:
            self._bexpr_to_pred = {}
            if self.col_coeff > 0 and priority >= 0:
                self._add_col_obj(plan, active_ts=active_ts)

            if priority == -2:
                """
                Initialize an linear trajectory while enforceing the linear constraints in the intermediate step.
                """
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(
                    plan,
                    priority=MAX_PRIORITY,
                    active_ts=active_ts,
                    verbose=verbose,
                    add_nonlin=False,
                )
                tol = 1e-3
                initial_trust_region_size = 1e3

            elif priority == -1:
                """
                Solve the optimization problem while enforcing every constraints.
                """
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_first_and_last_timesteps_of_actions(
                    plan,
                    priority=MAX_PRIORITY,
                    active_ts=active_ts,
                    verbose=verbose,
                    add_nonlin=True,
                )
                tol = 1e-3
            elif priority >= 0:
                obj_bexprs.extend(self._get_trajopt_obj(plan, active_ts))
                self._add_obj_bexprs(obj_bexprs)
                self._add_all_timesteps_of_actions(
                    plan,
                    priority=priority,
                    add_nonlin=True,
                    active_ts=active_ts,
                    verbose=verbose,
                )
                tol = 1e-3

        solv = Solver()
        solv.initial_trust_region_size = initial_trust_region_size
        if smoothing:
            solv.initial_penalty_coeff = self.smooth_penalty_coeff
        else:
            solv.initial_penalty_coeff = self.init_penalty_coeff
        solv.max_merit_coeff_increases = self.max_merit_coeff_increases

        # Call the solver on this problem now that it's been constructed
        success = solv.solve(
            self._prob,
            method=self.method,
            tol=tol,
            verbose=verbose,
            osqp_eps_abs=self.osqp_eps_abs,
            osqp_eps_rel=self.osqp_eps_rel,
            osqp_max_iter=self.osqp_max_iter,
            sigma=self.osqp_sigma,
            adaptive_rho=self.adaptive_rho,
        )

        # Update the values of the variables by leveraging the ll_param mapping
        self._update_ll_params()
        if priority >= 0:
            failed_preds = plan.get_failed_preds(
                tol=tol, active_ts=active_ts, priority=priority
            )
            success = success and len(failed_preds) == 0

        """
        if resample:
            # During resampling phases, there must be changes added to sampling_trace
            if len(plan.sampling_trace) > 0 and 'reward' not in plan.sampling_trace[-1]:
                reward = 0
                if len(plan.get_failed_preds(active_ts = active_ts, priority=priority)) == 0:
                    reward = len(plan.actions)
                else:
                    failed_t = plan.get_failed_pred(active_ts=(0,active_ts[1]), priority=priority)[2]
                    for i in range(len(plan.actions)):
                        if failed_t > plan.actions[i].active_timesteps[1]:
                            reward += 1
                plan.sampling_trace[-1]['reward'] = reward
        """

        ##Restore free_attrs values
        plan.restore_free_attrs()
        self.reset_variable()

        return success
