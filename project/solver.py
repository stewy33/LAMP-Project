import time
import tqdm
from typing import List

import numpy as np
import osqp
import scipy

from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver as SolverBase
from sco_py.expr import CompExpr, EqExpr, LExpr, LEqExpr
import sco_py.sco_osqp.osqp_utils as osqp_utils

from opentamp.pma.backtrack_ll_solver_OSQP import MAX_PRIORITY
from opentamp.pma.robot_solver import RobotSolverOSQP as RobotSolverOSQPBase


class Solver(SolverBase):
    def solve(
        self,
        prob: Prob,
        method: str,
        tol=None,
        verbose=False,
        osqp_eps_abs=osqp_utils.DEFAULT_EPS_ABS,
        osqp_eps_rel=osqp_utils.DEFAULT_EPS_REL,
        osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER,
        rho: float = osqp_utils.DEFAULT_RHO,
        adaptive_rho: bool = osqp_utils.DEFAULT_ADAPTIVE_RHO,
        sigma: float = osqp_utils.DEFAULT_SIGMA,
    ) -> bool:
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

        # Diagnostic info
        print(f"Linear objs: {len(prob._osqp_lin_objs)}")
        print(f"Quadratic objs: {len(prob._quad_obj_exprs)}")
        print(f"Nonquadratic objs: {len(prob._nonquad_obj_exprs)}")
        print(f"Linear constraints: {len(prob._osqp_lin_cnt_exprs)}")
        print(f"Nonlinear constraints: {len(prob._nonlin_cnt_exprs)}")

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
            return self._gradient_descent(prob, verbose=False)
        elif method == "mala":
            return self._mala(
                prob,
                verbose=verbose,
            )
        elif method == "hmc":
            return self._hmc(
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

    def _get_value(self, prob, penalty_coeff, linear_penalty=False):
        if len(prob._osqp_lin_objs) > 0:
            raise NotImplementedError("Evaluating linear objectives not implemented.")

        quad_value = 0.0
        for bound_expr in prob._quad_obj_exprs:
            quad_value += np.sum(bound_expr.eval())

        nonquad_value = 0.0
        for bound_expr in prob._nonquad_obj_exprs:
            nonquad_value += np.sum(bound_expr.eval())

        nonlin_penalty_value = 0.0
        for bound_expr in prob._nonlin_cnt_exprs:
            cnt_vio = prob._compute_cnt_violation(bound_expr)
            nonlin_penalty_value += penalty_coeff * np.sum(cnt_vio)

        value = quad_value + nonquad_value + nonlin_penalty_value
        info = dict(
            total=value,
            quad_obj=quad_value,
            nonquad_obj=nonquad_value,
            nonlin_cnt_penalty=nonlin_penalty_value,
        )

        if linear_penalty:
            lin_penalty_value = 0.0
            for cnt in prob._osqp_lin_cnt_exprs:
                y = np.dot(np.array([v.val for v in cnt.osqp_vars]), cnt.coeffs)
                cnt_vio = max((y - cnt.ub).item(), (cnt.lb - y).item(), 0)
                lin_penalty_value += penalty_coeff * cnt_vio

            value += lin_penalty_value
            info["total"] = value
            info["lin_cnt_penalty"] = lin_penalty_value

        return value, info

    def _grad(self, prob, penalty_coeff, var_to_index_dict, linear_penalty=False):
        if len(prob._osqp_lin_objs) > 0:
            raise NotImplementedError("Gradients of linear objectives not implemented.")

        obj_grad = np.zeros(len(prob._osqp_vars))
        for bound_expr in prob._quad_obj_exprs + prob._nonquad_obj_exprs:
            x = bound_expr.var.get_value()
            expr_grad = bound_expr.expr.grad(x)

            for var, g in zip(bound_expr.var._osqp_vars.squeeze(), expr_grad):
                obj_grad[var_to_index_dict[var]] += g

        nonlin_cnt_grad = np.zeros(len(prob._osqp_vars))
        for bound_expr in prob._nonlin_cnt_exprs:
            x = bound_expr.var.get_value()
            # No gradient if constraint is satisfied
            if bound_expr.expr.eval(x):
                continue

            expr_y = bound_expr.expr.expr.eval(x)
            expr_grad = bound_expr.expr.expr.grad(x)

            if isinstance(bound_expr.expr, EqExpr):
                expr_grad = np.sign(expr_y - bound_expr.expr.val) * expr_grad
            elif isinstance(bound_expr.expr, LEqExpr) or isinstance(
                bound_expr.expr, LExpr
            ):
                expr_grad = (expr_y >= bound_expr.expr.val) * expr_grad
            else:
                raise NotImplementedError()

            expr_grad = penalty_coeff * expr_grad.sum(axis=0)
            for var, g in zip(bound_expr.var._osqp_vars.squeeze(1), expr_grad):
                nonlin_cnt_grad[var_to_index_dict[var]] += g

        gradient = obj_grad + nonlin_cnt_grad
        info = dict(total=gradient, obj=obj_grad, nonlin_cnt=nonlin_cnt_grad)

        if linear_penalty:
            lin_cnt_grad = np.zeros(len(prob._osqp_vars))
            for cnt in prob._osqp_lin_cnt_exprs:
                y = np.dot(np.array([v.val for v in cnt.osqp_vars]), cnt.coeffs)
                expr_grad = (y >= cnt.ub) * cnt.coeffs - (y <= cnt.lb) * cnt.coeffs

                expr_grad = penalty_coeff * expr_grad
                for var, g in zip(cnt.osqp_vars, expr_grad):
                    lin_cnt_grad[var_to_index_dict[var]] += g

            gradient += lin_cnt_grad
            info["total"] = gradient
            info["lin_cnt"] = lin_cnt_grad

        return gradient, info

    def _get_max_cnt_violation(self, prob):
        """
        Helper function to get the maximum constraint violation among both linear
        and non-linear constraints.
        """
        # Maximum nonlinear constraint violation
        max_vio = prob.get_max_cnt_violation()

        # Maximum linear constraint violation
        for cnt in prob._osqp_lin_cnt_exprs:
            y = np.dot(np.array([v.val for v in cnt.osqp_vars]), cnt.coeffs)
            vio = max((y - cnt.ub).item(), (cnt.lb - y).item(), 0)
            max_vio = max(vio, max_vio)

        return max_vio

    def _get_linear_cnt_matrix(self, prob, var_to_index_dict):
        # Setup the A-matrix and l and u vectors to encode linear constraints for OSQP
        A_mat = np.zeros((len(prob._osqp_lin_cnt_exprs), len(prob._osqp_vars)))
        l_vec = np.zeros(len(A_mat))
        u_vec = np.zeros(len(A_mat))
        for row_num, lin_constraint in enumerate(prob._osqp_lin_cnt_exprs):
            l_vec[row_num] = lin_constraint.lb
            u_vec[row_num] = lin_constraint.ub
            for i in range(lin_constraint.coeffs.shape[0]):
                A_mat[
                    row_num, var_to_index_dict[lin_constraint.osqp_vars[i]]
                ] = lin_constraint.coeffs[i]

        return A_mat, l_vec, u_vec

    def _solve_osqp_program(self, P, q, A, l, u, verbose=False):
        m = osqp.OSQP()
        m.setup(
            P=scipy.sparse.csc_matrix(P),
            q=q,
            A=scipy.sparse.csc_matrix(A),
            rho=osqp_utils.DEFAULT_RHO,
            sigma=osqp_utils.DEFAULT_SIGMA,
            l=l,
            u=u,
            eps_abs=osqp_utils.DEFAULT_EPS_ABS,
            eps_rel=osqp_utils.DEFAULT_EPS_REL,
            delta=1e-07,
            polish=False,
            adaptive_rho=osqp_utils.DEFAULT_ADAPTIVE_RHO,
            warm_start=True,
            verbose=False,
            max_iter=osqp_utils.DEFAULT_MAX_ITER,
        )
        solve_res = m.solve()
        if solve_res.info.status_val == -2 and verbose:
            print(
                "ERROR! OSQP Solver hit max iteration limit. Either reduce your tolerances or increase the max iterations!"
            )
        return solve_res.x

    def _gradient_descent(self, prob, verbose=False):
        lr = 1e-3
        penalty_coeff = 1

        # Setup the linear constraint matrix
        var_to_index_dict = {osqp_var: i for i, osqp_var in enumerate(prob._osqp_vars)}
        A_mat, l_vec, u_vec = self._get_linear_cnt_matrix(prob, var_to_index_dict)

        # Initialize problem variables to zero
        x = np.zeros(len(prob._osqp_vars))
        self._set_prob_vars(prob, x, var_to_index_dict)

        for i in tqdm.trange(100):
            val, val_info = self._get_value(prob, penalty_coeff, linear_penalty=True)
            grad, grad_info = self._grad(
                prob, penalty_coeff, var_to_index_dict, linear_penalty=False
            )

            x = self._solve_osqp_program(
                P=lr * np.eye(len(x)),
                q=grad - lr * x,
                A=A_mat,
                l=l_vec,
                u=u_vec,
                verbose=verbose,
            )
            self._set_prob_vars(prob, x, var_to_index_dict)

            if verbose and i % 1 == 0:
                print(self._get_max_cnt_violation(prob), self.cnt_tolerance, lr)
                print(val_info)

            # Solve succeeded
            if self._get_max_cnt_violation(prob) < self.cnt_tolerance:
                prob._callback()
                return True

    def _mala(
        self,
        prob,
        verbose=False,
    ):
        lr = 1e-3
        penalty_coeff = 1

        # Setup the linear constraint matrix
        var_to_index_dict = {osqp_var: i for i, osqp_var in enumerate(prob._osqp_vars)}
        A_mat, l_vec, u_vec = self._get_linear_cnt_matrix(prob, var_to_index_dict)

        # Initialize problem variables to zero
        x = np.zeros(len(prob._osqp_vars))
        self._set_prob_vars(prob, x, var_to_index_dict)

        for i in tqdm.trange(100):
            val, val_info = self._get_value(prob, penalty_coeff, linear_penalty=True)
            grad, grad_info = self._grad(
                prob, penalty_coeff, var_to_index_dict, linear_penalty=False
            )

            x = self._solve_osqp_program(
                P=lr * np.eye(len(x)),
                q=grad - lr * x,
                A=A_mat,
                l=l_vec,
                u=u_vec,
                verbose=verbose,
            )
            self._set_prob_vars(prob, x, var_to_index_dict)

            if verbose and i % 1 == 0:
                print(self._get_max_cnt_violation(prob), self.cnt_tolerance, lr)
                print(val_info)

            # Solve succeeded
            if self._get_max_cnt_violation(prob) < self.cnt_tolerance:
                prob._callback()
                return True

        # Finish redacting this later
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

    def _hmc(
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
        leapfrog_steps = 3

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
            u = np.random.randn(*x.shape)

            self._set_prob_vars(prob, z, var_to_index_dict)
            new_val = -prob.get_value(penalty_coeff, vectorize=False)
            new_grad = self._grad(prob, penalty_coeff, var_to_index_dict)

            u_leapfrog = u_proposed + 0.5 * self.p * self.target.grad_log(x_leapfrog)

            for l in range(leapfrog_steps):
                if l < leapfrog_steps:
                    p_l = self.p
                else:
                    p_l = 0.5 * self.p

                # update x
                x_leapfrog = x_leapfrog + self.p * u_leapfrog
                # update u
                u_leapfrog = u_leapfrog - p_l * self.target.grad_log(x_leapfrog)

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


"""
Not important. A subclass of original RobotSolverOSQP that uses our custom Solver
object. Only a one line change.
"""


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
