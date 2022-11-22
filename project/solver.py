import numpy as np

from sco_py.sco_osqp.prob import Prob
from sco_py.sco_osqp.solver import Solver

from opentamp.pma.backtrack_ll_solver_OSQP import MAX_PRIORITY
from opentamp.pma.robot_solver import RobotSolverOSQP


class RobotSolverOSQP(RobotSolverOSQP):
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
            method="penalty_sqp",
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
