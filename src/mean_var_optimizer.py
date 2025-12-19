# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import pickle
import time
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from . import base_optimizer
from .mean_var_parameters import MeanVarParameters
from .portfolio import Portfolio
from .settings import ApiSettings

"""
Module: Mean-Variance Optimization
==================================
This module implements data structures and a class for Mean-Variance
portfolio optimization (Markowitz optimization).

A Mean-Variance optimizer chooses asset weights that maximize expected return
while minimizing portfolio variance (or, equivalently, minimizing a
risk-penalized loss).

Key features
------------
* Set up problem using different interfaces (CVXPY with bounds/parameters, cuOpt).
* Build models with customizable constraints based on MeanVarParameters.
* Print optimization results with detailed performance metrics and allocation.

Public classes
--------------
``MeanVar``
    Main Mean-Variance portfolio optimizer class that supports multiple solver
    interfaces (CVXPY and cuOpt). Handles Mean-Variance optimization with
    customizable constraints including weight bounds, cash allocation, leverage
    limits, variance hard limits, turnover restrictions, and cardinality constraints.

Usage Examples
--------------
Standard CVXPY solver (uses bounds by default):
    >>> optimizer = MeanVar(returns_dict, mean_var_params)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )

cuOpt GPU solver:
    >>> api_settings = ApiSettings(api="cuopt_python")
    >>> optimizer = MeanVar(returns_dict, mean_var_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem({
    ...     "time_limit": 60
    ... })

CVXPY with parameters:
    >>> api_settings = ApiSettings(
    ...     api="cvxpy",
    ...     weight_constraints_type="parameter",
    ...     cash_constraints_type="parameter"
    ... )
    >>> optimizer = MeanVar(returns_dict, mean_var_params, api_settings=api_settings)
    >>> result, portfolio = optimizer.solve_optimization_problem(
    ...     {"solver": cp.CLARABEL}
    ... )
"""


class MeanVar(base_optimizer.BaseOptimizer):
    """
    Mean-Variance portfolio optimizer with multiple API support.

    Solves Mean-Variance (Markowitz) optimization problems with the following constraints:
        - Weight bounds
        - Cash bounds
        - Leverage constraint
        - Hard variance limit (optional)
        - Turnover constraint (optional)
        - Cardinality constraint (optional)

    Key features:
    - Risk-adjusted return optimization using variance as risk measure
    - Supports both CVXPY and cuOpt Python APIs
    - GPU acceleration available via cuOpt
    - Performance monitoring with timing metrics
    - Automatic setup based on API choice
    """

    def __init__(
        self,
        returns_dict: dict,
        mean_var_params: MeanVarParameters,
        api_settings: Optional[ApiSettings] = None,
        existing_portfolio: Optional[Portfolio] = None,
    ):
        """Initialize Mean-Variance optimizer with data and constraints.

        Parameters
        ----------
        returns_dict : dict
            Input data containing regime info, mean returns, and covariance matrix.
        mean_var_params : MeanVarParameters
            Constraint parameters and optimization settings (deep-copied).
        api_settings : ApiSettings, optional
            API configuration including solver choice and constraint types.
            Uses CVXPY with bounds if not provided.
        existing_portfolio : Portfolio, optional
            An existing portfolio to measure the turnover from.
        """
        super().__init__(returns_dict, existing_portfolio, "variance")

        if api_settings is None:
            api_settings = ApiSettings()

        self.api_settings = api_settings
        self.api_choice = api_settings.api

        self.regime_name = returns_dict["regime"]["name"]
        self.regime_range = returns_dict["regime"]["range"]
        self.mean = returns_dict["mean"]
        self.covariance = returns_dict["covariance"]
        self.existing_portfolio = existing_portfolio
        self.params = self._store_mean_var_params(mean_var_params)

        # Set up the optimization problem based on API choice
        self._setup_optimization_problem()

        self.optimal_portfolio = None

        self._result_columns = [
            "regime",
            "solver",
            "solve time",
            "return",
            "variance",
            "obj",
        ]

    def _store_mean_var_params(self, mean_var_params: MeanVarParameters):
        """
        Store the Mean-Variance parameters in the optimizer.

        If w_min and w_max are input as floats, convert them to ndarrays
        with the same value repeated for all assets. Otherwise, store
        the ndarrays as is in the deepcopy.
        """
        params_copy = copy.deepcopy(mean_var_params)

        params_copy.w_min = self._update_weight_constraints(params_copy.w_min)
        params_copy.w_max = self._update_weight_constraints(params_copy.w_max)

        return params_copy

    def _setup_optimization_problem(self):
        """
        Set up the optimization problem based on the selected API choice.

        This unified method handles setup for both CVXPY and cuOpt APIs:
        - Times the setup process
        - Scales risk aversion parameter
        - Calls the appropriate API-specific setup method
        """
        set_up_start = time.time()

        if self.api_settings.scale_risk_aversion:
            self._scale_risk_aversion()

        # Call the appropriate setup method based on API choice
        if self.api_choice == "cvxpy":
            self._setup_cvxpy_problem()
            self._assign_cvxpy_parameter_values()

            # Save problem to pickle if requested
            pickle_path = self.api_settings.pickle_save_path
            if pickle_path is not None:
                self._save_problem_pickle(pickle_path)

        elif self.api_choice == "cuopt_python":
            (
                self._cuopt_problem,
                self._cuopt_variables,
                self.cuopt_timing_dict,
            ) = self._setup_cuopt_problem()
        else:
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        set_up_end = time.time()
        self.set_up_time = set_up_end - set_up_start

    def _scale_risk_aversion(self):
        """
        Heuristically scale risk aversion parameter by the ratio of
        the maximum return over standard deviation for single-asset portfolios.
        """
        # Calculate return/risk ratio for each asset
        std_devs = np.sqrt(np.diag(self.covariance))
        # Avoid division by zero
        std_devs = np.maximum(std_devs, 1e-10)
        return_risk_ratios = self.mean / std_devs

        self._risk_aversion_scalar = np.max(return_risk_ratios)

        self.params.update_risk_aversion(
            self.params.risk_aversion * self._risk_aversion_scalar
        )

    def _setup_cvxpy_problem(self):
        """
        Build the mean-variance optimization problem using CVXPY.

        Supports the following types of problems:
            1. (QP) 'basic mean-variance': basic Markowitz problem
                Minimize: lambda_risk * w^T Σ w - μ^T w
                Subject to: sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar.

            2. (QP) 'mean-variance with limit': hard limit on variance
                Maximize: μ^T w
                Subject to: w^T Σ w <= var_limit,
                            sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar.

            3. (QP) 'mean-variance with turnover':
                Minimize: lambda_risk * w^T Σ w - μ^T w
                Subject to: sum{w} + c = 1,
                            w_min_i <= w_i <= w_max_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar,
                            ||w - existing_portfolio||_1 <= T_tar.

            4. (MIQP) 'mean-variance with cardinality':
                Minimize: lambda_risk * w^T Σ w - μ^T w
                Subject to: sum{w} + c = 1,
                            w_min_i * y_i <= w_i <= w_max_i * y_i,
                            c_min <= c <= c_max,
                            ||w||_1 <= L_tar,
                            sum{y_i} <= cardinality.

        We can also combine the above constraints to form a more complex problem.
        """
        num_assets = self.n_assets

        # Create variables based on constraint type settings
        if self.api_settings.weight_constraints_type == "bounds":
            self.w = cp.Variable(
                num_assets,
                name="weights",
                bounds=[self.params.w_min, self.params.w_max],
            )
        else:
            self.w = cp.Variable(num_assets, name="weights")
            self.w_min_param = cp.Parameter(num_assets, name="w_min")
            self.w_max_param = cp.Parameter(num_assets, name="w_max")

        if self.api_settings.cash_constraints_type == "bounds":
            self.c = cp.Variable(
                1, name="cash", bounds=[self.params.c_min, self.params.c_max]
            )
        else:
            self.c = cp.Variable(1, name="cash")
            self.c_min_param = cp.Parameter(name="c_min")
            self.c_max_param = cp.Parameter(name="c_max")

        # Create parameters for optimization parameters
        self.risk_aversion_param = cp.Parameter(nonneg=True, name="risk_aversion")
        self.L_tar_param = cp.Parameter(nonneg=True, name="L_tar")
        self.T_tar_param = cp.Parameter(nonneg=True, name="T_tar")
        self.var_limit_param = cp.Parameter(nonneg=True, name="var_limit")
        self.cardinality_param = cp.Parameter(name="cardinality")

        # Set up expressions for optimization
        self.expected_ptf_returns = self.mean.T @ self.w
        self.portfolio_variance = cp.quad_form(self.w, self.covariance)

        # Add variable bounds constraints (only if using parameter constraints)
        constraints = []
        if self.api_settings.weight_constraints_type == "parameter":
            constraints.extend(
                [
                    self.w_min_param <= self.w,
                    self.w <= self.w_max_param,
                ]
            )
        if self.api_settings.cash_constraints_type == "parameter":
            constraints.extend(
                [
                    self.c_min_param <= self.c,
                    self.c <= self.c_max_param,
                ]
            )

        # Set up common constraints
        if self.params.cardinality is not None:
            raise NotImplementedError("MIQP is not implemented yet.")
            # self._problem_type = "MIQP"
            # print(f"{'=' * 50}")
            # print("MIXED-INTEGER QUADRATIC PROGRAMMING (MIQP) SETUP")
            # print(f"{'=' * 50}")
            # print(f"Cardinality Constraint: K ≤ {self.params.cardinality} assets")
            # print(f"{'=' * 50}")
            # y = cp.Variable(num_assets, boolean=True, name="cardinality")

            # if self.api_settings.weight_constraints_type == "parameter":
            #     constraints.extend(
            #         [
            #             cp.multiply(self.w_min_param, y) <= self.w,
            #             self.w <= cp.multiply(self.w_max_param, y),
            #         ]
            #     )
            # else:
            #     constraints.extend(
            #         [
            #             cp.multiply(self.params.w_min, y) <= self.w,
            #             self.w <= cp.multiply(self.params.w_max, y),
            #         ]
            #     )

            # constraints.extend(
            #     [
            #         cp.sum(self.w) + self.c == 1,
            #         cp.norm1(self.w) <= self.L_tar_param,
            #         cp.sum(y) <= self.cardinality_param,
            #     ]
            # )
        else:
            constraints.extend(
                [
                    cp.sum(self.w) + self.c == 1,
                    cp.norm1(self.w) <= self.L_tar_param,
                ]
            )

        # Set up objective
        if self.params.var_limit is None:
            obj = cp.Minimize(
                self.risk_aversion_param * self.portfolio_variance
                - self.expected_ptf_returns
            )
        else:
            obj = cp.Maximize(self.expected_ptf_returns)
            constraints.append(self.portfolio_variance <= self.var_limit_param)

        # Set up turnover constraint
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            w_prev = np.array(self.existing_portfolio.weights)
            z = self.w - w_prev
            constraints.append(cp.norm(z, 1) <= self.T_tar_param)

        # Set up group constraints
        if self.params.group_constraints is not None:
            for group_constraint in self.params.group_constraints:
                tickers_index = [
                    self.tickers.index(ticker)
                    for ticker in group_constraint["tickers"]
                ]
                constraints.append(
                    cp.sum(self.w[tickers_index])
                    <= group_constraint["weight_bounds"]["w_max"]
                )
                constraints.append(
                    cp.sum(self.w[tickers_index])
                    >= group_constraint["weight_bounds"]["w_min"]
                )

        self.optimization_problem = cp.Problem(obj, constraints)

    def _assign_cvxpy_parameter_values(self):
        """
        Assign values to all CVXPY parameters from current data and parameter settings.
        """
        if self.api_settings.weight_constraints_type == "parameter":
            self.w_min_param.value = self.params.w_min
            self.w_max_param.value = self.params.w_max

        if self.api_settings.cash_constraints_type == "parameter":
            self.c_min_param.value = self.params.c_min
            self.c_max_param.value = self.params.c_max

        self.risk_aversion_param.value = self.params.risk_aversion
        self.L_tar_param.value = self.params.L_tar

        if self.params.T_tar is not None:
            self.T_tar_param.value = self.params.T_tar

        if self.params.var_limit is not None:
            self.var_limit_param.value = self.params.var_limit

        if self.params.cardinality is not None:
            self.cardinality_param.value = self.params.cardinality

    def _setup_cuopt_problem(self):
        """
        Set up Mean-Variance optimization problem using cuOpt Python API.

        Creates cuOpt Problem instance with variables, constraints, and objective
        for Mean-Variance portfolio optimization.

        Note: cuOpt supports QP problems via SOCP reformulation or direct QP support.

        Returns
        -------
        problem : cuopt Problem instance
            cuOpt problem instance ready to solve
        variables : dict
            Dictionary containing problem variables for result extraction
        timing_dict : dict
            Timing information for each setup phase
        """
        from cuopt.linear_programming.problem import (
            CONTINUOUS,
            INTEGER,
            MAXIMIZE,
            MINIMIZE,
            Problem,
            LinearExpression,
        )

        num_assets = self.n_assets
        timing_dict = {}

        start_time = time.time()
        problem = Problem("Mean-Variance Portfolio Optimization")
        timing_dict["problem_creation"] = time.time() - start_time

        variables = {}

        # Add portfolio weight variables
        start_time = time.time()
        variables["w"] = []
        for i in range(num_assets):
            w_var = problem.addVariable(
                lb=float(self.params.w_min[i]),
                ub=float(self.params.w_max[i]),
                vtype=CONTINUOUS,
                name=f"w_{i}",
            )
            variables["w"].append(w_var)
        timing_dict["weight_variables"] = time.time() - start_time

        # Add cash variable
        start_time = time.time()
        variables["c"] = problem.addVariable(
            lb=float(self.params.c_min),
            ub=float(self.params.c_max),
            vtype=CONTINUOUS,
            name="cash",
        )
        timing_dict["cash_variable"] = time.time() - start_time

        # Add budget constraint: sum(w) + c = 1
        start_time = time.time()
        budget_vars = variables["w"] + [variables["c"]]
        budget_coeffs = [1.0] * num_assets + [1.0]
        budget_expr = LinearExpression(budget_vars, budget_coeffs, 0.0)
        problem.addConstraint(budget_expr == 1.0, name="budget_constraint")
        timing_dict["budget_constraint"] = time.time() - start_time

        # Add leverage constraint
        if self.params.L_tar < float("inf"):
            start_time = time.time()
            variables["w_pos"] = []
            variables["w_neg"] = []

            for i in range(num_assets):
                w_pos = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"w_pos_{i}"
                )
                w_neg = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"w_neg_{i}"
                )
                variables["w_pos"].append(w_pos)
                variables["w_neg"].append(w_neg)

            for i in range(num_assets):
                decomp_vars = [
                    variables["w"][i],
                    variables["w_pos"][i],
                    variables["w_neg"][i],
                ]
                decomp_coeffs = [1.0, -1.0, 1.0]
                decomp_expr = LinearExpression(decomp_vars, decomp_coeffs, 0.0)
                problem.addConstraint(
                    decomp_expr == 0.0, name=f"weight_decomposition_{i}"
                )

            leverage_vars = variables["w_pos"] + variables["w_neg"]
            leverage_coeffs = [1.0] * (2 * num_assets)
            leverage_expr = LinearExpression(leverage_vars, leverage_coeffs, 0.0)
            problem.addConstraint(
                leverage_expr <= self.params.L_tar, name="leverage_constraint"
            )
            timing_dict["leverage_constraints"] = time.time() - start_time
        else:
            timing_dict["leverage_constraints"] = 0.0

        # Add cardinality constraints
        if self.params.cardinality is not None:
            start_time = time.time()
            variables["y"] = []

            for i in range(num_assets):
                y_var = problem.addVariable(
                    lb=0, ub=1, vtype=INTEGER, name=f"y_{i}"
                )
                variables["y"].append(y_var)

            cardinality_coeffs = [1.0] * num_assets
            cardinality_expr = LinearExpression(
                variables["y"], cardinality_coeffs, 0.0
            )
            problem.addConstraint(
                cardinality_expr <= self.params.cardinality,
                name="cardinality_constraint",
            )

            for i in range(num_assets):
                lower_vars = [variables["w"][i], variables["y"][i]]
                lower_coeffs = [1.0, -float(self.params.w_min[i])]
                lower_expr = LinearExpression(lower_vars, lower_coeffs, 0.0)
                problem.addConstraint(
                    lower_expr >= 0.0, name=f"cardinality_lower_{i}"
                )

                upper_vars = [variables["w"][i], variables["y"][i]]
                upper_coeffs = [1.0, -float(self.params.w_max[i])]
                upper_expr = LinearExpression(upper_vars, upper_coeffs, 0.0)
                problem.addConstraint(
                    upper_expr <= 0.0, name=f"cardinality_upper_{i}"
                )

            timing_dict["cardinality_constraints"] = time.time() - start_time
            print(
                f"Cardinality Constraint: K ≤ {self.params.cardinality} assets (MIQP)"
            )
        else:
            timing_dict["cardinality_constraints"] = 0.0

        # Add turnover constraint
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            start_time = time.time()
            w_prev = np.array(self.existing_portfolio.weights)
            variables["turnover_pos"] = []
            variables["turnover_neg"] = []

            for i in range(num_assets):
                to_pos = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"turnover_pos_{i}"
                )
                to_neg = problem.addVariable(
                    lb=0.0, vtype=CONTINUOUS, name=f"turnover_neg_{i}"
                )
                variables["turnover_pos"].append(to_pos)
                variables["turnover_neg"].append(to_neg)

            for i in range(num_assets):
                decomp_vars = [
                    variables["w"][i],
                    variables["turnover_pos"][i],
                    variables["turnover_neg"][i],
                ]
                decomp_coeffs = [1.0, -1.0, 1.0]
                decomp_expr = LinearExpression(decomp_vars, decomp_coeffs, 0.0)
                problem.addConstraint(
                    decomp_expr == float(w_prev[i]),
                    name=f"turnover_decomposition_{i}",
                )

            turnover_vars = variables["turnover_pos"] + variables["turnover_neg"]
            turnover_coeffs = [1.0] * (2 * num_assets)
            turnover_expr = LinearExpression(turnover_vars, turnover_coeffs, 0.0)
            problem.addConstraint(
                turnover_expr <= self.params.T_tar, name="turnover_constraint"
            )
            timing_dict["turnover_constraints"] = time.time() - start_time
        else:
            timing_dict["turnover_constraints"] = 0.0

        # Add group constraints
        if self.params.group_constraints is not None:
            start_time = time.time()
            for group_idx, group_constraint in enumerate(self.params.group_constraints):
                tickers_index = [
                    self.tickers.index(ticker)
                    for ticker in group_constraint["tickers"]
                ]

                if len(tickers_index) > 0:
                    group_vars = [variables["w"][i] for i in tickers_index]
                    group_coeffs = [1.0] * len(tickers_index)
                    group_sum_expr = LinearExpression(group_vars, group_coeffs, 0.0)

                    problem.addConstraint(
                        group_sum_expr <= group_constraint["weight_bounds"]["w_max"],
                        name=f"group_{group_idx}_upper",
                    )
                    problem.addConstraint(
                        group_sum_expr >= group_constraint["weight_bounds"]["w_min"],
                        name=f"group_{group_idx}_lower",
                    )

            timing_dict["group_constraints"] = time.time() - start_time
            print(f"Group Constraints: {len(self.params.group_constraints)} groups")
        else:
            timing_dict["group_constraints"] = 0.0

        # Set up objective function
        # Note: For QP, cuOpt uses setQuadraticObjective or requires SOCP reformulation
        start_time = time.time()

        # Build expected return expression
        expected_return_coeffs = [float(self.mean[i]) for i in range(num_assets)]
        expected_return_expr = LinearExpression(
            variables["w"], expected_return_coeffs, 0.0
        )

        # For Mean-Variance, we need quadratic objective: w^T Σ w
        # cuOpt supports QP via setQuadraticObjective
        if self.params.var_limit is None:
            # Minimize: risk_aversion * w^T Σ w - μ^T w
            # Linear part: -μ^T w
            linear_coeffs = [-float(self.mean[i]) for i in range(num_assets)]
            linear_expr = LinearExpression(variables["w"], linear_coeffs, 0.0)
            problem.setObjective(linear_expr, sense=MINIMIZE)

            # Add quadratic part: risk_aversion * w^T Σ w
            Q = self.params.risk_aversion * self.covariance
            problem.setQuadraticObjective(variables["w"], Q)
        else:
            # Maximize: μ^T w subject to w^T Σ w <= var_limit
            problem.setObjective(expected_return_expr, sense=MAXIMIZE)
            # Add variance constraint as quadratic constraint
            problem.addQuadraticConstraint(
                variables["w"],
                self.covariance,
                self.params.var_limit,
                name="variance_limit",
            )

        timing_dict["objective_setup"] = time.time() - start_time

        print(f"{'=' * 50}")
        print("cuOpt MEAN-VARIANCE PROBLEM SETUP COMPLETED")
        print(f"{'=' * 50}")
        print(f"Variables: {num_assets} weights + 1 cash")
        if self.params.cardinality is not None:
            print(f"           + {num_assets} cardinality (integer)")
        if self.params.L_tar < float("inf"):
            print(f"           + {2 * num_assets} leverage decomposition")
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            print(f"           + {2 * num_assets} turnover decomposition")
        print("Problem Type: QP (Quadratic Programming)")
        print(f"{'=' * 50}")

        return problem, variables, timing_dict

    def _solve_cuopt_problem(self, solver_settings: dict = None):
        """
        Solve Mean-Variance optimization using cuOpt.

        Parameters
        ----------
        solver_settings : dict, optional
            cuOpt solver configuration.

        Returns
        -------
        result_row : pd.Series
            Performance metrics
        weights : np.ndarray
            Optimal asset weights
        cash : float
            Optimal cash allocation
        """
        from cuopt.linear_programming.solver_settings import SolverSettings

        settings = SolverSettings()
        if solver_settings:
            for param, value in solver_settings.items():
                settings.set_parameter(param, value)

        total_start = time.time()
        self._cuopt_problem.solve(settings)
        total_end = time.time()
        total_time = total_end - total_start
        solve_time = self._cuopt_problem.SolveTime
        self.cuopt_api_overhead = total_time - solve_time

        if self._cuopt_problem.Status.name != "Optimal":
            raise RuntimeError(
                f"cuOpt failed to find optimal solution. Status: "
                f"{self._cuopt_problem.Status.name}"
            )

        weights = np.array([var.getValue() for var in self._cuopt_variables["w"]])
        cash = self._cuopt_variables["c"].getValue()

        expected_return = np.dot(self.mean, weights)
        variance_value = weights @ self.covariance @ weights

        objective_value = self._cuopt_problem.ObjValue

        result_row = pd.Series(
            [
                self.regime_name,
                "cuOpt",
                solve_time,
                expected_return,
                variance_value,
                objective_value,
            ],
            index=self._result_columns,
        )

        print(f"cuOpt solution found in {solve_time:.2f} seconds")
        print(f"Status: {self._cuopt_problem.Status.name}")
        print(f"Objective value: {objective_value:.6f}")

        return result_row, weights, cash

    def _solve_cvxpy_problem(self, solver_settings: dict):
        """
        Solve the optimization problem using the user-specified solver.

        Parameters
        ----------
        solver_settings : dict
            Solver configuration dict for CVXPY.Problem.solve().

        Returns
        -------
        result_row : pd.Series
            Performance metrics
        weights : np.ndarray
            Optimal asset weights
        cash : float
            Optimal cash allocation
        """
        self.optimization_problem.solve(**solver_settings)
        weights = self.w.value
        cash = self.c.value

        solver_stats = getattr(self.optimization_problem, "solver_stats", None)
        reported_solve_time = (
            getattr(solver_stats, "solve_time", None)
            if solver_stats is not None
            else None
        )

        solver_time = (
            float(reported_solve_time)
            if reported_solve_time is not None
            else self.optimization_problem._solve_time
        )

        self.cvxpy_api_overhead = (
            self.optimization_problem._solve_time - solver_time
            if reported_solve_time is not None
            else None
        )

        result_row = pd.Series(
            [
                self.regime_name,
                str(solver_settings["solver"]),
                solver_time,
                self.expected_ptf_returns.value,
                self.portfolio_variance.value,
                self.optimization_problem.value,
            ],
            index=self._result_columns,
        )

        return result_row, weights, cash

    def _save_problem_pickle(self, pickle_save_path: str):
        """
        Save the CVXPY optimization problem to a pickle file.

        Parameters
        ----------
        pickle_save_path : str
            Path where to save the pickle file
        """
        try:
            os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)
            with open(pickle_save_path, "wb") as f:
                pickle.dump(self.optimization_problem, f)
            print(f"Mean-Variance problem saved to: {pickle_save_path}")
        except Exception as e:
            print(f"Warning: Failed to save Mean-Variance problem to pickle: {e}")

    def _print_mean_var_results(
        self,
        result_row: pd.Series,
        portfolio: Portfolio,
        time_results: dict,
        min_percentage: float = 1,
    ):
        """
        Display Mean-Variance optimization results and portfolio allocation.

        Parameters
        ----------
        result_row : pd.Series
            Optimization results
        portfolio : Portfolio
            Portfolio to display the readable allocation
        time_results : dict
            Additional timing breakdown
        min_percentage : float, default 1
            Only assets with absolute allocation >= min_percentage% will be shown
        """
        solver_name = result_row["solver"]
        solve_time = result_row["solve time"]
        expected_return = result_row["return"]
        variance_value = result_row["variance"]
        objective_value = result_row["obj"]

        print(f"\n{'=' * 60}")
        print("MEAN-VARIANCE OPTIMIZATION RESULTS")
        print(f"{'=' * 60}")

        print("PROBLEM CONFIGURATION")
        print(f"{'-' * 30}")
        print(f"Solver:              {solver_name}")
        print(f"Regime:              {self.regime_name}")
        print(f"Time Period:         {self.regime_range[0]} to {self.regime_range[1]}")
        print(f"Assets:              {self.n_assets}")

        if self.params.cardinality is not None:
            print(f"Cardinality Limit:   {self.params.cardinality} assets")
        if self.params.var_limit is not None:
            print(f"Variance Hard Limit: {self.params.var_limit:.6f}")
        if self.existing_portfolio is not None and self.params.T_tar is not None:
            print(f"Turnover Constraint: {self.params.T_tar:.3f}")

        print("\nPERFORMANCE METRICS")
        print(f"{'-' * 30}")
        print(
            f"Expected Return:     {expected_return:.6f} ({expected_return * 100:.4f}%)"
        )
        print(f"Variance:            {variance_value:.6f}")
        print(f"Std Deviation:       {np.sqrt(variance_value):.6f}")
        print(f"Objective Value:     {objective_value:.6f}")

        print("\nSOLVING PERFORMANCE")
        print(f"{'-' * 30}")
        if hasattr(self, "set_up_time"):
            print(f"Setup Time:          {self.set_up_time:.4f} seconds")
        if hasattr(self, "cvxpy_api_overhead") and self.cvxpy_api_overhead is not None:
            print(f"CVXPY API Overhead:  {self.cvxpy_api_overhead:.4f} seconds")
        if hasattr(self, "cuopt_api_overhead"):
            print(f"cuOpt API Overhead:  {self.cuopt_api_overhead:.4f} seconds")
        print(f"Solve Time:          {solve_time:.4f} seconds")

        for key, value in time_results.items():
            print(f"{key.title():20} {value:.4f} seconds")

        print("\nOPTIMAL PORTFOLIO ALLOCATION")
        print(f"{'-' * 30}")
        portfolio.print_clean(verbose=True, min_percentage=min_percentage)

        print(f"{'=' * 60}\n")

    def solve_optimization_problem(
        self, solver_settings: dict = None, print_results: bool = True
    ):
        """
        Unified solve method that calls the appropriate API-specific solver.

        Parameters
        ----------
        solver_settings : dict, optional
            Solver configuration. Format depends on API choice:
            - CVXPY: {"solver": cp.CLARABEL, "verbose": True}
            - cuOpt: {"time_limit": 60}
        print_results : bool, default True
            Enable formatted result output to console.

        Returns
        -------
        result_row : pd.Series
            Performance metrics: regime, solve_time, return, variance, objective.
        portfolio : Portfolio
            Optimized portfolio with weights and cash allocation.
        """
        time_results = {}

        if self.api_choice == "cvxpy":
            if solver_settings is None or solver_settings.get("solver") is None:
                raise ValueError("A solver must be provided for CVXPY API")
            result_row, weights, cash = self._solve_cvxpy_problem(solver_settings)
            portfolio_name = str(solver_settings["solver"]) + "_optimal"
        elif self.api_choice == "cuopt_python":
            result_row, weights, cash = self._solve_cuopt_problem(solver_settings)
            portfolio_name = "cuOpt_optimal"
        else:
            raise ValueError(f"Unsupported api_choice: {self.api_choice}")

        portfolio = Portfolio(
            name=portfolio_name,
            tickers=self.tickers,
            weights=weights,
            cash=cash,
            time_range=self.regime_range,
        )

        if print_results:
            self._print_mean_var_results(
                result_row, portfolio, time_results, min_percentage=1
            )

        return result_row, portfolio

    def _extract_problem_cone_data(self, problem_data_dir: str):
        """
        Extract the cone data from the problem and save to pickle file.

        Parameters
        ----------
        problem_data_dir : str
            Path where to save the pickle file
        """
        data = self.optimization_problem.get_problem_data("SCS")
        P = data[0].get("P", None)
        q = data[0].get("c", None)
        A = data[0].get("A", None)
        b = data[0].get("b", None)
        dims = data[0].get("dims", None)

        os.makedirs(problem_data_dir, exist_ok=True)

        regime_name = getattr(self, "regime_name", "unknown")
        filename = f"mean_var_{regime_name}.pkl"
        pickle_file_path = os.path.join(problem_data_dir, filename)

        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Problem data saved to: {pickle_file_path}")

        return P, q, A, b, dims

