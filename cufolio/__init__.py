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

"""
cufolio package - re-exports from src for development use.

Allows imports to work without installing (venv only): package __path__ points at src.
"""

from pathlib import Path

_workspace_root = Path(__file__).parent.parent
_src_path = str(_workspace_root / "src")

__path__ = [_src_path]

import cufolio.backtest as backtest
import cufolio.base_parameters as base_parameters
import cufolio.cvar_data as cvar_data
import cufolio.cvar_optimizer as cvar_optimizer
import cufolio.cvar_parameters as cvar_parameters
import cufolio.cvar_utils as cvar_utils
import cufolio.mean_variance_optimizer as mean_variance_optimizer
import cufolio.mean_variance_parameters as mean_variance_parameters
import cufolio.portfolio as portfolio
import cufolio.rebalance as rebalance
import cufolio.scenario_generation as scenario_generation
import cufolio.settings as settings
import cufolio.strategy_backtest as strategy_backtest
import cufolio.utils as utils

from cufolio.base_parameters import BaseParameters
from cufolio.cvar_parameters import CvarParameters
from cufolio.mean_variance_parameters import MeanVarianceParameters
from cufolio.settings import (
    ApiSettings,
    KDESettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)

__all__ = [
    "backtest",
    "base_parameters",
    "cvar_data",
    "cvar_optimizer",
    "cvar_parameters",
    "cvar_utils",
    "mean_variance_optimizer",
    "mean_variance_parameters",
    "portfolio",
    "rebalance",
    "scenario_generation",
    "settings",
    "strategy_backtest",
    "utils",
    "BaseParameters",
    "CvarParameters",
    "MeanVarianceParameters",
    "ApiSettings",
    "KDESettings",
    "ReturnsComputeSettings",
    "ScenarioGenerationSettings",
]
