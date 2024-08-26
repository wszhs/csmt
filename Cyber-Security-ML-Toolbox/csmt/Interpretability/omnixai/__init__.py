#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/csmt/Interpretability/")
from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("omnixai")
except DistributionNotFound:
    __version__ = "Please install OmniXAI with setup.py"
else:
    __version__ = dist.version

