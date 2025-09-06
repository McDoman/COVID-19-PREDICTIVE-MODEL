# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:28:39 2025

@author: LAdedo
"""

import pandas as pd
res = pd.DataFrame(best["params"], index=[0])
res["model"] = best["name"]; res["cv_roc_auc"] = best["auc"]
print(res.T)