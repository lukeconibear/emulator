"""
Find kernel with highest accuracy within Gaussian Process Regressor
Cross-validated randomised search of hyperparameters
For five random test grid cells, each with 500 iterations
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     RandomizedSearchCV)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct,
                                              ConstantKernel,
                                              WhiteKernel,
                                              Matern,
                                              ExpSineSquared,
                                              RationalQuadratic,
                                              RBF)

with open('/nfs/a336/earlacoa/paper_aia_china/emulator/dfs_gridcell_01_t1-t18.pickle', 'rb') as ds:
    dfs_gridcell = pickle.load(ds)
    
np.random.seed(123)
random_indexes = np.random.randint(0, len(dfs_gridcell.keys()), size=5)

kernel_list = [DotProduct(i) for i in np.arange(0.1, 5, 0.1)] + \
              [ConstantKernel(i) for i in np.arange(0.1, 5, 0.1)] + \
              [WhiteKernel(i) for i in np.arange(0.1, 5, 0.1)] + \
              [Matern(i) for i in np.arange(0.1, 5, 0.1)] + \
              [ExpSineSquared(i) for i in np.arange(0.1, 5, 0.1)] + \
              [RationalQuadratic(i) for i in np.arange(0.1, 5, 0.1)] + \
              [RBF(i) for i in np.arange(0.1, 5, 0.1)]

param_grid = {'kernel': kernel_list,
              'n_restarts_optimizer': np.arange(100, 200, 5)}

for index in random_indexes:
    X = dfs_gridcell[index][['RES', 'IND', 'TRA', 'AGR', 'POW']].values
    y = dfs_gridcell[index].PM2_5_DRY.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    model = GaussianProcessRegressor(random_state=123)
    gp = RandomizedSearchCV(model,
                            param_grid,
                            n_jobs=-1,
                            n_iter=500,
                            random_state=123,
                            verbose=2)
    gp.fit(X_train, y_train)

    print(f"val. score: {gp.best_score_:.4f}")
    print(f"test score: {gp.score(X_test, y_test)}")
    print(f"best estimator: {gp.best_estimator_}")