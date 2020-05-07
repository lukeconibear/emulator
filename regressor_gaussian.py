# -*- coding: utf-8 -*-

"""
Custom TPOT config based on regressor.py
- Including:
    - Gaussian Process Regressor with Matern kernel
    - Additional preprocessors e.g. PowerTransformer, QuantileTransformer
- Excluding:
    - All other regressors
    - Dimensionality reduction e.g. FeatureAgglomeration, PCA, FastICA
    - Feature selection e.g. VarianceThreshold, SelectPercentile, SelectFwe
"""

import numpy as np

tpot_config = {

    # Regressors
    'sklearn.gaussian_process.GaussianProcessRegressor': {
        'kernel': {
            'sklearn.gaussian_process.kernels.Matern': {
                'length_scale': np.arange(0.1, 5.0, 0.1),
                'nu': [0.5, 1.5, 2.5]
            }
        },
        'normalize_y': [True, False],
        'n_restarts_optimizer': np.arange(5, 300, 5)
    },

    # Preprocesssors
    'sklearn.preprocessing.PowerTransformer': {
    },

    'sklearn.preprocessing.QuantileTransformer': {
        'output_distribution': ['uniform', 'normal']
    },

    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    # Selectors
    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}
