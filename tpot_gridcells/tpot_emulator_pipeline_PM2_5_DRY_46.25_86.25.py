import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=123)

# Average CV score on the training set was: 0.9781034161338924
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.25, kernel="additive_chi2", n_components=10),
    StackingEstimator(estimator=GaussianProcessRegressor(kernel=Matern(length_scale=0.7000000000000001, nu=1.5), n_restarts_optimizer=130, normalize_y=True)),
    GaussianProcessRegressor(kernel=Matern(length_scale=2.9000000000000004, nu=2.5), n_restarts_optimizer=155, normalize_y=True)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 123)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
