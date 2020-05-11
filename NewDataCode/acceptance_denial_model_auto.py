import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.7143607839557473
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=0.001, dual=True, penalty="l2")),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=6, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.2)),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features=0.4, min_samples_leaf=7, min_samples_split=14, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
