import warnings

warnings.warn("Please use fin_utils.candles.labels instead of fin_utils.labels", category=DeprecationWarning)
from fin_utils.candles.labels import smooth_oracle_labels, oracle_labels, triple_barrier_labels
