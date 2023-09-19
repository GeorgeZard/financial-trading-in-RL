import warnings

warnings.warn("Please use fin_utils.candles.labels instead of fin_utils.labels", category=DeprecationWarning)
from fin_utils.candles.labels import generate_optimal_limits, generate_prc_volatility_pred
