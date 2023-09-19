import pandas as pd
from pathlib import Path
from fin_utils.candles import resample_candles

from fin_utils.candles.loading import load_feather_dir

# def load_feather_dir(path: Path, resample=None, pairs=None) :
#     path = Path(path).expanduser()
#     feather_dict = dict()
#
#     assert path.exists()
#     assert len(list(path.iterdir())) > 0
#     # ensure all pair names are lowercase
#     if pairs is not None:
#         pairs = [p.lower() for p in pairs]
#
#     for f in path.iterdir():
#         # if pair filter is present and file stem is not in pairs skip.
#         if pairs and f.stem.lower() not in pairs:
#             continue
#         if f.suffix == '.feather':
#             df = pd.read_feather(str(f))
#             if 'date' in df.columns:
#                 df.set_index('date', inplace=True)
#             elif 'time' in df.columns:
#                 df.set_index('time', inplace=True)
#             feather_dict[f.stem.lower()] = df
#             if resample:
#                 feather_dict[f.stem] = resample_candles(feather_dict[f.stem], window=resample)
#     assert len(feather_dict) > 0, f"No feathers were loaded from {str(path)}"
#     return feather_dict


if __name__ == '__main__':
    candle_dict = load_feather_dir('~/financial_data/histdata_feathers/')
    print(candle_dict)
