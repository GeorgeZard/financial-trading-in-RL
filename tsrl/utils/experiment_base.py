from pathlib import Path
import numpy as np
import logging
import random
from .persistent_dict import PersistentDict


def sorted_dict_string(d):
    return ",".join([f"{k}={v}" for k, v in sorted(d.items())])


class BaseExperiment:
    """
    The BaseExperiment class is desinged to give you a base for building your saved_models_experiments off of.

    The basic parameter you must provide to the class initializer is the experiment path. You can utilize
    the path as you wish within your experiment by accessing the BaseExperiment.exp_path attribute.
    Ideperndently the BaseExperiment object saves all keyword arguements given to it in a persistent pickle
    dictionary saved within the path. You can modify this dict by accessing the BaseExperiment.db attribute.


    """

    def __init__(self, exp_path, **kwargs):
        self.exp_path = Path(exp_path).expanduser()
        self.logger = kwargs.get('logger', logging.getLogger(kwargs.get('logger_name', self.exp_path.stem)))
        if 'seed' in kwargs:
            random.seed(kwargs['seed'])
            np.random.seed(kwargs['seed'])

        db_obj_path = self.exp_path / "experiment_params.pkl"
        db_exists = db_obj_path.exists()
        self.db = PersistentDict(str(db_obj_path))
        if not db_exists:
            print(f"Path {self.exp_path} does not exist. Creating it now.")
            self.exp_path.mkdir(parents=True, exist_ok=True)
            self.db["class_name"] = type(self).__name__

    def save_attrs(self, attr_keys):
        for k in attr_keys:
            self.db[k] = getattr(self, k)

    # def backtest(self, no_cache=False, **kwargs):
    #     if no_cache:
    #         return None
    #     key = f'backtest_{sorted_dict_string(kwargs)}'
    #     res = self.db.get(key, None)
    #     return res
    #
    # def save_backtest(self, bt, **kwargs):
    #     key = f'backtest_{sorted_dict_string(kwargs)}'
    #     self.db[key] = bt
