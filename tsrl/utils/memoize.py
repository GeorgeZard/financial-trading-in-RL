"""
This module contains the memoize decorator, which can be used to memoize functions on disk to avoid
constant recalculation of steps such as preprocessing in cases where data are constantly changing.
"""
from pathlib import Path
from datetime import datetime
import time
# import pickle
import dataset
import uuid
from types import FunctionType
from copy import deepcopy
import gc
import pandas as pd
import functools
import dill as pickle

import hashlib
try:
    import pandas as pd
except ImportError:
    pass


def normalize_date(dt, td, side='left'):
    td = pd.to_timedelta(td)
    if side == 'right':
        dt.ceil(td)
    else:
        dt.floor(td)
    return dt


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def measure(self):
        return time.time() - self.start


def parse_val(v):
    if type(v) is dict:
        return dict_to_tuple(v)
    elif isinstance(v, FunctionType):
        return v.__name___
    elif isinstance(v, pd.DataFrame):
        val_hash = hashlib.sha256(pd.util.hash_pandas_object(v, index=True).values).hexdigest()
        cols = '_'.join(sorted(v.columns.tolist()))
        return hashlib.sha256((val_hash + cols).encode('utf-8')).hexdigest()
    else:
        return v


def dict_to_tuple(d):
    if type(d) is dict:
        keys = list(d.keys())
        vals = []
        for v in d.values():
            vals.append(parse_val(v))
        return tuple(sorted(list(zip(keys, vals))))
    elif type(d) in [list, tuple]:
        vals = []
        for v in d:
            vals.append(parse_val(v))
        return tuple(sorted(vals))
    return tuple()


def memoize(extra_id=None, ignore_kwargs=tuple()):
    """

    :param extra_id: An extra id to use for avoiding using a memoized function when something changes in its
                     "dirty" state. (i.e. the data on disk that are loaded by a preprocessor are changed.)
    :param ignore_kwargs:
    :return:
    """
    class SqliteMemoizer:
        def __init__(self, func):
            self.func = func
            # self.__call__ = functools.wraps(self.func)(self.__call__)
            self.path = Path('~/saved_models_experiments/memoized').expanduser()
            self.path.mkdir(exist_ok=True, parents=True)
            self.pickle_path = self.path / 'pickles'
            self.pickle_path.mkdir(exist_ok=True)

        def __call__(self, *args, **kwargs):
            cpy_kwargs = deepcopy(kwargs)

            # check if caching should not occur
            if kwargs.get('no_cache', False):
                kwargs.pop('no_cache')
                return self.func(*args, **kwargs)

            # create hashing key
            key = dict_to_tuple(args)
            for k in list(cpy_kwargs.keys()):
                if k in ignore_kwargs:
                    cpy_kwargs.pop(k)
            key += ('sentinel',) + dict_to_tuple(cpy_kwargs)
            if extra_id is not None:
                key += ('id_sentinel', extra_id)
            key = str(key)
            db_path = "sqlite:///" + str(self.path / 'memoize.db')
            db = dataset.connect(db_path)
            table = db[f"{self.func.__name__}"]
            res = table.find_one(arg_key=key)
            r = None
            # print('\n'.join([str((q['arg_key'], q['file'])) for q in list(table.all())]), '\n', key)
            # print(res)
            if res:
                # print(f"Found cached version for {self.func.__name__}")
                pickle_file = Path(res['file'])
                if pickle_file.exists():
                    with open(pickle_file, 'rb') as f:
                        r = pickle.load(f)
            del db, table
            gc.collect()
            if r is None:
                r = self.func(*args, **kwargs)
                if type(r).__name__ not in ['generator']:
                    filename = str(self.pickle_path / (self.func.__name__ + '_' + str(uuid.uuid4().hex)))
                    with open(filename, 'wb') as f:
                        pickle.dump(r, f)
                    db = dataset.connect(db_path)
                    table = db[f"{self.func.__name__}"]
                    table.upsert(dict(arg_key=key, file=filename), ['arg_key'])
                    del db, table
            return r

        def clear_cache(self):
            db_path = "sqlite:///" + str(self.path / 'memoize.db')
            db = dataset.connect(db_path)
            table = db[f"{self.func.__name__}"]
            res = list(table.all())
            for r in res:
                p = Path(r['file'])
                if p.exists():
                    p.unlink()
                table.delete(file=r['file'])

    return SqliteMemoizer
