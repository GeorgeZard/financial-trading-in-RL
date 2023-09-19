from tsrl.utils.experiment_base import BaseExperiment
from pathlib import Path
import torch
import numpy as np
from ray import tune


class TorchExperiment(BaseExperiment):
    def __init__(self, *args, gpu_id=None, **kwargs):
        super(TorchExperiment, self).__init__(*args, **kwargs)
        if 'seed' in kwargs and kwargs['seed'] is not None:
            torch.manual_seed(kwargs['seed'])
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda")
            if torch.cuda.device_count() > 1:
                if gpu_id:
                    self.device = torch.device(f"cuda:{gpu_id}")
                else:
                    self.device = torch.device("cuda")

            print(f"Running {self.exp_path.stem} on {torch.cuda.get_device_name(self.device)}")
        else:
            print(f"Running {self.exp_path.stem} on {self.device}")

    def checkpoint(self, global_step, **kwargs):
        state_dict = dict(model_state_dict=self.model.state_dict(),
                          optimizer_state_dict=self.optimizer.state_dict(),
                          torch_rng_state=torch.random.get_rng_state(),
                          numpy_rng_state=np.random.get_state(),
                          **kwargs)
        torch.save(state_dict, str(self.exp_path / 'exp_state_dict.pkl'))
        if not tune.is_session_enabled():
            pass
            # checkpoint_dir = (self.exp_path / f"checkpoint_{str(global_step)}")
            # checkpoint_dir.mkdir(parents=True, exist_ok=False)
            # torch.save(state_dict, str(Path(checkpoint_dir) / 'exp_state_dict.pkl'))
        else:
            with tune.checkpoint_dir(global_step) as checkpoint_dir:
                torch.save(state_dict, str(Path(checkpoint_dir) / 'exp_state_dict.pkl'))

    def load_checkpoint(self, checkpoint_path, model_class, load_rng_states=True):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'exp_state_dict.pkl'
        exp_state = torch.load(checkpoint_path, map_location=self.device)
        if self.model is None:
            self.model = model_class(**self.db['model_params'])
        self.model.load_state_dict(exp_state['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(exp_state['optimizer_state_dict'])
        if load_rng_states and 'torch_rng_state' in exp_state:
            torch.random.set_rng_state(exp_state['torch_rng_state'])
            np.random.set_state(exp_state['numpy_rng_state'])
        return exp_state
