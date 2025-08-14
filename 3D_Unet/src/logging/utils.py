import json
from types import SimpleNamespace
from abc import ABC, abstractmethod

try:
    import wandb
except:
    print("wandb not installed.")

class BaseLogger(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.hparams = {}
        self.process_config(self.cfg.__dict__)

    def is_json_serializable(self, obj):
        try:
            json.dumps(obj)
            return True
        except TypeError:
            return False

    def process_config(self, cfg, parent_key=''):
        for key, value in cfg.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                self.process_config(value, current_key)
            else:
                if self.is_json_serializable(value):
                    self.hparams[current_key] = value
    
    @abstractmethod
    def log(self, d: dict = {}, commit: bool = True):
        pass

    @abstractmethod
    def finish(self):
        pass

class NoLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def log(self, d: dict = {}, commit: bool = True):
        return
    
    def finish(self, ):
        return

class WandbLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        wandb.init(
            project= cfg.project,
            config= self.hparams,
            )

    def log(self, d: dict = {}, commit: bool = True):
        wandb.log(d, commit=commit)
        return

    def finish(self, ):
        wandb.finish()
        return

def get_logger(cfg):
    if cfg.local_rank == 0:
        return NoLogger(cfg)
    elif cfg.logger == "wandb":
        return WandbLogger(cfg)
    else:
        return NoLogger(cfg)
