import optuna
import torch.nn as nn
import torch, os, json, argparse
from train import engine

class HyperParameter:
    """
    This is the general class for hyperparamrter tuning using Optuna.
    """
    def __init__(self, args, n_trials, store_best_dir, config):
        self.args = args
        self.n_trials = n_trials
        self.store_best_dir = store_best_dir
        self.args_config = config

    def objective(self, trial):
        """ Objective for the hyperparameters. """
        args_dict = vars(self.args).copy()
        def sample_param(name, space):
            """ Sampling. """
            vmin, vmax, vstep = space["min"], space["max"], space["step"]
    
            if isinstance(vstep, int):
                return trial.suggest_int(name, vmin, vmax, step=vstep)
            if isinstance(vstep, float):
                return trial.suggest_float(name, vmin, vmax, step=vstep)
            if isinstance(vstep, str) and vstep == "log":
                return trial.suggest_float(name, vmin, vmax, log=True)
    
            raise ValueError(f"Unsupported step type for {name}: {vstep}")
    
        for k, v in args_dict.items():
            if not isinstance(v, (dict, list)): continue
            if isinstance(v, list):
                args_dict[k] = [
                    sample_param(f"{k}_{idx}", space)
                    for idx, space in enumerate(v)
                ]
                continue
            args_dict[k] = sample_param(k, v)
    
        suggest_hp = argparse.Namespace(**args_dict)
        loss, model = engine(self.args_config, suggest_hp)
        return loss

    def hyperparameter_tuning(self):
        """
        Find the best config among different trials
        """
        os.makedirs(self.store_best_dir, exist_ok=True)        
        db_path = os.path.join(self.store_best_dir, 'optuna.db')
        
        study = optuna.create_study(study_name='imputation', storage= f"sqlite:///{db_path}", load_if_exists=True, direction="minimize") # create study
        study.optimize(self.objective, n_trials=self.n_trials - len(study.trials)) # start optimizing
        best_trial = study.best_trial # select the best hyperparameters
        print(f"Best loss: {best_trial.value:.6f}")
        
        args_dict = vars(self.args).copy() #namespace to args
        for k, v in best_trial.params.items(): args_dict[k] = v # init args with the best hyperparameter.
        
        os.makedirs(self.store_best_dir, exist_ok=True)
        with open(os.path.join(self.store_best_dir, 'best_hyperparameter.json'), 'w') as f: json.dump(args_dict, f, indent=4) # indent for pretty-printing

