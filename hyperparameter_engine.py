import argparse, json
from utils.utils import parse_args_from_yml, set_seed
from hyperparameter_optuna import HyperParameter
from pypots.utils.random import set_random_seed
from train import engine
import os 
# python hyperparameter_engine.py --model_name SAITS --ParamRangeDir param_range.json --config-path config.yml --NTrials 3 --is_evaluate
def parse_args():
    """
    read config file and extract args.
    """
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--ParamRangeDir', type=str, required=True, help='Path to the param_range.json file')
    parser.add_argument('--config-path', type=str, required=True, help='Path to the general training config file. ')
    parser.add_argument('--NTrials', type=int, required=True, help='Number of trials HPO can examine')
    parser.add_argument('--train_best', action='store_true', help='Train the best configurations')
    parser.add_argument('--is_evaluate', action='store_true', help='Test best model based on differenct configs. ')

    parsed_vars = parser.parse_args()
    train_best = parsed_vars.train_best
    is_evaluate = parsed_vars.is_evaluate


    if is_evaluate:
        set_random_seed(7)
        set_seed(seed=7)
    
    config = parse_args_from_yml(parsed_vars.config_path)
    config.model = parsed_vars.model_name
    config.train_best = train_best
    config.is_evaluate = is_evaluate

    test_param_dir = os.path.join(config.saving_path, config.model, 'best_hyperparameter.json') if train_best or is_evaluate else None
    
    return parse_args_from_json(test_param_dir if (train_best or is_evaluate) and (config.model not in ['Lerp', 'Mean', 'Median', 'LOCF']) else parsed_vars.ParamRangeDir, parsed_vars.model_name, train_best, is_evaluate), config, parsed_vars.NTrials, train_best, is_evaluate

def parse_args_from_json(ParamRangeDir, model_name, train_best, is_evaluate):    
    """ Read json file. """
    with open(ParamRangeDir, 'r') as f:
        if train_best or is_evaluate:
            config_dict = json.load(f)
        else:
            config_dict = json.load(f)[model_name]
            
    return argparse.Namespace(**config_dict)

if __name__ == '__main__':
    args, config, n_trials, train_best, is_evaluate = parse_args()
    store_best_dir = os.path.join(config.saving_path, config.model)
    if train_best or is_evaluate:
        engine(config, args)
    else:
        HyperParameter(args, n_trials, store_best_dir, config).hyperparameter_tuning()
