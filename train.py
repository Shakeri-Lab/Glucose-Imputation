import torch
import json
import numpy as np
import argparse, os, time
from pypots.imputation import *
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed
from utils.model_configs import *
from utils.utils import load_data, skew_norm, unskew_norm, parse_args_from_yml, visualize_imputation, eval_metrics
from utils.risk_calc import risk_calc

SUPPORT_MODELS = {
    "Autoformer": Autoformer,
    "BRITS": BRITS,
    "Crossformer": Crossformer,
    "CSDI": CSDI,
    "DLinear": DLinear,
    "ETSformer": ETSformer,
    "FiLM": FiLM,
    "FreTS": FreTS,
    "GPVAE": GPVAE,
    "GRUD": GRUD,
    "Informer": Informer,
    "iTransformer": iTransformer,
    "Koopa": Koopa,
    "MICN": MICN,
    "MRNN": MRNN,
    "NonstationaryTransformer": NonstationaryTransformer,
    "PatchTST": PatchTST,
    "Pyraformer": Pyraformer,
    "SAITS": SAITS,
    "SCINet": SCINet,
    "StemGNN": StemGNN,
    "TimesNet": TimesNet,
    "Transformer": Transformer,
    "USGAN": USGAN,

    "TimeMixer": TimeMixer,
    "TimeMixerPP": TimeMixerPP,
    "ModernTCN": ModernTCN,
    "TSLANet": TSLANet,
    "TEFN": TEFN,
    "TOTEM": TOTEM,
    "GPT4TS": GPT4TS,
    "Lerp": Lerp,
    "LOCF": LOCF,
    "Median": Median, 
    "Mean": Mean,
}

def parse_args():
    """ Parse arguments and load config file. """
    parser = argparse.ArgumentParser(description="Run Benchmark.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML config file")
    return parse_args_from_yml(parser.parse_args().config_path)

def init_hyperparameters(args, suggest_hp):
    """ Init hyperparameters. """
    round_saving_path = os.path.join(args.saving_path, args.model)
    args.plot_dir = os.path.join(round_saving_path, 'plots')
    if not os.path.exists(round_saving_path): os.makedirs(round_saving_path)

    hyperparameters = init_configs[args.model].copy()
    
    if suggest_hp:
        suggest_dict = vars(suggest_hp)
        hyperparameters.update(suggest_dict)

    if "lr" in hyperparameters:
        lr = hyperparameters.pop("lr")
    else:
        lr = 0.001 

    hyperparameters["device"] = args.device
    hyperparameters["saving_path"] = round_saving_path
    hyperparameters["model_saving_strategy"] = "best"
    
    if args.model == "USGAN":
        hyperparameters["G_optimizer"] = Adam(lr=lr)
        hyperparameters["D_optimizer"] = Adam(lr=lr)
    else:
        hyperparameters["optimizer"] = Adam(lr=lr)

    return hyperparameters

def init_training(args, suggest_hp):
    """ Init hyperparameters, model and dataset. """

    if args.model not in ['Lerp', 'Mean', 'Median', 'LOCF']:
        hyperparameters = init_hyperparameters(args, suggest_hp)
        model = SUPPORT_MODELS[args.model](**hyperparameters)
    else:
        round_saving_path = os.path.join(args.saving_path, args.model)
        args.plot_dir = os.path.join(round_saving_path, 'plots')
        if not os.path.exists(round_saving_path): os.makedirs(round_saving_path)
        
        hyperparameters = {}
        model = SUPPORT_MODELS[args.model]()
        
    dataset_dict = load_data(args)
    dataset_dict = data_dict_prep(dataset_dict)
    return model, dataset_dict, hyperparameters

def data_dict_prep(dataset_dict):
    """ Create Dictionaries of missed and original dataset. """
    dict_list = []
    for splt in ['train', 'val', 'test']:
        dict_list.append({'X': dataset_dict['missed'][splt], 'X_ori': dataset_dict['org'][splt]})    
    return dict_list
    
def test_evaluate(test_set_imputation, noNan_test_set, test_indicating_mask):
    """ Calculate model performance. """
    scores = []
    for mtd in [calc_mae, calc_mse, calc_mre]:
        scores.append(mtd(test_set_imputation, noNan_test_set, test_indicating_mask))
    return scores
    

def seg_evaluate(args, test_set_imputation, noNan_test_set, test_indicating_mask):
    """ Calculate model performance. """
    save_dir = os.path.join(args.saving_path, args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    mask = test_indicating_mask.astype(bool)
    tr_ts = noNan_test_set[mask]
    pr_ts = test_set_imputation[mask]

    result_name = args.miss_config['type'] + '_' + str(args.miss_config['protocol_mask_ratio'] if args.miss_config['type'] == 'A' 
                                                       else args.miss_config['num_meal_hide'] if args.miss_config['type'] == 'B'
                                                       else '')    
    np.save(os.path.join(save_dir, result_name + '_result_prediction.npy'), np.stack([noNan_test_set, test_set_imputation, mask], axis=-1))
    with open(os.path.join(save_dir, f'{result_name}_metrics.json'), 'w') as f:
        json.dump(eval_metrics(noNan_test_set, test_set_imputation, mask), f, indent=4)

    risk_calc(tr_ts, pr_ts, os.path.join(save_dir, result_name), args.seg_path)


def test_calc(args, dataset_dict, test_set_imputation):
    """ Auxilary def to handle performance calc. """
    ground_truth_data, corrupted_data = dataset_dict[2]['X_ori'][:, :, 0], dataset_dict[2]['X'][:, :, 0]
    
    if hasattr(ground_truth_data, 'cpu'): ground_truth_data = ground_truth_data.cpu().numpy()
    if hasattr(corrupted_data, 'cpu'): corrupted_data = corrupted_data.cpu().numpy()

    test_indicating_mask = (~np.isnan(ground_truth_data) & np.isnan(corrupted_data))
    noNan_test_set = unskew_norm(np.nan_to_num(ground_truth_data), 1, 40, 400)
    
    if args.is_evaluate:
        seg_evaluate(args, test_set_imputation, noNan_test_set, test_indicating_mask)
    
    mae, mse, mre = test_evaluate(test_set_imputation, noNan_test_set, test_indicating_mask)         
    logger.info(f"Result: MAE: {mae}, MSE: {mse}, MRE: {mre}")

    result_dir = os.path.join(args.plot_dir, args.miss_config['type'] + '_' + str(args.miss_config['protocol_mask_ratio'] if args.miss_config['type'] == 'A'
                                                                                  else args.miss_config['num_meal_hide'] if args.miss_config['type'] == 'B' 
                                                                                  else '') + '_plots')
    visualize_imputation(unskew_norm(corrupted_data, 1, 40, 400), unskew_norm(ground_truth_data, 1, 40, 400), test_set_imputation, test_indicating_mask, s_idx=0, f_idx=0, title="Imp_Quality", plot_dir=result_dir)
        

def engine(args, suggest_hp):
    """ Train, and test model. """
    try:
        model, dataset_dict, hyperparameters = init_training(args, suggest_hp)   
        
        if args.model not in ['Lerp', 'Mean', 'Median', 'LOCF']:
            if args.is_evaluate:
                model.load(os.path.join(args.saving_path, args.model, 'best_model.pypots'))
            if not args.is_evaluate:
                model.fit(train_set={"X": dataset_dict[0]['X']}, val_set=dataset_dict[1])
            if args.train_best: model.save(os.path.join(args.saving_path, args.model, 'best_model'))

        if args.model in ["CSDI", "GPVAE"]:
            results = model.predict({"X": dataset_dict[2]['X']}, n_sampling_times=10)
            test_set_imputation = unskew_norm(results["imputation"][:, :, :, 0].mean(axis=1), 1, 40, 400)
        else:
            results = model.predict({"X": dataset_dict[2]['X']})
            test_set_imputation = unskew_norm(results["imputation"][:, :, 0], 1, 40, 400)
    
        test_calc(args, dataset_dict, test_set_imputation)

        final_loss = getattr(model, 'best_loss', float('inf'))
        return final_loss, model
    
    except Exception as e:
        print(f"Error in engine execution: {e}")
        return float('inf'), None


# if __name__ == "__main__":
#     args = parse_args()
#     set_random_seed(7)
#     engine(args, suggest_hp=None)
