

import tensorflow as tf
import logging
import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from datetime import datetime
from utils.evaluation_utils import get_tensors_from_checkpoint

from TCFimt_model import TCFimt_Model
from utils.evaluation_utils import write_results_to_file, load_trained_model, get_processed_data, corruption




def fit_TCFimt_encoder(dataset_train, tilde_dataset_train, dataset_val, tilde_dataset_val, model_name, model_dir, hyperparams_file,
                    b_hyperparam_opt, sim_num=200):
    # dataset_train, notice_train = dataset_train_tp
    # dataset_val, notice_val = dataset_val_tp

    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]
    num_inputs = dataset_train['current_covariates'].shape[-1] + dataset_train['current_treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 1000}

    hyperparams = dict()
    num_simulations = sim_num#1  #50
    best_validation_mse = 1000000

    if b_hyperparam_opt:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['br_size'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['fc_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * (hyperparams['br_size']))
            hyperparams['learning_rate'] = np.random.choice([0.1, 0.01, 0.001, 0.0001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256, 512]) #65, 128, 256
            hyperparams['rnn_keep_prob'] = np.random.choice([0.3, 0.4, 0.5, 0.6])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = TCFimt_Model(params, hyperparams)
            model.train(dataset_train, tilde_dataset_train, dataset_val, tilde_dataset_val, model_name, model_dir)
            validation_mse, _, predictions = model.evaluate_predictions(dataset_val)

            if (validation_mse < best_validation_mse):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_mse, validation_mse))
                best_validation_mse = validation_mse
                best_hyperparams = hyperparams.copy()

                checkpoint_name = model_name + "_final"
                model.save_network(model_dir, checkpoint_name)
            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        logging.info("Using default hyperparameters")
        best_hyperparams = {
            'rnn_hidden_units': 24,
            'br_size': 12,
            'fc_hidden_units': 36,
            'learning_rate': 0.01,
            'batch_size': 128,
            'rnn_keep_prob': 0.9}
        logging.info("Best hyperparams: \n {}".format(best_hyperparams))
        write_results_to_file(hyperparams_file, best_hyperparams)

    # model = TCFimt_Model(params, best_hyperparams)
    # model.train(dataset_train, tilde_dataset_train, dataset_val, tilde_dataset_val, model_name, model_dir)
    return model

def test_TCFimt_encoder(pickle_map, models_dir,
                     encoder_model_name, encoder_hyperparams_file,
                     b_encoder_hyperparm_tuning, encoder_simulation_num):

    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']



    scaling_data = pickle_map['scaling_data']

    training_processed = get_processed_data(training_data, scaling_data)
    validation_processed = get_processed_data(validation_data, scaling_data)
    tilde_training_processed = corruption(training_processed)
    tilde_validation_processed = corruption(validation_processed)

    test_processed = get_processed_data(test_data, scaling_data)
    # pd.DataFrame(training_processed).to_csv("training_processed.csv")
    pickle.dump(training_processed, open("training_processed.p", 'wb'))
    TCFimt_encoder = fit_TCFimt_encoder(dataset_train=training_processed, tilde_dataset_train=tilde_training_processed, dataset_val=validation_processed,
                    tilde_dataset_val=tilde_validation_processed,
                    model_name=encoder_model_name, model_dir=models_dir,
                    hyperparams_file=encoder_hyperparams_file, b_hyperparam_opt=b_encoder_hyperparm_tuning, sim_num=encoder_simulation_num)

    # checkpoint_name = encoder_model_name + "_final"
    # ck_tensors = get_tensors_from_checkpoint(models_dir, checkpoint_name)

    # TCFimt_encoder = load_trained_model(validation_processed, encoder_hyperparams_file, encoder_model_name, models_dir)
    mean_mse, mae, _ = TCFimt_encoder.evaluate_predictions(test_processed)

    # rmse = np.sqrt(mean_mse)

    rmse = (np.sqrt(mean_mse)) / 1150 * 100  # Max tumour volume = 1150
    mae = mae / 1150 * 100
    return rmse, mae, TCFimt_encoder
