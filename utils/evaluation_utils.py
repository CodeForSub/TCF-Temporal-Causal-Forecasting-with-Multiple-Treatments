

import numpy as np
import pandas as pd
import os
import tensorflow
from TCFimt_model import TCFimt_Model

import pickle


def write_results_to_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

def append_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)


def load_trained_model(dataset_test, hyperparams_file, model_name, model_folder, b_decoder_model=False):
    _, length, num_covariates = dataset_test['current_covariates'].shape
    num_treatments = dataset_test['current_treatments'].shape[-1]

    num_outputs = dataset_test['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    model = TCFimt_Model(params, best_hyperparams)
    if (b_decoder_model):
        model = TCFimt_Model(params, best_hyperparams, b_train_decoder=True)

    checkpoint_name = model_name + "_final"
    # ck_tensors = get_tensors_from_checkpoint(model_folder, checkpoint_name)
    model.load_model(model_name=model_name, model_folder=model_folder)

    return model

#compute the treatment accuracy and treatment timing accuracy
def accuracy_evaluation(truth, predictions):
    '''
    truth: N*T*
    '''

    shape=truth.shape
    treatment_acc=0
    treatment_timing_acc=0
    j=0
    k=0
    for i in range(shape[0]):
        matr_1=predictions[i,:,1:3]

        matr_2=truth[i,:,1:3]
        cor_1= np.argmax(matr_1)
        cor_1= divmod(cor_1, 2)
        print("cor_1 is: ",cor_1)
        pre_timing=cor_1[0]
        pre_treatment=cor_1[1]

        cor_2= np.argmax(matr_2)
        cor_2= divmod(cor_2, 2)
        print("cor_2 is: ",cor_2)
        #sums=list(np.sum(matr_2,axis=0))
        #print("sums are: ",sums)
        #xs=list(np.argmax(matr_2,axis=0))
        #print("xs are: ",xs)
        true_treatment=cor_2[1]
        true_timing=cor_2[0]

        '''
        if sums[0] < shape[1]:
            true_treatment=sums.index(1)
            j+=1
            if xs[0] == 0:
                true_timing=max(xs)

            if sums[-1] > 0:
                k+=1
        '''

        if pre_timing==true_timing:
            treatment_timing_acc+=1

        if pre_treatment==true_treatment:
            treatment_acc+=1

        print("pre: ",pre_timing,pre_treatment)
        print("true: ",true_timing,true_treatment)
        #print("the number of numpoints that have treatments are: ",j)
        #print("the number of numpoints that have the last kind of treatment is: ",k)

    return treatment_acc/shape[0], treatment_timing_acc/shape[0]
    

def treat_accuracy_evaluation(seq_truth, seq_predictions):
    '''
    seq_truth: dict, including patient tracing info
    
    seq_predictions: N*T*1
    '''
    
    patient_types = seq_truth['patient_types'] #value only 2 and 3
    patient_ids_all_trajectories = seq_truth['patient_ids_all_trajectories']
    # patient_current_t = seq_truth['patient_current_t']
    
    truth_treatments = seq_truth['current_treatments']  #N*T*
    N, T, _ = seq_predictions.shape
    treatment_acc, treatment_timing_acc = 0, 0
    i = 0
    onehot_treatments = {}
    onehot_treatments[1] = [1, 0, 0, 0]
    onehot_treatments[2] = [0, 1, 0, 0]
    onehot_treatments[3] = [0, 0, 1, 0]

    while True:
        # find all samples belong to patient i
        try:
            i_all_idx = np.argwhere(patient_ids_all_trajectories == i)
            min_pred_value = np.argmin(np.array([seq_predictions[k, ...] for k in i_all_idx]))
        except:
            break
        (pred_idx_row, pred_idx_col, _) = np.unravel_index(min_pred_value, np.array(seq_predictions).shape)
        pred_idx = i_all_idx[pred_idx_row, 0]

        treatment_acc_flag = False
        treatment_timing_acc_flag = False
        # search from all time steps to match the treatment action
        for t in range(T):
            if np.all(onehot_treatments[patient_types[pred_idx]] == truth_treatments[pred_idx, t, ...]):
                treatment_acc_flag = True
                if pred_idx_col == t:
                    treatment_timing_acc_flag = True

        if treatment_acc_flag:
            treatment_acc += 1
        if treatment_timing_acc_flag:
            treatment_timing_acc += 1
        i += 1
    return treatment_acc/i, treatment_timing_acc/i


def corruption(data):
    '''
    mainly change the 'current_treatments' and 'previous_treatments'
    data: dict
    return cor_data: dict
    '''
    cor_data = data.copy()
    current_treatments = data['current_treatments']  # N*T*d
    N, T, D_t = current_treatments.shape
    for i in range(N):
        for j in range(T):
            assigned_t_ind = np.random.choice(D_t)
            real_ind = np.where(current_treatments[i, j, ...])[0][0]
            if assigned_t_ind != real_ind:
                cor_data['current_treatments'][i, j, ...] = np.zeros((1, D_t))
                cor_data['current_treatments'][i, j, assigned_t_ind] = 1
    cor_data['previous_treatments'] = cor_data['current_treatments'][:, :-1, :]
    return cor_data

def get_processed_data(raw_sim_data,
                       scaling_params):
    """
    Create formatted data to train both encoder and seq2seq atchitecture.
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    mean['chemo_application'] = 0
    mean['radio_application'] = 0
    std['chemo_application'] = 1
    std['radio_application'] = 1

    input_means = mean[
        ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
    input_stds = std[['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()

    # Continuous values
    cancer_volume = (raw_sim_data['cancer_volume'] - mean['cancer_volume']) / std['cancer_volume']
    patient_types = (raw_sim_data['patient_types'] - mean['patient_types']) / std['patient_types']

    patient_types = np.stack([patient_types for t in range(cancer_volume.shape[1])], axis=1)

    # Binary application
    chemo_application = raw_sim_data['chemo_application']
    radio_application = raw_sim_data['radio_application']
    sequence_lengths = raw_sim_data['sequence_lengths']

    # Convert treatments to one-hot encoding

    treatments = np.concatenate(
        [chemo_application[:, :-offset, np.newaxis], radio_application[:, :-offset, np.newaxis]], axis=-1)

    one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 0):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 0):
                one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 1):
                one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 1):
                one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

    one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

    current_covariates = np.concatenate(
        [cancer_volume[:, :-offset, np.newaxis], patient_types[:, :-offset, np.newaxis]], axis=-1)
    outputs = cancer_volume[:, horizon:, np.newaxis]

    output_means = mean[['cancer_volume']].values.flatten()[0]  # because we only need scalars here
    output_stds = std[['cancer_volume']].values.flatten()[0]

    print(outputs.shape)

    # Add active entires
    active_entries = np.zeros(outputs.shape)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])
        active_entries[i, :sequence_length, :] = 1

    raw_sim_data['current_covariates'] = current_covariates
    raw_sim_data['previous_treatments'] = one_hot_previous_treatments
    raw_sim_data['current_treatments'] = one_hot_treatments
    raw_sim_data['outputs'] = outputs
    raw_sim_data['active_entries'] = active_entries

    raw_sim_data['unscaled_outputs'] = (outputs * std['cancer_volume'] + mean['cancer_volume'])
    raw_sim_data['input_means'] = input_means
    raw_sim_data['inputs_stds'] = input_stds
    raw_sim_data['output_means'] = output_means
    raw_sim_data['output_stds'] = output_stds

    return raw_sim_data


def get_mse_at_follow_up_time(mean, output, active_entires):
    mses = np.sum(np.sum((mean - output) ** 2 * active_entires, axis=-1), axis=0) \
           / active_entires.sum(axis=0).sum(axis=-1)

    return pd.Series(mses, index=[idx for idx in range(len(mses))])

def get_mae_at_follow_up_time(mean, output, active_entires):
    maes = np.sum(np.sum(np.abs(mean - output) * active_entires, axis=-1), axis=0) \
           / active_entires.sum(axis=0).sum(axis=-1)

    return pd.Series(maes, index=[idx for idx in range(len(maes))])

def train_BR_optimal_model(dataset_train, dataset_val, hyperparams_file, model_name, model_folder,
                           b_decoder_model=False):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    print("Best Hyperparameters")
    print(best_hyperparams)

    if (b_decoder_model):
        print(best_hyperparams)
        model = TCFimt_Model(params, best_hyperparams, b_train_decoder=True)
    else:
        model = TCFimt_Model(params, best_hyperparams)
    model.train(dataset_train, dataset_val, model_name=model_name, model_folder=model_folder)


def get_tensors_from_checkpoint(model_dir, checkpoint_name):
    from tensorflow.python import pywrap_tensorflow
    tensors = {}
    checkpoint_path = os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name))
    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map:
        print('tensor_name:', key)
        tensors[key] = reader.get_tensor(key)
    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    # print_tensors_in_checkpoint_file(file_name,  # ckpt文件名字
    #                                 None,  # 如果为None,则默认为ckpt里的所有变量
    #                                 all_tensors=True,  # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
    #                                 all_tensor_names=True)  # bool 是否打印所有的tensor的name
    return tensors
