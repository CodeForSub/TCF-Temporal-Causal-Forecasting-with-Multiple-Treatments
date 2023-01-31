

import tensorflow
import os
import argparse
import logging
import time
import sys
import codecs

from TCFimt_encoder_evaluate import test_TCFimt_encoder
from TCFimt_decoder_evaluate import test_TCFimt_decoder
from utils.cancer_simulation import get_cancer_sim_data
from utils.evaluation_utils import get_tensors_from_checkpoint

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=5, type=int)
    parser.add_argument("--radio_coeff", default=5, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="demo_for_TCFimt")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=True, type=bool)  #False
    parser.add_argument("--b_decoder_hyperparm_tuning", default=True, type=bool)  #False
    parser.add_argument("--encoder_num_simulation", default=200, type=int) #200
    parser.add_argument("--decoder_num_simulation", default=200, type=int) #200
    parser.add_argument("--num_patient", default=10000, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--horizon", default=5, type=int)
    parser.add_argument("--only_decoder", default=False, type=bool)
    return parser.parse_args()


if __name__ == '__main__':

    start_time=time.time()
    
    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # use tensorflow-gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    pickle_map = get_cancer_sim_data(num_p=args.num_patient, chemo_coeff=args.chemo_coeff, radio_coeff=args.radio_coeff, b_load=False,
                                          b_save=True, model_root=args.results_dir)



    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)


    models_dir = '{}/TCFimt_models_{}'.format(args.results_dir,args.model_name)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # encoder
    # check checkpoint variables
    # checkpoint_name = "encoder_" + args.model_name + "_final"
    # ck_tensors = get_tensors_from_checkpoint(models_dir, checkpoint_name)
    if args.only_decoder:
        f = open(args.model_name + '_result2.txt', 'w+')
    else:
        f = open(args.model_name + '_result.txt', 'w+')
    if not args.only_decoder:
        rmse_encoder, mae_encoder, encoder_model = test_TCFimt_encoder(pickle_map=pickle_map, models_dir=models_dir,
                                        encoder_model_name=encoder_model_name,
                                        encoder_hyperparams_file=encoder_hyperparams_file,
                                        b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning,
                                        encoder_simulation_num=args.encoder_num_simulation)
        #


        f.write("RMSE for one-step-ahead prediction: \n")
        f.write(str(mae_encoder) + '\n')

        print("RMSE for one-step-ahead prediction.")
        print(rmse_encoder, mae_encoder)

    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    """
    The counterfactual test data for a sequence of treatments in the future was simulated for a 
    projection horizon of 5 timesteps. 
   
    """

    # max_projection_horizon = args.horizon #5
    # projection_horizon = args.horizon #5 prediction step
    logging.info("Chemo coeff {} | Radio coeff {}".format(args.chemo_coeff, args.radio_coeff))
    for pred_horizon in range(3, 5):
        max_projection_horizon = pred_horizon # 5
        projection_horizon =pred_horizon # 5  prediction step
    # decoder
        rmse_decoder, mae_decoder, decoder_predictions, treatment_acc, treatment_timing_acc = test_TCFimt_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                        projection_horizon=projection_horizon,
                                        models_dir=models_dir,
                                        encoder_model_name=encoder_model_name,
                                        encoder_hyperparams_file=encoder_hyperparams_file,
                                        decoder_model_name=decoder_model_name,
                                        decoder_hyperparams_file=decoder_hyperparams_file,
                                         b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning,
                                        decoder_simulation_num=args.decoder_num_simulation)
    
        f.write("{}-step-ahead prediction: \n".format(pred_horizon))
        f.write(str(mae_decoder)+'\n')

    f.close()

    used_time=time.time()-start_time
    print(str(used_time/60)+"min")