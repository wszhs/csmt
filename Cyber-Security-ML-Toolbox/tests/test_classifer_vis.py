
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
from csmt.singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from csmt.singleVis.SingleVisualizationModel import VisModel
from csmt.singleVis.losses import UmapLoss, ReconstructionLoss, SingleVisLoss
from csmt.singleVis.edge_dataset import DataHandler
from csmt.singleVis.trainer import SingleVisTrainer
from csmt.singleVis.data import NormalDataProvider
from csmt.singleVis.spatial_edge_constructor import SingleEpochSpatialEdgeConstructor
from csmt.singleVis.projector import DVIProjector
from csmt.singleVis.eval.evaluator import Evaluator
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
import numpy as np
import json
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import random
from csmt.figure import CFigure
from math import ceil
from csmt.singleVis.visualizer import visualizer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)

def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
    

def plot_headmap(X,a_score,model_name):
    X_x=X[:,0]
    X_y=X[:,1]
    plt.scatter(X_x, X_y, marker='o', c=a_score, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(model_name)
    plt.show()

if __name__=='__main__':
     
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name=options.datasets
     orig_models_name=options.algorithms

     X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

     trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)
    
     # trained_models=models_load(datasets_name,orig_models_name)
     y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
     
    #  plot_headmap(X_test,y_pred[0][:,0],orig_models_name)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
     
     ENCODER_DIMS=[512,256,256,256,256,2]
     DECODER_DIMS= [2,256,256,256,256,512]
     EPOCH_START=1
     EPOCH_END=3
     EPOCH_PERIOD=1
     DEVICE=torch.device('cpu')
     VIS_MODEL_NAME='vis'
     LAMBDA1=0.1
     CONTENT_PATH=""
     EPOCH_NAME=""
     CLASSES=""
     net=trained_models
     I=2
     S_N_EPOCHS=1
     B_N_EPOCHS=2
     N_NEIGHBORS=2
     PATIENT=1
     MAX_EPOCH=2
     save_dir='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
     DATASET='mnist'
     VIS_METHOD='xx'
     
     # Define data_provider
     data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, classes=CLASSES, epoch_name=EPOCH_NAME, verbose=1)
    
     # Define visualization models
     model = VisModel(ENCODER_DIMS, DECODER_DIMS)

    # Define Losses
     negative_sample_rate = 5
     min_dist = .1
     _a, _b = find_ab_params(1.0, min_dist)
     umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
     recon_loss_fn = ReconstructionLoss(beta=1.0)
     # Define Projector
     projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
     # Define DVI Loss
     criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)

    # Define training parameters
     optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
    # Define Edge dataset
     t0 = time.time()
     spatial_cons = SingleEpochSpatialEdgeConstructor(data_provider, I, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS)
     edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()
     t1 = time.time()

     probs = probs / (probs.max()+1e-3)
     eliminate_zeros = probs>1e-2#1e-3
     edge_to = edge_to[eliminate_zeros]
     edge_from = edge_from[eliminate_zeros]
     probs = probs[eliminate_zeros]
     
     dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)

     n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
    # chose sampler based on the number of dataset
     if len(edge_to) > pow(2,24):
        sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
     else:
        sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
     edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

    ########################################################################################################################
    #                                                       TRAIN                                                          #
    ########################################################################################################################

     trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)

     t2=time.time()
     trainer.train(PATIENT, MAX_EPOCH)
     t3 = time.time()

    # save time result
     save_dir = data_provider.model_path
     file_name = "time_{}".format(VIS_MODEL_NAME)
     save_file = os.path.join(save_dir, file_name+".json")
     if not os.path.exists(save_file):
        evaluation = dict()
     else:
        f = open(save_file, "r")
        evaluation = json.load(f)
        f.close()
     if "complex_construction" not in evaluation.keys():
        evaluation["complex_construction"] = dict()
     evaluation["complex_construction"][str(I)] = round(t1-t0, 3)
     if "training" not in evaluation.keys():
        evaluation["training"] = dict()
     evaluation["training"][str(I)] = round(t3-t2, 3)
     with open(save_file, 'w') as f:
        json.dump(evaluation, f)

     save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(I))
     trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

     vis = visualizer(data_provider, projector, 200, "tab10")
     vis.save_fig(I, path=os.path.join(save_dir, "{}_{}_{}.png".format(DATASET, I, VIS_METHOD)))



     
    
 









