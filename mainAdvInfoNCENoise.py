import os

import model
import optimizer
from dataset import dataloader

os.environ["MKL_NUM_THREADS"] = "10"
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['OMP_NUM_THREADS'] = '10'

from pprint import pprint

from tools import world, procedure, utils
from tools.world import cprint
import torch
from tensorboardX import SummaryWriter
import time
from os.path import join
import nni
from tools.logger import CompleteLogger

if not "NNI_PLATFORM" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]
else:
    optimized_params = nni.get_next_parameter()
    world.config.update(optimized_params)

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
cprint("ValidTopks" + str(world.valid_topks))

file_path = os.path.join("./normal_data", world.config["datapath"])
dataroot = os.path.join(file_path, world.config["dataset"])
logroot = os.path.join("./log", world.config["dataset"])

dataset = dataloader.Loader(path=dataroot)

MODELS = {
    "mf": model.model_MF.MFModel,
    "lgn": model.model_LightGCN.LightGCNModel,
    "XSimGCL": model.model_XSimGCL.XSimGCLModel

}
LOSSES = {
    "advInfoNCE": optimizer.optim_AdvInfoNCE.AdvInfoNCEOptimizer
}

if world.config["loss"] == "bpr" or world.config["loss"] == "bce" or world.config["loss"] == "rmse":
    world.config["norm_emb"] = 0
    world.config["num_negative_items"] = 1

Recmodel = MODELS[world.model_name](config=world.config, num_users=dataset.n_users, num_items=dataset.m_item,
                                    Graph=dataset.Graph).cuda()
loss_func = LOSSES[world.config["loss"]](model=Recmodel, config=world.config)


if "NNI_PLATFORM" in os.environ:
    save_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard")
    print("save_dir: ", save_dir)
    w: SummaryWriter = SummaryWriter(save_dir)
else:
    save_dir = join(
        logroot,
        time.strftime("%m-%d-%Hh%Mm%Ss-")
        + "-"
        + world.comment
        + "-"
        + world.config["loss"]
        + "-"
        + world.config["model"],
    )
    i = 0
    while os.path.exists(save_dir):
        new_save_dir = save_dir + str(i)
        i += 1
        save_dir = new_save_dir
    w: SummaryWriter = SummaryWriter(save_dir)
    logger = CompleteLogger(root=save_dir)

print("User_num: ", dataset.n_users)
print("Item_num: ", dataset.m_items)
print("Interaction_num: ", dataset.traindataSize + dataset.testDataSize + dataset.validDataSize)
print("===========config================")
pprint(world.config)
print("Comment:", world.comment)
print("Test Topks:", world.topks)
print("===========end===================")

w.add_text("config", str(world.config), 0)

if world.config["resume_dir"] != "":
    Recmodel.load_state_dict(torch.load(world.config["resume_dir"]))
    print("load model from: ", world.config["resume_dir"])

best_recall, best_ndcg, best_hit, best_precision = (
    [0] * len(world.topks),
    [0] * len(world.topks),
    [0] * len(world.topks),
    [0] * len(world.topks),
)
patience = 0
start_total = time.time()

best_Recmodel = MODELS[world.model_name](config=world.config, num_users=dataset.n_users, num_items=dataset.m_item,
                                         Graph=dataset.Graph).cuda()

current_eta = 0

for epoch in range(world.TRAIN_epochs):
    start = time.time()
    if epoch % 5 == 0 and epoch != 0:

        valid_res = procedure.Valid(dataset, Recmodel, epoch, w, world.config["multicore"])
        valid_recall, valid_ndcg, valid_hit, valid_precision = (
            valid_res["recall"],
            valid_res["ndcg"],
            valid_res["hitratio"],
            valid_res["precision"],
        )
        if "NNI_PLATFORM" in os.environ:
            metric = {
                "default": valid_ndcg[0],
                "recall": valid_recall[0],
                "hit": valid_hit[0],
                "precision": valid_precision[0],
            }
            nni.report_intermediate_result(metric)

        if valid_ndcg[0] > best_ndcg[0] + 0.0001:
            patience = 0
            if "NNI_PLATFORM" not in os.environ:
                loss_func.save(os.path.join(save_dir, "best_model.pth"))
            best_Recmodel.load_state_dict(Recmodel.state_dict())
        else:
            patience += 1
            print("Patience: {}/5".format(patience))
            if patience >= 5:
                print("Early stop!")
                cprint('[Test]')
                test_res = procedure.Test(dataset, best_Recmodel, epoch, w, world.config["multicore"])
                print(test_res)
                break

        for i in range(len(world.valid_topks)):
            best_recall[i], best_ndcg[i], best_hit[i], best_precision[i] = (
                max(best_recall[i], valid_recall[i]),
                max(best_ndcg[i], valid_ndcg[i]),
                max(best_hit[i], valid_hit[i]),
                max(best_precision[i], valid_precision[i]),
            )
        print(valid_res)
    
    if (epoch + 3) % world.config['adv_interval'] == 0 and current_eta < world.config['eta_epochs'] and epoch > 0:
        procedure.TrainAdvInfoNCE(
            dataset = dataset, 
            recommend_model = Recmodel, 
            loss_class = loss_func, 
            epoch = epoch, 
            config = world.config, 
            w=w,
            adv_training_flag = True
        )
        current_eta += 1
    output_information = procedure.NoiseTrainAdvInfoNCE(
        dataset = dataset, 
        recommend_model = Recmodel, 
        loss_class = loss_func, 
        epoch = epoch, 
        config = world.config, 
        w=w,
        adv_training_flag = False
    )
    print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}")

print(
    "Best_hit: {}, Best_ndcg: {}, Best_precision: {}, Best_recall: {}".format(
        best_hit, best_ndcg, best_precision, best_recall
    )
)
if "NNI_PLATFORM" in os.environ:
    metric = {"default": best_ndcg[0], "recall": best_recall[0], "hit": best_hit[0], "precision": best_precision[0]}
    nni.report_final_result(metric)

print("Total time:{}".format(time.time() - start_total))
w.close()
