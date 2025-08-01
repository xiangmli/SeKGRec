import torch.nn.functional as F
import world
import utils
from world import cprint
from parse import parse_args
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from sparse_moe import RecMoE
from moe_config import MoEConfig
import json
from datetime import datetime
import os

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

import register
from register import dataset
import logging
import os
from datetime import datetime


class CustomFilter(logging.Filter):
    def filter(self, record):
        unwanted_keywords = [
            "Summary name",
            "is illegal",
            "using",
            "instead"
        ]
        return not any(keyword in record.getMessage() for keyword in unwanted_keywords)


def setup_logging(dataset_name):
    log_dir = f"../logs/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    custom_filter = CustomFilter()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    for handler in logging.getLogger().handlers:
        handler.addFilter(custom_filter)

    return log_file


args = parse_args()
dataset_name = args.dataset

log_file = setup_logging(dataset_name)
logger = logging.getLogger(__name__)

config = {
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'n_item': dataset.m_item,
    'n_user': dataset.n_user,
    'keep_prob': 0.6,
    'A_split': False,
    'dropout': True,
    'null_dim': args.null_dim,
    'standardization': args.stand,
    'hidden_dim': args.hidden_dim,
    'ID_embs_init_type': args.ID_embs_init_type,
    'language_embs_scale': args.language_embs_scale,
    'null_thres': args.null_thres,
    'cover': args.cover,
    'item_frequency_flag': args.item_frequency_flag,
    'user_language_embs_path': f'../data/{dataset_name}/user_embeddings.pt',
    'item_language_embs_path': f'../data/{dataset_name}/item_embeddings.pt'
}
device = world.device
original = True
if original:
    Recmodel = register.MODELS[world.model_name](world.config, dataset)

Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = 1

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

best_recall = 0
patience = 100
patience_counter = 0
best_epoch = 0
best_results = None
total_epochs_run = 0

folder_path = f"ID_embs/{dataset_name}/{args.recdim}/"
if not os.path.exists(folder_path):
    try:
        for epoch in range(world.TRAIN_epochs):
            total_epochs_run = epoch + 1
            start = time.time()

            if epoch % 10 == 0:
                cprint("[TEST]")
                test_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

                current_recall = test_results['recall'][0]

                if current_recall > best_recall:
                    best_recall = current_recall
                    best_epoch = epoch
                    best_results = test_results.copy()
                    patience_counter = 0
                    torch.save(Recmodel.state_dict(), weight_file.replace('.pth', '_best.pth'))
                    print(f"New best recall@{world.topks[0]}: {best_recall:.4f} at epoch {epoch}")
                else:
                    patience_counter += 1
                    print(
                        f"No improvement for {patience_counter} test intervals. Best recall@{world.topks[0]}: {best_recall:.4f} at epoch {best_epoch}")

                if patience_counter >= patience // 10:
                    print(
                        f"Early stopping at epoch {epoch}. Best recall@{world.topks[0]}: {best_recall:.4f} at epoch {best_epoch}")
                    break

            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            print(f'Training LightGCN EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

        print(f"\n{'=' * 50}")
        print("TRAINING COMPLETED")
        print(f"{'=' * 50}")
        print(f"Total epochs run: {total_epochs_run}")
        print(f"Best results achieved at epoch {best_epoch}:")
        if best_results:
            print(f"Best results:")
            print(f"  Recall@{world.topks}: {best_results['recall']}")
            print(f"  Precision@{world.topks}: {best_results['precision']}")
            print(f"  NDCG@{world.topks}: {best_results['ndcg']}")

    finally:
        if world.tensorboard:
            w.close()

    Recmodel.eval()
    user_id_emb, item_id_emb, user_ego_emb, item_ego_emb = Recmodel.getAllEmbedding()

    folder_path = f"ID_embs/{dataset_name}/{args.recdim}/"
    os.makedirs(folder_path, exist_ok=True)

    torch.save(user_id_emb, os.path.join(folder_path, "user_id_emb.pt"))
    torch.save(item_id_emb, os.path.join(folder_path, "item_id_emb.pt"))
    torch.save(user_ego_emb, os.path.join(folder_path, "user_ego_emb.pt"))
    torch.save(item_ego_emb, os.path.join(folder_path, "item_ego_emb.pt"))

    print(f"Embeddings saved to {folder_path}")
else:
    user_id_emb_path = os.path.join(folder_path, "user_id_emb.pt")
    item_id_emb_path = os.path.join(folder_path, "item_id_emb.pt")
    user_ego_emb_path = os.path.join(folder_path, "user_ego_emb.pt")
    item_ego_emb_path = os.path.join(folder_path, "item_ego_emb.pt")

    if (os.path.exists(user_id_emb_path) and
            os.path.exists(item_id_emb_path) and
            os.path.exists(user_ego_emb_path) and
            os.path.exists(item_ego_emb_path)):
        user_id_emb = torch.load(user_id_emb_path)
        item_id_emb = torch.load(item_id_emb_path)
        user_ego_emb = torch.load(user_ego_emb_path)
        item_ego_emb = torch.load(item_ego_emb_path)

user_id_emb = user_id_emb.clone().detach()
item_id_emb = item_id_emb.clone().detach()
user_ego_emb = user_ego_emb.clone().detach()
item_ego_emb = item_ego_emb.clone().detach()
user_emb_path = f"../data/{dataset_name}/user_embeddings.pt"
item_emb_path = f"../data/{dataset_name}/item_embeddings.pt"
try:
    pretrained_user_emb = torch.load(user_emb_path)
    pretrained_item_emb = torch.load(item_emb_path)
    if pretrained_user_emb.shape[1] > args.recdim:
        print("Semantic Concentrating!")
        pretrained_user_emb = Recmodel.semantic_space_decomposion(pretrained_user_emb, args.recdim, config)
        pretrained_item_emb = Recmodel.semantic_space_decomposion(pretrained_item_emb, args.recdim, config)
    print(f'Pretrained embeddings loaded!')
except FileNotFoundError as e:
    print(f'Pretrained embedding file not found: {e}')

user_id_emb = user_id_emb.to(device)
user_ego_emb = user_ego_emb.to(device)
pretrained_user_emb = pretrained_user_emb.to(device)
item_id_emb = item_id_emb.to(device)
item_ego_emb = item_ego_emb.to(device)
pretrained_item_emb = pretrained_item_emb.to(device)

user_features = [user_id_emb, pretrained_user_emb]
item_features = [item_id_emb, pretrained_item_emb]

moe_config = MoEConfig(args.num_experts, args.recdim * 2, args.moe_hidden_size, args.moe_output_size, args.router_type, args.loss_coef, top_k=args.top_k)
recmoe = RecMoE(moe_config, moe_config)
recmoe = recmoe.to(device)

best_recall = 0
patience_counter = 0
lr_patience_counter = 0
best_epoch = 0
best_results = None
total_epochs_run = 0

try:
    current_lr = args.moe_lr
    lr_decay_factor = 0.5
    lr_decay_patience = 100
    early_stop_patience = 300

    logger.info("Training started")
    logger.info(f"Initial learning rate: {current_lr}")
    logger.info(f"Dataset: {world.dataset}")

    for epoch in range(world.TRAIN_epochs):
        total_epochs_run = epoch + 1
        start = time.time()

        if epoch % 10 == 0:
            logger.info("[TEST]")
            test_results = Procedure.Test_MoE(dataset, recmoe, epoch, user_features, item_features, w,
                                              world.config['multicore'])
            current_recall = test_results['recall'][0]

            if current_recall > best_recall:
                best_recall = current_recall
                best_epoch = epoch
                best_results = test_results.copy()
                patience_counter = 0
                lr_patience_counter = 0
                logger.info(f"New best recall@{world.topks[0]}: {best_recall:.4f} at epoch {epoch}")
            else:
                patience_counter += 1
                lr_patience_counter += 1
                logger.info(
                    f"No improvement for {patience_counter} test intervals. Best recall@{world.topks[0]}: {best_recall:.4f} at epoch {best_epoch}")

            if lr_patience_counter >= lr_decay_patience // 10:
                current_lr *= lr_decay_factor
                logger.info(f"Learning rate decayed to {current_lr:.6f} at epoch {epoch}")

                for param_group in bpr.opt.param_groups:
                    param_group['lr'] = current_lr

                lr_patience_counter = 0
                logger.info(f"LR patience counter reset.")

            if patience_counter >= early_stop_patience // 10:
                logger.info(
                    f"Early stopping at epoch {epoch}. Best recall@{world.topks[0]}: {best_recall:.4f} at epoch {best_epoch}")
                break

        output_information = Procedure.Contrastive_train_MoE(dataset, recmoe, bpr, epoch, current_lr, user_features,
                                                             item_features, neg_k=Neg_k, w=w)
        logger.info(f'Training MoE EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} (LR: {current_lr:.6f})')

    logger.info(f"\n{'=' * 50}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'=' * 50}")
    logger.info(f"Total epochs run: {total_epochs_run}")
    logger.info(f"Final learning rate: {current_lr:.6f}")
    logger.info(f"Best results achieved at epoch {best_epoch}:")
    if best_results:
        logger.info(f"Best results:")
        logger.info(f"  Recall@{world.topks}: {best_results['recall']}")
        logger.info(f"  Precision@{world.topks}: {best_results['precision']}")
        logger.info(f"  NDCG@{world.topks}: {best_results['ndcg']}")

    logger.info(f"Training log saved to: {log_file}")
finally:
    logger.info("All set!")
