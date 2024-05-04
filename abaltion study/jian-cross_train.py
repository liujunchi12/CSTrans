# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling_orig import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.transform import get_transform
from utils.dist_util import get_world_size
from utils.utils import visda_acc

from torchvision import transforms, datasets
from data.data_list_image import ImageList, Normalize, rgb_loader, ImageListIndex
import torch.nn.functional as F
import math
from models import contrastive_loss
from models import lossZoo

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, is_adv=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    if not is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    else:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
    model.load_from(np.load(args.pretrained_dir))

    ###########################################
    # model_dict = model.state_dict()
    # modelCheckpoint = torch.load(args.pretrained_dir)
    # new_dict = {k: v for k, v in modelCheckpoint.items() if k in model_dict.keys()}
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    ###########################################
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            # print("preds_shape:", preds.shape)
            # print("preds:", preds)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    if args.dataset == 'visda17':
        return accuracy, classWise_acc
    else:
        return accuracy, None


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)

    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    ########################################## MM1 Begin #########################################
    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    ########################################## MM1 Eed ###########################################

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD([
        {'params': model.transformer.parameters(), 'lr': args.learning_rate / 10},   #args.learning_rate / 10
        # {'params': model.conv1.parameters(), 'lr': args.learning_rate},
        # {'params': model.relu.parameters(), 'lr': args.learning_rate},
        # {'params': model.conv2.parameters(), 'lr': args.learning_rate},
        # {'params': model.bn2.parameters(), 'lr': args.learning_rate},
        # {'params': model.conv3.parameters(), 'lr': args.learning_rate},
        # {'params': model.bn3.parameters(), 'lr': args.learning_rate},
        # {'params': model.dwconv.parameters(), 'lr': args.learning_rate},
        # {'params': model.bn1.parameters(), 'lr': args.learning_rate},
        # {'params': model.relu.parameters(), 'lr': args.learning_rate},
        # {'params': model.pwconv1.parameters(), 'lr': args.learning_rate},
        # {'params': model.act.parameters(), 'lr': args.learning_rate},
        # {'params': model.pwconv2.parameters(), 'lr': args.learning_rate},
        {'params': model.head1.parameters(), 'lr': args.learning_rate},
        # {'params': model.bn1.parameters(), 'lr': args.learning_rate},
        {'params': model.layer_norm1.parameters(), 'lr': args.learning_rate},
        # {'params': model.avgpool.parameters(), 'lr': args.learning_rate},
        # {'params': model.fc.parameters(), 'lr': args.learning_rate},
        # {'params': model.activation.parameters(), 'lr': args.learning_rate},
        {'params': model.cluster_projector.parameters(), 'lr': args.learning_rate},
        {'params': model.head2.parameters(), 'lr': args.learning_rate},
        # {'params': model.conv1.parameters(), 'lr': args.learning_rate},
        # {'params': model.conv2.parameters(), 'lr': args.learning_rate},
        # {'params': model.instance_projector.parameters(), 'lr': args.learning_rate},
        # {'params': model.cluster_projector_soft.parameters(), 'lr': args.learning_rate},
        {'params': model.head3.parameters(), 'lr': args.learning_rate},
    ],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    # global_step, best_acc = 0, 0
    best_acc = 0
    best_classWise_acc = ''
    len_source = len(source_loader)
    len_target = len(target_loader)
    # print("len_source:", len_source)  #100 batch=8
    # print("len_target:", len_target)  #353 batch=8

    ##############################################
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.train_batch_size, args.instance_temperature,
                                                       loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(args.class_num, args.cluster_temperature, loss_device).to(
        loss_device)
    ##############################################

    for global_step in range(1, t_total):
        # while True:
        model.train()

        if (global_step - 1) % (len_source - 1) == 0:
            iter_source = iter(source_loader)
        if (global_step - 1) % (len_target - 1) == 0:
            iter_target = iter(target_loader)

        data_source = iter_source.next()
        data_target = iter_target.next()

        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        x_t, _, index_t = tuple(t.to(args.device) for t in data_target)

        # label_source_pred, loss_lmmd, z_i, z_j, c_i, c_j = model(x_s, x_t, y_s)
        ################################################################
        # label_source_pred, logits_t1, logits_t2, loss_lmmd1, loss_lmmd2 = model(x_s, x_t, y_s)
        label_source_pred, logits_t1, loss_lmmd1 = model(x_s, x_t, y_s)

        ################################################################
        # label_source_pred, loss_lmmd = model(x_s, x_t, y_s)
        ################################################################
        # label_source_pred = model(x_s)
        # loss_instance = criterion_instance(z_i, z_j)
        # loss_cluster = criterion_cluster(c_i, c_j)
        # loss_unsup = loss_instance + loss_cluster

        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), y_s)
        lambd = 2 / (1 + math.exp(-10 * (global_step) / t_total)) - 1
        # print("global_step:", global_step)
        # print("t_total:", t_total)
        # loss = loss_cls + args.weight * lambd * loss_lmmd
        # loss = loss_cls + 0.1 * loss_lmmd + 0.005 * loss_unsup
        # loss = loss_cls + 0.1 * loss_lmmd + 0.05 * loss_unsup

        ################################################################
        loss_im1 = lossZoo.im(logits_t1.view(-1, args.num_classes))
        # loss_im2 = lossZoo.im(logits_t2.view(-1, args.num_classes))
        # loss = loss_cls + 0.1 * loss_lmmd + 0.5 * loss_im
        ############################################################################################
        # loss = loss_cls + 0.1 * loss_lmmd1 + 0.1 * loss_lmmd2 + 0.4 * loss_im1 + 0.4 * loss_im2
        # loss = loss_cls + 0.1 * loss_lmmd1 + 0.1 * loss_lmmd2
        # loss = loss_cls + 0.1 * loss_lmmd1
        loss = loss_cls + 0.1 * loss_lmmd1 + 0.4 * loss_im1
        ############################################################################################
        # loss = loss_cls + 0.1 * loss_lmmd1 + 0.4 * loss_im1 + 0.4 * loss_im2
        ###############################################################
        # loss = loss_cls + 0.1 * loss_lmmd
        ################################################################
        # loss = loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()




        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_cls", scalar_value=loss_cls.item(), global_step=global_step)
            writer.add_scalar("train/loss_lmmd", scalar_value=loss_lmmd1.item(), global_step=global_step)
            # writer.add_scalar("train/loss_lmmd", scalar_value=loss_lmmd2.item(), global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            accuracy, classWise_acc = valid(args, model, writer, test_loader, global_step)
            if best_acc < accuracy:
                save_model(args, model)
                # save_model(args, is_adv=True)
                best_acc = accuracy

                if classWise_acc is not None:
                    best_classWise_acc = classWise_acc
            model.train()
            logger.info("Current Best Accuracy: %2.5f" % best_acc)
            logger.info("Current Best element-wise acc: %s" % best_classWise_acc)

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best element-wise Accuracy: \t%s" % best_classWise_acc)
    logger.info("End Training!")

    # epoch_iterator = tqdm(train_loader,
    #                       desc="Training (X / X Steps) (loss=X.X)",
    #                       bar_format="{l_bar}{r_bar}",
    #                       dynamic_ncols=True,
    #                       disable=args.local_rank not in [-1, 0])
    # for step, batch in enumerate(epoch_iterator):
    #     batch = tuple(t.to(args.device) for t in batch)
    #     x, y = batch
    #     loss = model(x, y)
    #
    # if args.gradient_accumulation_steps > 1:
    #     loss = loss / args.gradient_accumulation_steps
    # if args.fp16:
    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
    #         scaled_loss.backward()
    # else:
    #     loss.backward()

    # if (step + 1) % args.gradient_accumulation_steps == 0:
    #     losses.update(loss.item()*args.gradient_accumulation_steps)
    #     if args.fp16:
    #         torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
    #     else:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #     scheduler.step()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     global_step += 1
    #
    #     epoch_iterator.set_description(
    #         "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
    #     )
    #     if args.local_rank in [-1, 0]:
    #         writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
    #         writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
    #     if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
    #         accuracy, _ = valid(args, model, writer, test_loader, global_step)
    #         if best_acc < accuracy:
    #             save_model(args, model)
    #             best_acc = accuracy
    #         model.train()
    #
    #     if global_step % t_total == 0:
    #         break
    # losses.reset()
    # if global_step % t_total == 0:
    #     break
    #
    # if args.local_rank in [-1, 0]:
    #     writer.close()
    # logger.info("Best Accuracy: \t%f" % best_acc)
    # logger.info("End Training!")


def test(args):
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.pretrained_dir))
    model.to(args.device)

    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
    ])
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    writer = None
    accuracy, classWise_acc = valid(args, model, writer, test_loader, global_step=1)
    print(accuracy)
    print(classWise_acc)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.")
    parser.add_argument("--source_list", help="Path of the training data.")
    parser.add_argument("--target_list", help="Path of the test data.")
    parser.add_argument("--test_list", help="Path of the test data.")
    parser.add_argument("--num_classes", default=10, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")

    parser.add_argument("--img_size", default=256, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    ############################################################
    parser.add_argument('--instance_temperature', type=float,
                        help='instance_temperature', default=0.5)
    parser.add_argument('--cluster_temperature', type=float,
                        help='cluster_temperature size', default=1.0)
    parser.add_argument('--class_num', type=float,
                        help='class_num', default=31)
    ############################################################
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    if args.is_test:
        test(args)
    else:
        args, model = setup(args)
        train(args, model)


if __name__ == "__main__":
    main()
