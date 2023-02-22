import argparse
import logging
import os
import torch
import torch.nn as nn
import transformers
import time
import datetime
import random
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter, AutoDoor, import_source_file
from irmas_dataset import IRMASDataset
from pathlib import Path
import transformers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings("ignore")


# https://github.com/pytorch/vision/blob/main/references/classification/train.py
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Instrument Recognition Training", add_help=add_help)

    parser.add_argument("--model", type=str, required=True, help="model name ['han_model','resnet50']")
    # training config
    parser.add_argument("--seed", type=int, default=2333, help="A seed for reproducible training.")
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--bs", required=True, type=int, metavar="N", help="training batch size")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--lr-warmup-epochs", default=0, type=float, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--els", default=3, type=int, metavar="N", help="early stopping patience")
    parser.add_argument("--save-metric", default='loss', type=str,help="metric for save the best model")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
        dest="weight_decay",
    )
    parser.add_argument("--optim", default='no-decay', type=str, help="lr scheduler")

    # datasets
    parser.add_argument("--train-meta", type=str, required=True, help="path to training metadata")
    parser.add_argument("--valid-meta", type=str, required=True, help="path to valid metadata")
    parser.add_argument("--wav-dir", type=str, required=True, help="path to IRMAS dataset")

    parser.add_argument('--normalize_amp', action='store_true')
    parser.add_argument('--no-normalize_amp', dest='normalize_amp', action='store_false')
    parser.set_defaults(normalize_amp=False)
    # parser.add_argument(
    #     "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    # )

    # other training config
    # parser.add_argument(
    #     "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    # )

    # experiment dirs
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the final model.")

    # model configs
    parser.add_argument('--imagenet_pretrained', action='store_true')
    parser.add_argument('--no-imagenet_pretrained', dest='imagenet_pretrained', action='store_false')
    parser.set_defaults(imagenet_pretrained=False)

    parser.add_argument("--ch_expand", type=str, required=True, help="channel conversion")
    # parser.add_argument("--win_len", type=int, required=True, help="n fft")
    # parser.add_argument("--feature_dim", required=True, type=int, help="feature dim")

    return parser


def train(args):
    prepare_experiments(args)
    writer = SummaryWriter(args.output_dir)

    logging.basicConfig(
        handlers=[
            logging.FileHandler("{}/run-{}.log".format(args.output_dir, args.seed)),
            logging.StreamHandler()
        ],
        format='[%(levelname)s] %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S %p | ',
        level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('training on %s', device)

    # prepare datasets
    train_loader, valid_loader, n_tr_samples, n_va_samples = prepare_data(args)

    # model
    model = import_source_file(Path("./models/{}.py".format(args.model)),
                               "m").IRNet(
        pretrained=args.imagenet_pretrained,
        ch_expand=args.ch_expand,
        # n_fft=args.win_len,
        # feature_dim=args.feature_dim,
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.warmup_steps = int(args.lr_warmup_epochs * len(train_loader))
    args.training_steps = int(args.epochs * len(train_loader))

    main_lr_scheduler = get_scheduler(args, optimizer)
    if args.lr_warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=args.lr_warmup_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    # training
    logging.info("Start training")
    save_metric = AutoDoor(args.save_metric)
    start_time = time.time()
    global_step = 0
    for epoch in range(0, args.epochs):
        loss_meter = AverageMeter()

        correct = 0

        # train one epoch
        model.train()
        progress_bar = tqdm(train_loader)
        for i, batch in enumerate(progress_bar):
            signals = batch['feature'].to(device)
            labels_one_hot = batch['one_hot_label'].to(device)
            labels = batch['label'].to(device)
            _, outputs = model(signals)
            loss = criterion(outputs, labels_one_hot)
            _, predicted = torch.max(outputs.data, 1)

            # warm up
            if global_step <= args.warmup_steps and args.lr_warmup_epochs > 0:
                warm_lr = (global_step / args.warmup_steps) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr

                # print('[{}/{}]warm-up learning rate is {:f}'.format(
                #     global_step,
                #     args.warmup_steps,
                #     optimizer.param_groups[0]['lr'])
                # )
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'] , global_step)
            global_step += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            n_samples = signals.shape[0]
            loss_meter.update(loss.item(), n_samples)
            correct += (predicted == labels).sum().item()

            progress_bar.set_description(
                "Train Epoch{}/{} Step[{}/{}] Loss:{:.4f}".format(
                    epoch, args.epochs, i + 1, len(train_loader), loss.item()
                )
            )

        train_info = 'Epoch: {} | lr:{:.4f} | Train Loss:{:.4f} | Train Accuracy: {:.4f}'.format(
            epoch, optimizer.param_groups[0]['lr'], loss_meter.avg, correct / n_tr_samples
        )
        writer.add_scalar('Loss/train', loss_meter.avg, epoch)
        writer.add_scalar('Accuracy/train', correct / n_tr_samples , epoch)
        loss_meter.reset()
        correct = 0
        lr_scheduler.step()


        # evaluation
        model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(valid_loader):
                signals = batch['feature'].to(device)
                labels_one_hot = batch['one_hot_label'].to(device)
                labels = batch['label'].to(device)
                _, outputs = model(signals)
                loss = criterion(outputs, labels_one_hot)
                _, predicted = torch.max(outputs.data, 1)

                n_samples = signals.shape[0]
                loss_meter.update(loss.item(), n_samples)
                correct += (predicted == labels).sum().item()
        valid_info = ' | Valid Loss:{:.4f} | Valid Accuracy: {:.4f}'.format(
            loss_meter.avg, correct / n_va_samples
        )
        writer.add_scalar('Loss/valid', loss_meter.avg, epoch)
        writer.add_scalar('Accuracy/valid', correct / n_va_samples, epoch)
        logging.info(train_info + valid_info)

        # save best model
        if args.save_metric == 'loss':
            save_metric.update(loss_meter.avg)
        elif args.save_metric == 'acc':
            save_metric.update(correct / n_va_samples)
        else:
            raise NotImplementedError
        save_state = {
            "model": model.state_dict(),
            # "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            # "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
        }
        # for wa
        if epoch % 35 == 0:
            torch.save(save_state, os.path.join(args.output_dir, "epoch_{}.pth".format(epoch)))
            logging.info("model saved to {}/epoch_{}.pth".format(args.output_dir,epoch))

        if save_metric.save:
            torch.save(save_state, os.path.join(args.output_dir, "best.pth"))
            logging.info("Epoch: {} model saved to {}/best.pth".format(epoch, args.output_dir))
        if save_metric.els >= args.els and epoch > 24:  # early stopping
            logging.info("early stopping at epoch {}".format(epoch))
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('training ends in epoch {}, with {}'.format(epoch, total_time_str))

    torch.save(save_state, os.path.join(args.output_dir, "last.pth"))
    logging.info("model saved to {}/last.pth".format(args.output_dir))


def prepare_experiments(args):
    if args.seed:
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    config_dict = vars(args)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(config_dict, outfile)
    print(config_dict)


def prepare_data(args):
    train_dataset = IRMASDataset(meta_path=args.train_meta,
                                 wav_dir=args.wav_dir,
                                 normalize_amp=args.normalize_amp,
                                 )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)

    valid_dataset = IRMASDataset(meta_path=args.valid_meta,
                                 wav_dir=args.wav_dir,
                                 normalize_amp=args.normalize_amp,
                                 )
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=512,  # fix the valid batch size
                                               shuffle=False,
                                               num_workers=8,
                                               pin_memory=False)
    logging.info('Data Loaded:\n {} Train Samples \n {} Valid Samples'.format(
        len(train_dataset), len(valid_dataset)
    ))
    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)


def get_scheduler(args, optimizer):
    optim_name = args.optim.lower()
    if optim_name.startswith('cos'):
        # optim = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0
        # )
        optim = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.training_steps,
        )
    elif optim_name.startswith('no-decay'):  # for reproduce han's work (no lr decay)
        optim = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9999
        )
    elif optim_name.startswith('exp'):
        optim = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.97
        )
    else:
        raise NotImplementedError
    return optim


if __name__ == '__main__':
    configs = get_args_parser().parse_args()
    train(configs)
