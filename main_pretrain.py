import os
import argparse
# from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from trainer import train, validate
from models import create_model
from dataset import create_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Pretraining Config')
    parser.add_argument('--data_dir',       type=str,   default='./data',   help='path to dataset')
    parser.add_argument('--dataset',        type=str,   default='',         help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--size_train',     type=int,   default=2500,       help='train set size (default: 2500)')

    parser.add_argument('--model',          type=str,   default='ResNet18', help='model architechture (default: "ResNet18"')
    parser.add_argument('--num_classes',    type=int,   default=None,       help='number of label classes (Model default if None)')
    parser.add_argument('--input_size',     type=int,   default=None,       nargs=3, help='Input all image dimensions (d h w, e.g. --input_size 3 224 224)')
    parser.add_argument('--batch_size',     type=int,   default=128,        help='input batch size for training (default: 128)')

    parser.add_argument('--opt',            type=str,   default='sgd',      help='Optimizer (default: "sgd")')
    parser.add_argument('--momentum',       type=float, default=0.9,        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay',   type=float, default=5e-4,       help='weight decay (default: 5e-4)')

    parser.add_argument('--sched',          type=str,   default='cosine',   help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr',             type=float, default=None,       help='learning rate, overrides lr-base if set (default: None)')
    parser.add_argument('--epochs',         type=int,   default=200,        help='number of epochs to train (default: 200)')

    parser.add_argument('--seed',           type=int,   default=42,         help='random seed (default: 42)')
    args = parser.parse_args()


    utils.random_seed(args.seed)
    save_path = os.path.join("./save", f"{args.model}-{args.dataset}", f"train-{str(args.size_train)}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    weights_path = os.path.join(save_path, "original.pth.tar")

    # dataloaders
    dataset = create_dataset(dataset_name=args.dataset, setting="Partial", root=args.data_dir, img_size=args.input_size[-1])
    dataset.set_train_valid_shadow_idx(size_train=args.size_train, seed=args.seed)

    trainset = dataset.get_subset(dataset.train_idx)
    validset = dataset.get_subset(dataset.valid_idx)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # get network
    model = create_model(model_name=args.model, num_classes=args.num_classes)
    model.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = None 
    best_epoch = None 
    for epoch in range(1, args.epochs + 1):
        train_metrics = train(epoch, train_loader, model, loss_function, optimizer)
        eval_metrics = validate(valid_loader, model, loss_function)
        scheduler.step()
        utils.update_summary(
            epoch, train_metrics, eval_metrics, os.path.join(save_path, "summary.csv"),
            write_header=best_acc is None,
        )
        if best_acc is None or best_acc < eval_metrics["top1"]:
            print("saving weights file to {}".format(weights_path))
            torch.save(model.state_dict(), weights_path)
            best_acc = eval_metrics["top1"]
            best_epoch = epoch
    print('*** Best metric: {0} (epoch {1})'.format(best_acc, best_epoch))


if __name__ == '__main__':
    main()