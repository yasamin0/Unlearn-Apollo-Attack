import time 
from collections import OrderedDict
import torch 
import utils
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epoch, loader, model, loss_function, optimizer, scheduler=None):
    print("Training Epoch: ", epoch)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    for i, (images, labels) in enumerate(pbar := tqdm(loader)):        
        model.train()
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels.to(torch.int64))
        loss.backward()
        optimizer.step()

        acc1 = utils.accuracy(outputs.data, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        if scheduler is not None:
            scheduler.step()
        
        pbar.set_postfix({
            "Loss": losses.avg,
            "Accuracy": top1.avg,
        })
    lr = optimizer.param_groups[0]['lr']

    return OrderedDict([('loss', losses.avg), ('top1', top1.avg), ('lr', lr)])