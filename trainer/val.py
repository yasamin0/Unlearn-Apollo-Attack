import time 
from collections import OrderedDict
import torch 
import utils
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate(loader, model, loss_function, desc=""):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    for i, (images, labels) in enumerate(pbar:= tqdm(loader)):
        model.eval()
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)
        loss = loss_function(outputs, labels.to(torch.int64))

        acc1 = utils.accuracy(outputs.data, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

    print(f"{desc}Test Loss {losses.avg:.4f} Acc {top1.avg:.4f}")
    return OrderedDict([('loss', losses.avg), ('top1', top1.avg)])