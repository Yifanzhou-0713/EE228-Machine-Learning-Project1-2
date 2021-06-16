from functools import partial

import torch.nn as nn
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from fastai.vision import get_transforms
from torch import optim

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, softDiceLoss
import torch

import os



def main(args):
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    batch_size = 16
    num_workers = 4
    optimizer = optim.SGD
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, softDiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet(1, 1, first_out_channels=16)
    # model = nn.DataParallel(model.to(device))

    # model.load_state_dict(torch.load('./model_weights.pth'))

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)

    

    databunch = DataBunch(dl_train, dl_val,
        collate_fn=FracNetTrainDataset.collate_fn)


    learn = Learner(
        databunch,
        model,
        opt_func=optimizer,
        loss_func=criterion,
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial]
    )


    lr = (1e-1)*batch_size/256
    learn.fit_one_cycle(1, lr)

    learn.fit_one_cycle(
        cyc_len = 50,
        max_lr = 1e-1,
        pct_start=0,
        div_factor=1000,
        callbacks=[
            ShowGraph(learn),
        ]
    )

    if args.save_model:
        torch.save(model.state_dict(),"./model_weights.pth")
        # torch.save(model.module.state_dict(), "./model_weights_50.pth")


if __name__ == "__main__":
    import argparse

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True,
        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", required=True,
        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", required=True,
        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", required=True,
        help="The validation label nii directory.")
    parser.add_argument("--save_model", default=True,
        help="Whether to save the trained model.")
    args = parser.parse_args()

    main(args)
