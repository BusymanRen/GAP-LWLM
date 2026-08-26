import os
import argparse
import math
import random

import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.backends.cuda import sdp_kernel, SDPBackend

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from wlfm_prompt import *


class Wrapper(pl.LightningModule):
    def __init__(self, EnvPara):
        super().__init__()

        self.channel_fdmdl = wireless_loc_fm(
            fshape=EnvPara["fshape"], tshape=EnvPara["tshape"], fstride=EnvPara["fstride"], tstride=EnvPara["tstride"],
            input_fmap=EnvPara["input_fmap"], embed_dim=EnvPara["embed_dim"], depth=EnvPara["depth"],
            num_heads=EnvPara["num_heads"], device=EnvPara["device"], EnvPara=EnvPara
        )
        self.task = EnvPara["task"]
        self.train_epoch_loss = [] 
        self.valepoch_loss = []
        self.EnvPara = EnvPara
        torch.autograd.set_detect_anomaly(True)
        sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)

    def forward(self, channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all, scene_graphs=None):
        # 移除了冗余的 if-elif 分支，直接统一调用，代码更干净
        loss = self.channel_fdmdl(
            channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa, 
            channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri, 
            channel_delay_angle_ri=channel_delay_angle_ri, 
            UElocation_all=UElocation_all, 
            BSconf_all=BSconf_all, 
            task=self.task,
            scene_graphs=scene_graphs
        )
        return loss
    
    def training_step(self, batch, batch_idx):
        channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all, _, scene_graphs = batch
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.float()
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.float()
        channel_delay_angle_ri = channel_delay_angle_ri.float()
        UElocation_all = UElocation_all.float()
        BSconf_all = BSconf_all.float()
        scene_graphs = scene_graphs.float()

        loss = self.forward(
            channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa, 
            channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri, 
            channel_delay_angle_ri=channel_delay_angle_ri, 
            UElocation_all=UElocation_all, 
            BSconf_all=BSconf_all,
            scene_graphs=scene_graphs
        )
                    
        self.train_epoch_loss.append(loss.detach())
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # calculate average loss
        avg_loss = torch.stack(self.train_epoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Training Loss: {avg_loss.item()}", len(self.train_epoch_loss))
        self.log('train/ave_loss', avg_loss.item(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        # Clear the loss list for the next epoch of training
        self.train_epoch_loss.clear()
    
    def validation_step(self, batch, batch_idx):
        channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all, _, scene_graphs = batch
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.float()
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.float()
        channel_delay_angle_ri = channel_delay_angle_ri.float()
        UElocation_all = UElocation_all.float()
        BSconf_all = BSconf_all.float()
        scene_graphs = scene_graphs.float()

        loss = self.forward(
            channel_Antenna_subcarrier_aa=channel_Antenna_subcarrier_aa, 
            channel_Antenna_subcarrier_ri=channel_Antenna_subcarrier_ri, 
            channel_delay_angle_ri=channel_delay_angle_ri, 
            UElocation_all=UElocation_all, 
            BSconf_all=BSconf_all,
            scene_graphs=scene_graphs
        )

        self.valepoch_loss.append(loss.detach())
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # calculate average loss
        avg_loss = torch.stack(self.valepoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_loss.item()}", len(self.valepoch_loss))
        self.log('val/ave_loss', avg_loss.item(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        # Clear the loss list for the next epoch of training
        self.valepoch_loss.clear()

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.EnvPara["lr"], weight_decay=1e-4, eps=1e-6)
        
        #self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=1e-3, total_steps=15625000, pct_start=0.1, final_div_factor=1e2)

        return {
            'optimizer': self.optim, 
            #'lr_scheduler': {'scheduler': self.schedule, 'interval': 'step'}
        }