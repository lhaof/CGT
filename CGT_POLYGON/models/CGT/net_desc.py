import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x,UpSample4x)
from .utils import crop_op, crop_to_shape
from .van import *
from .bilinear import bilinear_edge,bilinear_mask,bilinear_centers
from .deepgcn import DeeperGCN
from .LSTM import LSTMClassifier
from .RNN import RNN
from .query2label import tokengt
####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'original': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.encoder = van_base(pretrained=True)
        self.roialigner = torchvision.ops.RoIAlign(output_size=7,sampling_ratio=2,spatial_scale=0.25)
        self.conv_bot = nn.Conv2d(512, 320, 1, stride=1, padding=0, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.deepgcn = DeeperGCN(data_channels=256,edge_channels=64,hidden_channels=64, num_layers=2).to(device)
        self.transformer = tokengt(nr_types-1,hidden_dim=256,encoder_layers=2, decoder_layers=2)
        #self.shapercnn = RNN(embedding_dim=67, hidden_dim=128, output_size=64)
        self.cls = nn.Conv2d(256*3, nr_types-1, 1, stride=1, padding=0, bias=True)
        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(320, 128, 1, stride=1, padding=0, bias=False)),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(128, 64, ksize, stride=1, padding=1, bias=False)),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(64, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder
        ksize = 3 if mode == 'original' else 3
        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                ]
            )
        )

        self.upsample2x = UpSample2x()
        self.upsample4x = UpSample4x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    #def forward(self, imgs, bboxes,centers,edge_points,edge_indexes,batch_pos_emb,batch_length,batch_select_shape_feats,mode):
    def forward(self, imgs,centers,edge_points,edge_indexes,batch_pos_emb,batch_length,mode):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            with torch.set_grad_enabled(not self.freeze):
                d = self.encoder(imgs,self.freeze)
            d3 = d[3]
            d3 = self.conv_bot(d3)
            d = [d[0], d[1], d[2], d3]
            #print(d0.shape, d1.shape, d2.shape, d3.shape)
        else:
            d = self.encoder(imgs)
            d3 = d[3]
            d3 = self.conv_bot(d3)
            d = [d[0], d[1], d[2], d3]

        # TODO: switch to `crop_to_shape` ?

        #print(d[0].shape, d[1].shape)
        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)
            #print(u2.shape)
            if mode == 'train':
  
            #print(pred_class.shape)
                u1 = self.upsample2x(u2) + d[-4]
                u1 = branch_desc[2](u1)
                u0 = self.upsample4x(u1)
                u0 = branch_desc[3](u0)
                #aligned_features = self.roialigner(u1,bboxes)
                #inst_features = self.gap(aligned_features).reshape(aligned_features.shape[0],aligned_features.shape[1])
                #print(inst_features.shape)
                aligned_center_features = []
                node_feature = []
                batch_feature = []
                for i in range(len(centers)):
                    single_center = centers[i]
                    #single_shape_feature = batch_select_shape_feats[i]
                    #print(single_shape_feature.shape)
                    #print(single_center.shape)
                    single_pos_emb = batch_pos_emb[i]
                    aligned_center_feature = bilinear_centers(u1[i,:,:,:],single_center,scale=0.25)
                    #print(aligned_mask_feature.shape,aligned_centers.shape)
                    #single_shape_feature = single_shape_feature.reshape(single_shape_feature.shape[0],single_shape_feature.shape[1],single_shape_feature.shape[2])

                    #aligned_shape_feature = self.shapercnn(single_shape_feature)
                    #print(aligned_mask_feature.shape)
                    single_edge_points = edge_points[i]
                    single_edge_indexs = edge_indexes[i]
                    #print(single_edge_indexs.shape,single_edge_points.shape)
                    single_edge_points = bilinear_edge(u1[i,:,:,:],single_edge_points,scale=0.25)
                    #single_edge_points = aligned_edge_matrix.reshape(batch_length[i],batch_length[i])
                    #print(single_edge_points.shape)
                    node_feature = torch.cat((aligned_center_feature,single_pos_emb),1)
                    #print(node_feature.shape)
                    #updated_feature = node_feature[:,:192]
                    #updated_pos = node_feature[:,192:]
                    updated_feature = self.transformer(node_feature,single_edge_points,single_edge_indexs)
                    #print(updated_feature.shape)
                    #updated_feature = self.deepgcn(node_feature,single_edge_indexs,single_edge_points)
                    batch_feature.append(updated_feature)
                batch_feature = torch.cat(batch_feature,0)
                #print(batch_feature.shape)
                #print(aligned_features.max(),aligned_features.min())
            #print(aligned_features.shape)
                
            #print(inst_features.shape)
                batch_feature = batch_feature.reshape(batch_feature.shape[0],batch_feature.shape[1],1,1)
            #print(inst_features.shape)
                pred_class = self.cls(batch_feature)
                #print(pred_class.shape)
                pred_class = pred_class.reshape(pred_class.shape[0],pred_class.shape[1])
            #print(u3.shape, u2.shape, u1.shape, u0.shape)
                out_dict[branch_name] = [u0,pred_class]
            else:
                u1 = self.upsample2x(u2) + d[-4]
                u1 = branch_desc[2](u1)
                u0 = self.upsample4x(u1)
                u0 = branch_desc[3](u0)
            #print(u3.shape, u2.shape, u1.shape, u0.shape)
                out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)

