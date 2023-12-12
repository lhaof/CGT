import torch
import torch.nn.functional as F

import torch.nn as nn
from torch.nn import LayerNorm, Linear, ReLU
from fairseq import utils
from .models.tokengt import TokenGTEncoder
import torch
import numpy as np
import pyximport

from . import algos
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos_spd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num-atoms", type=int, metavar="D",default=2500,  help="dropout prob")
parser.add_argument("--num-edges", type=int, metavar="D",default=2500*4,  help="dropout prob")
parser.add_argument("--num-in-degree", type=int, metavar="D",default=2500,  help="dropout prob")
parser.add_argument("--num-out-degree", type=int, metavar="D",default=2500,  help="dropout prob")
parser.add_argument("--num-edge-dis", type=int, metavar="D",default=512,  help="dropout prob")
parser.add_argument("--num-spatial", type=int, metavar="D",default=512,  help="dropout prob")
parser.add_argument("--multi-hop-max-dist", type=int, metavar="D",default=5,  help="dropout prob")
parser.add_argument("--spatial-pos-max", type=int, metavar="D",default=1024,  help="dropout prob")
parser.add_argument("--edge-type", default="multi_hop",  help="dropout prob")

parser.add_argument("--dropout", type=float, metavar="D",default=0.0,  help="dropout prob")
parser.add_argument("--attention-dropout", type=float,default=0.1,  metavar="D", help="dropout prob for attention weights")
parser.add_argument("--act-dropout", type=float,default=0.1,  metavar="D", help="dropout prob after activation in FFN")

parser.add_argument("--encoder-ffn-embed-dim", type=int,default=256, metavar="N", help="encoder embedding dim for FFN")
parser.add_argument("--encoder-layers", type=int,default=4, metavar="N", help="num encoder layers")
parser.add_argument("--encoder-attention-heads", type=int,default=8, metavar="N", help="num encoder attention heads")
parser.add_argument("--encoder-embed-dim", type=int,default=256*3, metavar="N", help="encoder embedding dimension")
parser.add_argument("--share-encoder-input-output-embed", action="store_true",
                    help="share encoder input and output embeddings")
parser.add_argument("--max-nodes", action="store_true",default=128, help="use random feature node identifiers")
parser.add_argument("--rand-node-id", action="store_true", help="use random feature node identifiers")
parser.add_argument("--rand-node-id-dim", type=int, metavar="N", help="dim of random node identifiers")
parser.add_argument("--orf-node-id", action="store_true", help="use orthogonal random feature node identifiers")
parser.add_argument("--orf-node-id-dim", type=int, metavar="N", help="dim of orthogonal random node identifier")
parser.add_argument("--lap-node-id",default=True, action="store_true", help="use Laplacian eigenvector node identifiers")
parser.add_argument("--lap-node-id-k",default=16, type=int, metavar="N",
                    help="number of Laplacian eigenvectors to use, from smallest eigenvalues")
parser.add_argument("--lap-node-id-sign-flip", default=False,action="store_true", help="randomly flip the signs of eigvecs")
parser.add_argument("--lap-node-id-eig-dropout", type=float,default=0.0, metavar="D", help="dropout prob for Lap eigvecs")
parser.add_argument("--type-id", default=True,action="store_true", help="use type identifiers")

parser.add_argument("--stochastic-depth", default=True,action="store_true", help="use stochastic depth regularizer")

parser.add_argument("--performer",default=False, action="store_true", help="linearized self-attention with Performer kernel")
parser.add_argument("--performer-nb-features", type=int, metavar="N",
                    help="number of random features for Performer, defaults to (d*log(d)) where d is head dim")
parser.add_argument("--performer-feature-redraw-interval", type=int,default=1000, metavar="N",
                    help="how frequently to redraw the projection matrix for Performer")
parser.add_argument("--performer-generalized-attention", action="store_true",default=False,
                    help="defaults to softmax approximation, but can be set to True for generalized attention")
parser.add_argument("--performer-finetune", action="store_true",
                    help="load softmax checkpoint and fine-tune with performer")

parser.add_argument("--apply-graphormer-init", action="store_true", help="use Graphormer initialization")
parser.add_argument("--activation-fn", choices=utils.get_available_activation_fns(),default="gelu", help="activation to use")
parser.add_argument("--encoder-normalize-before", action="store_true", help="apply layernorm before encoder")
parser.add_argument("--prenorm",default=True, action="store_true", help="apply layernorm before self-attention and ffn")
parser.add_argument("--postnorm",default=False, action="store_true", help="apply layernorm after self-attention and ffn")
parser.add_argument("--return-attention", action="store_true",default=True, help="obtain attention maps from all layers",)



@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long).to('cuda:0')
    x = x + feature_offset
    return x



class tokengt(nn.Module):
    """Modified Query2Label model
    Unlike the model described in the paper (which uses a modified DETR 
    transformer), this version uses a standard, unmodified Pytorch Transformer. 
    Learnable label embeddings are passed to the decoder module as the target 
    sequence (and ultimately is passed as the Query to MHA).
    """
    def __init__(
        self, num_classes, hidden_dim=256, nheads=8, 
        encoder_layers=6, decoder_layers=6, use_pos_encoding=False):
        """Initializes model
        Args:
            model (str): Timm model descriptor for backbone.
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
            nheads (int, optional): Number of MHA heads. Defaults to 8.
            encoder_layers (int, optional): Number of encoders. Defaults to 6.
            decoder_layers (int, optional): Number of decoders. Defaults to 6.
            use_pos_encoding (bool, optional): Flag for use of position encoding. 
            Defaults to False.
        """        
        
        super().__init__()
        args = parser.parse_args()
        #print(args)
        self.model = TokenGTEncoder(args)


    def forward(self,x, edge_attr,edge_index):
        """Passes batch through network
        Args:
            x (Tensor): Batch of images
        Returns:
            Tensor: Output of classification head
        """        
        # produces output of shape [N x C x H x W]
        #print(x.shape)
        edge_int_feature, edge_index, node_int_feature = edge_attr, edge_index, x
        #print(edge_int_feature.shape, edge_index.shape, node_int_feature.shape)
        #node_data = convert_to_single_emb(node_int_feature)
        node_data = node_int_feature
        if len(edge_int_feature.size()) == 1:
            edge_int_feature = edge_int_feature[:, None]
        #edge_data = convert_to_single_emb(edge_int_feature)
        edge_data = edge_int_feature
        N = node_int_feature.size(0)
        dense_adj = torch.zeros([N, N], dtype=torch.bool)
        dense_adj[edge_index[0, :], edge_index[1, :]] = True
        in_degree = dense_adj.long().sum(dim=1).view(-1)
        lap_eigvec, lap_eigval = algos.lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]
        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)
        #print(edge_data.shape, in_degree, lap_eigvec.shape)
        node_num = [node_data.size(0)]
        edge_num = [edge_data.size(0)]
        batched_data = {}
        batched_data["node_data"] = node_data
        batched_data["in_degree"] = in_degree
        batched_data["out_degree"] = in_degree
        batched_data["node_num"] =  node_num
        batched_data["lap_eigvec"] = lap_eigvec
        batched_data["lap_eigval"] = lap_eigval
        batched_data["edge_index"] = edge_index
        batched_data["edge_data"] = edge_data
        batched_data["edge_num"] = edge_num
        B = 1
        h,atten = self.model(batched_data)
        
        h = h[0,:N,:]
        #print(h.shape)
        return h




