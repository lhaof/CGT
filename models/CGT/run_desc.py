import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, xentropy_loss, CE_loss,get_bboxes,focal_loss,weighted_entropy_loss,add_class,get_infer_bboxes
import os
from collections import OrderedDict
####

training_weights = [5,2,4] ##Change based on different datasets
edge_num = 4

def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    loss_func_dict = {
        "bce": weighted_entropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "focal": focal_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})
    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    ####
    imgs = batch_data["img"]
    inst_map = batch_data["inst_map"]
    #cv2.imwrite('batch_img.png',np.array(imgs[0,:,:]))
    #cv2.imwrite('batch.png',np.uint8(np.array(inst_map[0,:,:])))
    #true_hv = batch_data["hv_map"]

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW

    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    #inst_map = inst_map.to("cuda").type(torch.int64)
    #true_hv = true_hv.to("cuda").type(torch.float32)

    #batch_bboxes = torch.stack(batch_bboxes,0)
    #print(batch_bboxes.shape)
    #true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        #"np": true_np_onehot,
        #"hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        #print(true_tp.shape)
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot
    #print(inst_map.shape)
    type_map = batch_data["tp_map"]
    batch_boxes = []
    inst_classes= []
    batch_centers = []
    batch_edge_points = []
    batch_edge_indexs = []
    batch_pos_emb = []
    batch_length = []
    batch_select_shape_feats = []
    for i in range(inst_map.shape[0]):
       single_map = inst_map[i,:,:].reshape(1,inst_map.shape[1],inst_map.shape[2])
       single_type_map = type_map[i,:,:]
       #boxes,inst_class,centers,edge_points,edge_index,pos_emd,select_shape_feats = get_bboxes(single_map,single_type_map)                                  
       inst_class,centers,edge_points,edge_index,pos_emd = get_bboxes(single_map,single_type_map,edge_num)
       #print(true_bboxes.shape,inst_class.shape)
       #batch_select_shape_feats.append(select_shape_feats)
       batch_length.append(centers.shape[0])
       #batch_boxes.append(boxes)  
       batch_centers.append(centers)
       batch_edge_points.append(edge_points)
       batch_edge_indexs.append(edge_index)
       batch_pos_emb.append(pos_emd)
       inst_classes.append(inst_class.reshape(inst_class.shape[0],1))
    #batch_centers = torch.stack(batch_centers)
    #batch_edge_matrix = torch.stack(batch_edge_matrix)
    #batch_pos_emb = torch.stack(batch_pos_emb)
    inst_classes = torch.cat(inst_classes,0)
    true_inst_classes = torch.squeeze(inst_classes).to("cuda").type(torch.int64)
    true_inst_classes_onehot = F.one_hot(true_inst_classes, num_classes=model.module.nr_types-1)
    true_inst_classes_onehot = true_inst_classes_onehot.type(torch.float32)
    #print(true_inst_classes_onehot.shape)
    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    #pred_dict = model(imgs,batch_boxes,batch_centers,batch_edge_points,batch_edge_indexs,batch_pos_emb,batch_length,batch_select_shape_feats,mode='train')
    pred_dict = model(imgs,batch_centers,batch_edge_points,batch_edge_indexs,batch_pos_emb,batch_length,mode='train')
    #pred_dict = OrderedDict(
        #[[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    #)
    #pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    pred_map, pred_classes = pred_dict["tp"][0],pred_dict["tp"][1]
    pred_map = pred_map.permute(0, 2, 3, 1).contiguous()
    pred_map = F.softmax(pred_map, dim=-1)
    pred_classes = F.softmax(pred_classes, dim=-1)
    #print(pred_map.shape,pred_classes.shape)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        #for loss_name, loss_weight in loss_opts[branch_name].items():
            #print(loss_name)
            #loss_func = loss_func_dict[loss_name]
            #loss_args = [true_dict[branch_name], pred_map]
            #if loss_name == "msge":
                #loss_args.append(true_np_onehot[..., 1])
            #term_loss = loss_func(*loss_args)
            #track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            #print(term_loss)
            #loss += loss_weight * term_loss
        ce_loss = CE_loss(pred_classes,true_inst_classes_onehot)
        class_loss = focal_loss(pred_classes,true_inst_classes_onehot,training_weights)
        track_value("loss_%s_%s" % (branch_name, "focal"), class_loss.cpu().item())
        track_value("loss_%s_%s" % (branch_name, "ce"), ce_loss.cpu().item())
        print(class_loss.cpu().item(),ce_loss.cpu().item())
        loss += class_loss
        loss += ce_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update
    correct = inst_classes.cpu().numpy().reshape(inst_classes.shape[0])
    #print('correct',correct)
    prediction = torch.argmax(pred_classes, 1).cpu().numpy()
    #print('prediction',prediction)
    acc_all = (prediction == correct).mean()
    acc = [0 for c in [0,1,2]]
    print('accu_all',acc_all)
    #print(prediction)
    #print(correct)
    for c in [0,1,2]:
        acc[c] = ((prediction == correct) * (correct == c)).sum() / (max((correct == c).sum(), 1))
    print('accu_per',acc)
    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = 0
    #print(imgs[0].shape)
    imgs = (imgs[0].view(1,imgs.shape[1],imgs.shape[2],imgs.shape[3])).byte()  # to uint8
    #print(imgs.shape)
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    #pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    #true_dict["np"] = true_np
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        #"np": (true_dict["np"], pred_dict["np"]),
        #"hv": (true_dict["hv"], pred_dict["hv"]),
    }
    return result_dict


####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    #true_np = batch_data["np_map"]
    #true_hv = batch_data["hv_map"]
    inst_map = batch_data["inst_map"]
    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    #true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
    #true_hv = torch.squeeze(true_hv).to("cuda").type(torch.float32)

    true_dict = {
        #"np": true_np,
        #"hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_dict["tp"] = true_tp
    type_map = batch_data["tp_map"]
    batch_boxes = []
    inst_classes= []
    batch_centers = []
    batch_edge_points = []
    batch_edge_indexs = []
    batch_pos_emb = []
    batch_length = []
    batch_select_shape_feats = []
    for i in range(inst_map.shape[0]):
       single_map = inst_map[i,:,:].reshape(1,inst_map.shape[1],inst_map.shape[2])
       single_type_map = type_map[i,:,:]
       #boxes,inst_class,centers,edge_points,edge_index,pos_emd,select_shape_feats = get_bboxes(single_map,single_type_map)
       inst_class,centers,edge_points,edge_index,pos_emd = get_bboxes(single_map,single_type_map,edge_num)
       batch_length.append(centers.shape[0])
       #batch_select_shape_feats.append(select_shape_feats)
       #batch_boxes.append(boxes)
       batch_centers.append(centers)
       batch_edge_points.append(edge_points)
       batch_edge_indexs.append(edge_index)
       batch_pos_emb.append(pos_emd)
       inst_classes.append(inst_class.reshape(inst_class.shape[0],1))
    inst_classes = torch.cat(inst_classes,0)
    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        #pred_dict = model(imgs_gpu,batch_boxes,batch_centers,batch_edge_points,batch_edge_indexs,batch_pos_emb,batch_length,batch_select_shape_feats,mode='train')
        pred_dict = model(imgs_gpu,batch_centers,batch_edge_points,batch_edge_indexs,batch_pos_emb,batch_length,mode='train')
        pred_map, pred_classes = pred_dict["tp"][0],pred_dict["tp"][1]
        pred_map = pred_map.permute(0, 2, 3, 1).contiguous()
        type_map = torch.argmax(pred_map, dim=-1, keepdim=False)
        type_map = type_map.type(torch.float32)
        pred_dict["tp"] = type_map


    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            #"pred_classes": pred_dict["np"].cpu().numpy(),
            #"pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }
    #print(true_dict["tp"].cpu().numpy().shape)
    #print(type_map.cpu().numpy().shape)
    if model.module.nr_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].cpu().numpy()
        result_dict["raw"]["pred_tp"] = type_map.cpu().numpy()
    return result_dict

####
def infer_step(batch_data, model,mode='test'):

    ####
    patch_imgs = batch_data['img']
    inst_map = batch_data["inst_map"]
    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()
    batch_bboxes = []
    batch_centers = []
    batch_edge_points = []
    batch_edge_indexs = []
    batch_pos_emb = []
    batch_length = []
    batch_select_shape_feats = []
    #print(inst_map.shape)
    model.eval()
    outputs = []
    with torch.no_grad():  # dont compute gradient
        for i in range(inst_map.shape[0]):
            single_map = inst_map[i,:,:].reshape(inst_map.shape[1],inst_map.shape[2])
            if single_map.max() > 0:
                centers,edge_points,edge_index,pos_emd = get_infer_bboxes(single_map,edge_num)
                #print(centers.shape,edge_points.shape,edge_index)
                #classes,centers,edge_points,edge_index,pos_emd = get_bboxes(single_map)
                #print(centers,edge_points,edge_index,classes)
                pred_dict = model(patch_imgs_gpu[i].unsqueeze(0),[centers],[edge_points],[edge_index],[pos_emd],[centers.shape[0]],mode='test')
                pred_map, pred_classes = pred_dict["tp"][0],pred_dict["tp"][1]
                pred_map = pred_map.permute(0, 2, 3, 1).contiguous()
        #pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        
                pred_map = F.softmax(pred_map, dim=-1)

                pre_class = F.softmax(pred_classes, dim=-1)
                prediction = torch.argmax(pre_class, 1).cpu().numpy()
                #print(single_map.shape,prediction.shape)
                output = add_class(single_map,prediction)
            #print(output.shape)
                outputs.append(output)
            else:
                output = np.zeros((single_map.shape[0],single_map.shape[1]))
                outputs.append(output)
    
    outputs = np.stack(outputs)
    #print(outputs)
    #print(outputs.shape)
    #print('prediction',prediction.max(),prediction.min())	
    
    #print('output',output.max(),output.min())
    # * Its up to user to define the protocol to process the raw output per step!
    return outputs


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    #true_np, pred_np = raw_data["np"]
    #true_hv, pred_hv = raw_data["hv"]

    true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_tp.shape), list(pred_tp.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        #true_viz_list.append(colorize(true_np[idx], 0, 1))
        #true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        #true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        #pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        #pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        #pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    #prob_np = raw_data["prob_np"]
    #print(len(prob_np))
    #true_np = raw_data["true_np"]
    #print(len(true_np))
    #for idx in range(len(raw_data["true_np"])):
        #patch_prob_np = prob_np[idx]
        #patch_true_np = true_np[idx]
        #patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        #inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        #correct = (patch_pred_np == patch_true_np).sum()
        #over_inter += inter
        #over_total += total
        #over_correct += correct
    #nr_pixels = len(true_np) * np.size(true_np[0])
    #acc_np = over_correct / nr_pixels
    #dice_np = 2 * over_inter / (over_total + 1.0e-8)
    #track_value("np_acc", acc_np, "scalar")
    #track_value("np_dice", dice_np, "scalar")

    # * TP statistic
    pred_tp = raw_data["pred_tp"]
    true_tp = raw_data["true_tp"]
    #print(len(true_tp))
    #print(pred_tp.shape)
    for type_id in range(0, nr_types):
        over_inter = 0
        over_total = 0
        for idx in range(len(raw_data["true_tp"])):
            patch_pred_tp = pred_tp[idx]
            patch_true_tp = true_tp[idx]
            inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
            over_inter += inter
            over_total += total
        dice_tp = 2 * over_inter / (over_total + 1.0e-8)
        track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # * HV regression statistic
    #pred_hv = raw_data["pred_hv"]
    #true_hv = raw_data["true_hv"]

    #over_squared_error = 0
    #for idx in range(len(raw_data["true_np"])):
     #   patch_pred_hv = pred_hv[idx]
      #  patch_true_hv = true_hv[idx]
       # squared_error = patch_pred_hv - patch_true_hv
        #squared_error = squared_error * squared_error
        #over_squared_error += squared_error.sum()
    #mse = over_squared_error / nr_pixels
    #track_value("hv_mse", mse, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    #true_np = np.array([true_np[idx] for idx in selected_idx])
    #true_hv = np.array([true_hv[idx] for idx in selected_idx])
    #prob_np = np.array([prob_np[idx] for idx in selected_idx])
    #pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs}


    true_tp = np.array([true_tp[idx] for idx in selected_idx])
    pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
    viz_raw_data["tp"] = (true_tp, pred_tp)
    #print('1111111',viz_raw_data)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict
