import logging
import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import sys
import warnings
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool
import PIL.Image as Image
import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import SerializeArray, SerializeFileList,FileLoader
from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.viz_utils import colorize, visualize_instances_dict,visualize_instances_map
from skimage import color

import convert_format
from . import base


####
def _prepare_patching(img,inst_map, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.
    
    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img
        
    """

    win_size = window_size
    msk_size = step_size = mask_size
    #print(win_size,step_size)
    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w
    #print(img.shape)
    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "constant")
    #print(img.shape)
    inst_map = np.lib.pad(inst_map, ((padt, padb), (padl, padr)), "constant")
    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    
    #print(patch_info)
    if not return_src_top_corner:
        return img,inst_map, patch_info
    else:
        return img,inst_map, patch_info, [padt, padl]


####
def _post_process_patches(
    post_proc_func, post_proc_kwargs, patch_info, image_info, type_colour,
):
    """Apply post processing to patches.
    
    Args:
        post_proc_func: post processing function to use
        post_proc_kwargs: keyword arguments used in post processing function
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)
    
    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    src_shape = image_info["src_shape"]
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])
    
    src_image = image_info['src_image']
    inst_map = image_info['inst_map']
    #cv2.imwrite('inst_map.png', inst_map*255)
    #src_image = cv2.resize(src_image, (1000,1000), cv2.INTER_CUBIC)
    #print(inst_map.shape,inst_map.max())
    #inst_map = Image.fromarray(inst_map)
    #inst_map = inst_map.resize((1000,1000), resample=Image.NEAREST)
    #pred_map = Image.fromarray(pred_map)
    #pred_map = pred_map.resize((1000,1000), resample=Image.NEAREST)
    #inst_map = np.array(inst_map)
    #pred_map = np.array(pred_map)
    #cv2.imwrite('inst_map_resize.png', inst_map*255)
    #print(pred_map)
    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type
    type_map = post_proc_func(pred_map,**post_proc_kwargs)
    pred_inst, inst_info_dict = process(type_map,inst_map, **post_proc_kwargs)
    #type_colour = [[0  ,   0,   0], [255, 204, 204],[150,200,150], [160, 90, 160], [50, 50, 50], [255, 200, 0] ]
    overlaid_img = visualize_instances_dict(
        src_image.copy(), inst_info_dict,True,type_colour
    )

    return image_info["name"], type_map, pred_inst, inst_info_dict, overlaid_img
def process(pred_map,pred_inst,nr_types=None, return_centroids=True):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """
    pred_type = pred_map.astype(np.int32)


    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    #print(inst_info_dict)
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, inst_info_dict
'''
def visualize_instances_dict(input_image,type_map):
    number = 5
    print('33333',number,type_map.shape)
    color_map = [[0  ,   0,   0], [255, 204, 204],[150,200,150], [160, 90, 160], [50, 50, 50], [255, 200, 0] ]
    
    for i in range(1,number+1):
        mask = np.zeros((type_map.shape[0],type_map.shape[0]))
        #print('44444')
        mask[type_map==i] = 255
        #print('7777777')
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print('666666')
        cv2.drawContours(input_image, contours, -1, color_map[i], 3)

    #print(input_image.shape)

    return input_image
'''    
class InferManager(base.InferManager):
    """Run inference on tiles."""

    ####
    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        assert self.mem_usage < 1.0 and self.mem_usage > 0.0

        # * depend on the number of samples and their size, this may be less efficient
        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
        inst_path_list = glob.glob(patterning("%s/*" % self.inst_dir))
        file_path_list.sort() 
        inst_path_list.sort()        # ensure same order
        assert len(file_path_list) > 0, 'Not Detected Any Files From Path'
        
        rm_n_mkdir(self.output_dir + '/json/')
        rm_n_mkdir(self.output_dir + '/mat/')
        rm_n_mkdir(self.output_dir + '/overlay/')
        if self.save_qupath:
            rm_n_mkdir(self.output_dir + "/qupath/")
        '''
        def proc_callback(results):
            """Post processing callback.
            
            Output format is implicit assumption, taken from `_post_process_patches`

            """
            img_name, pred_map, overlaid_img,inst_map = results
            print(img_name)
            print(overlaid_img.shape)
            mat_dict = {
                "inst_map" : pred_inst,
                "type_map" : nuc_uid_list,
            }
            #if self.save_raw_map:
                #mat_dict["raw_map"] = pred_map
            #save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
            #sio.savemat(save_path, pred_map)

            save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

            return img_name
        '''
        def proc_callback(results):
            """Post processing callback.
            
            Output format is implicit assumption, taken from `_post_process_patches`

            """
            img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results

            nuc_val_list = list(inst_info_dict.values())
            # need singleton to make matlab happy
            nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
            nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
            nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
            #print(nuc_coms_list)
            mat_dict = {
                "inst_map" : pred_inst,
                "inst_uid" : nuc_uid_list,
                "inst_type": nuc_type_list,
                "inst_centroid": nuc_coms_list
            }
            if self.nr_types is None: # matlab does not have None type array
                mat_dict.pop("inst_type", None) 

            if self.save_raw_map:
                mat_dict["raw_map"] = pred_map
            save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
            sio.savemat(save_path, mat_dict)

            save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

            if self.save_qupath:
                nuc_val_list = list(inst_info_dict.values())
                nuc_type_list = np.array([v["type"] for v in nuc_val_list])
                nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
                save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
                convert_format.to_qupath(
                    save_path, nuc_coms_list, nuc_type_list, self.type_info_dict
                )

            save_path = "%s/json/%s.json" % (self.output_dir, img_name)
            self.__save_json(save_path, inst_info_dict, None)
            return img_name
        def detach_items_of_uid(items_list, uid, nr_expected_items):
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True:
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        while len(file_path_list) > 0:

            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, "available")
            available_ram = int(available_ram * self.mem_usage)
            # available_ram >> 20 for MB, >> 30 for GB

            # TODO: this portion looks clunky but seems hard to detach into separate func

            # * caching N-files into memory such that their expected (total) memory usage
            # * does not exceed the designated percentage of currently available memory
            # * the expected memory is a factor w.r.t original input file size and
            # * must be manually provided
            file_idx = 0
            use_path_list = []
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            cache_inst_list = []
            use_inst_path_list = []
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)
                inst_path = inst_path_list.pop(0)
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                src_shape = img.shape
                inst_map = sio.loadmat(inst_path)['inst_map']
                #####Patching
                img, inst_map,patch_info, top_corner = _prepare_patching(
                    img,inst_map, self.patch_input_shape, self.patch_output_shape, True
                )
                self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
                patch_info = np.concatenate([patch_info, self_idx[:, None]], axis=-1)
                # ? may be expensive op
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0:
                    break

                file_idx += 1
                # if file_idx == 4: break
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                cache_inst_list.append(inst_map)
                cache_image_info_list.append([src_shape, len(patch_info), top_corner])
                # TODO: refactor to explicit protocol

            # * apply neural net on cached data
            #dataset = FileLoader(
                #cache_image_list, cache_inst_list
            #)
            dataset = SerializeFileList(
                cache_image_list,cache_inst_list, cache_patch_info_list, self.patch_input_shape
            )

            dataloader = data.DataLoader(
                dataset,
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
                drop_last=False,
            )

            pbar = tqdm.tqdm(
                desc="Process Patches",
                leave=True,
                total=int(len(cache_image_list) / self.batch_size) + 1,
                ncols=80,
                ascii=True,
                position=0,
            )

            #accumulated_patch_output = []
            #for batch_idx, batch_data in enumerate(dataloader):
                #print(batch_data)
                #sample_output_list = self.run_step(batch_data)
                #print(sample_output_list.shape)
                #accumulated_patch_output.append(sample_output_list)
                #pbar.update()
            #pbar.close()
            accumulated_patch_output = []
            for batch_idx, batch_data in enumerate(dataloader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self.run_step(sample_data_list)
                sample_info_list = sample_info_list.numpy()
                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(
                    sample_output_list, curr_batch_size, axis=0
                )
                #print(sample_info_list,curr_batch_size)
                sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
                sample_output_list = list(zip(sample_info_list, sample_output_list))
                accumulated_patch_output.extend(sample_output_list)
                pbar.update()
            pbar.close()

            # * parallely assemble the processed cache data for each file if possible
            future_list = []
            for file_idx, file_path in enumerate(use_path_list):
                #img = cv2.imread(file_path)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #inst_map = sio.loadmat(use_inst_path_list[file_idx])['inst_map']
                #type_map = accumulated_patch_output[file_idx]
                #print(type_map.shape)
                image_info = cache_image_info_list[file_idx]
                file_ouput_data, accumulated_patch_output = detach_items_of_uid(
                    accumulated_patch_output, file_idx, image_info[1]
                )
                #print(file_ouput_data.shape,accumulated_patch_output.shape)
                src_pos = image_info[2]  # src top left corner within padded image
                src_image = cache_image_list[file_idx]
                src_image = src_image[
                    src_pos[0] : src_pos[0] + image_info[0][0],
                    src_pos[1] : src_pos[1] + image_info[0][1],
                ]
                
                src_inst = cache_inst_list[file_idx]
                src_inst = src_inst[
                    src_pos[0] : src_pos[0] + image_info[0][0],
                    src_pos[1] : src_pos[1] + image_info[0][1],
                ]

                base_name = pathlib.Path(file_path).stem
                file_info = {
                    "src_shape": image_info[0],
                    "src_image": src_image,
                    "inst_map":src_inst,
                    "name": base_name,
                }

                post_proc_kwargs = {
                    "nr_types": self.nr_types,
                    "return_centroids": True,
                }  # dynamicalize this

                overlay_kwargs = {
                    "draw_dot": self.draw_dot,
                    "type_colour": self.type_info_dict,
                    "line_thickness": 2,
                }
                func_args = (
                    self.post_proc_func,
                    post_proc_kwargs,
                    file_ouput_data,
                    file_info,
                    overlay_kwargs,
                )
                #print(type_map.shape)
                name, pred_map, pred_inst, inst_info_dict, overlaid_img = _post_process_patches(self.post_proc_func, post_proc_kwargs, file_ouput_data, file_info, self.type_info_dict)
                # dispatch for parallel post-processing

                file_path = proc_callback([name, pred_map, pred_inst, inst_info_dict, overlaid_img])
                log_info("Done Assembling %s" % file_path)
        return

