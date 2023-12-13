
import torch
import logging
import os
import copy
from misc.utils import log_info
from docopt import docopt
from compute_stats import run_nuclei_type_stat
#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    nr_gpus = torch.cuda.device_count()
    log_info('Detect #GPUS: %d' % nr_gpus)
    

    # ***
    run_args = {
        'batch_size' : 8,

        'nr_inference_workers' : 8,
        'nr_post_proc_workers' : 16,
    }


    run_args['patch_input_shape'] = 256
    run_args['patch_output_shape'] = 256

    run_args.update({
        'input_dir'      : '/media/louwei@sribd.cn/T7 Shield/BRCA/test/images/',
        'output_dir'     : 'brca_output/',
        'inst_dir'    : '/media/louwei@sribd.cn/T7 Shield/BRCA/mcspatnet_predictions/',
        'mem_usage'   : 0.1,
        'draw_dot'    : True,
        'save_qupath' : True,
        'save_raw_map': True,
    })

    from infer.tile import InferManager
    for i in range(10,50,1):
        print('epoch: ',i)
        method_args = {
            'method' : {
                'model_args' : {
                    'nr_types'   : 4,
                    'mode'       : 'original' ,
                },
                'model_path' : 'logs/00/' + 'net_epoch=' + str(i) + '.tar',
            },
            'type_info_path'  : 'type_info.json'
        }
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
        run_nuclei_type_stat('brca_output/mat/','/media/louwei@sribd.cn/T7 Shield/BRCA/gt_mat/')

