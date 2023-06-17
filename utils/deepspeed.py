from .logger import LOGGER as logger
from pprint import pformat
import torch


def get_deepspeed_config(args):
        config_params = {
            'train_batch_size': args.effective_batch_size,
        }

        use_fp16 = args.deepspeed  # args.deepspeed_fp16
        use_amp = not args.deepspeed  # not args.deepspeed_fp16 
        # by default, if not use deepspeed fp16, will enable deepspeed amp 

        if use_amp:
            config_params['amp'] = {
                'enabled': True,
                'opt_level': 'O2',
            }

        if use_fp16:
            config_params['fp16'] = {
                'enabled': True,
            }

        gradient_clip = args.max_grad_norm
        if gradient_clip > 0:
            config_params['gradient_clipping'] = gradient_clip

        config_params['flops_profiler'] = {
            'enabled': False,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 3,
            'detailed': True,
        }

        config_params['logging'] = {
            'steps_per_print': args.logging_steps*10,
        }
        # if hasattr(args, "zero_opt_stage") and args.zero_opt_stage > 0:
        config_params['zero_optimization'] = {
            'stage': 1,
        }
        # if args.zero_opt_stage > 0:
        #     config_params['fp16'] = {
        #         'enabled': True
        #     }
        config_params['zero_allow_untested_optimizer'] = True

        logger.info(pformat(config_params))
        return config_params


def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch
