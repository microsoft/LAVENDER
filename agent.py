from dataset import move_to_cuda
from utils.lib import *
from utils.dist import (
    is_main_process, get_world_size,
    reduce_dict, get_local_rank)
from utils.misc import humanbytes
from utils.deepspeed import get_deepspeed_config, fp32_to_fp16
import deepspeed
from torch import nn
import torch.nn.functional as F


class WarmupLinearLR(T.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=1e-8,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio*max_iter)
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        tot_step = self.max_iter
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        elif step > tot_step:
            step = tot_step
        return max(0, (tot_step-step)/(tot_step-warmup_step))

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [
            max(self.min_lr, base_lr * warmup_factor)
            for base_lr in self.base_lrs
        ]


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], "
        "computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        ipos = T.diag(i_logsm)
        loss_i = ipos.sum() / len(ipos)

        jpos = T.diag(j_logsm)
        loss_j = jpos.sum() / len(jpos)

        return - loss_i - loss_j


class Agent_Base:
    def __init__(self, args, model):
        super().__init__()
        self.args, self.model = args, model
        self.loss_func = T.nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.optzr = self.build_optimizer()
        self.lr_scheduler = WarmupLinearLR(
            self.optzr, args.max_iter)
        self.scaler = T.cuda.amp.GradScaler()
        self.log = None
        if hasattr(model, 'tokzr') and self.model.tokzr is not None:
            self.tokzr = self.model.tokzr
        else:
            self.tokzr = transformers.AutoTokenizer.from_pretrained(
                args.tokenizer)
        (self.cls_token_id, self.sep_token_id,
            self.pad_token_id, self.mask_token_id,
            self.unk_token_id) = self.tokzr.convert_tokens_to_ids(
            [self.tokzr.cls_token,
                self.tokzr.sep_token, self.tokzr.pad_token,
                self.tokzr.mask_token,
                self.tokzr.unk_token])
        self.true_token_id = self.tokzr.convert_tokens_to_ids(
            ["true"])[0]
        self.false_token_id = self.tokzr.convert_tokens_to_ids(
            ["false"])[0]
        self.global_step = 0

    def build_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [
            (n, p) for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [
            (n, p) for n, p in param_optimizer
            if any(nd in n for nd in no_decay)]

        decay_swin_param = [
            (n, p) for n, p in decay_param_tp
            if "swin." in n]
        decay_other_param = [
            (n, p) for n, p in decay_param_tp
            if "swin." not in n]

        no_decay_swin_param = [
            (n, p) for n, p in no_decay_param_tp
            if "swin." in n]
        no_decay_other_param = [
            (n, p) for n, p in no_decay_param_tp
            if "swin." not in n]

        weight_decay = self.args.decay
        coef_lr = self.args.vis_backbone_lr_mul
        lr = self.args.lr
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_swin_param],
             'weight_decay': weight_decay,
             'lr': lr * coef_lr},
            {'params': [p for n, p in decay_other_param],
             'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_swin_param],
             'weight_decay': 0.0,
             'lr': lr * coef_lr},
            {'params': [p for n, p in no_decay_other_param],
             'weight_decay': 0.0}
        ]

        optzr = T.optim.AdamW(
            optimizer_grouped_parameters, lr=lr,
            betas=(0.9, 0.98), weight_decay=weight_decay)
        return optzr

    def reduce_dict(self, data):
        return reduce_dict(data)

    def reduce_mean(self, v):
        world_size = get_world_size()
        if world_size < 2:
            return v
        else:
            v = T.tensor(v).cuda()
            DIST.all_reduce(v)
            v = v.item() / world_size
        return v

    def save_training_meta(self):
        if is_main_process():
            os.makedirs(self.args.path_output, exist_ok=True)
            print(self.args)
            json.dump(
                self.args,
                open(f'{self.args.path_output}/args.json', 'w'), indent=2)
            self.save_model(0)

    def save_model(self, ep):
        if is_main_process():
            output_dir = self.args.path_output
            # save gaurd output_dir
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = self.model.module if hasattr(
                self.model, 'module') else self.model
            state_dict = {
                k: v.cpu() if isinstance(v, T.Tensor) else v
                for k, v in model_to_save.state_dict().items()}
            T.save(
                state_dict,
                f"{output_dir}/ckpt_violet_{self.args.task}_{ep}.pt")
            if self.log is not None:
                json.dump(
                    self.log,
                    open(f"{output_dir}/log.json", 'w'), indent=2)
        # DIST.barrier()
        # synchronize()

    def log_memory(self, ep=-1, step=-1):
        if ep == -1 and step == -1:
            step = self.global_step
            step_str = f"global step: {step},"
        else:
            step_str = f"ep: {ep}, step: {step},"

        memory = humanbytes(T.cuda.max_memory_allocated())
        lr_swin = f'{self.optzr.param_groups[0]["lr"]:.2e}'
        lr_bert = f'{self.optzr.param_groups[1]["lr"]:.2e}'
        return f"{step_str} lr_swin: {lr_swin}, " +\
            f"lr_bert: {lr_bert}, max memory: {memory}"

    def prepare_batch(self, batch):
        batch = move_to_cuda(batch)
        if self.args.deepspeed:
            batch = fp32_to_fp16(batch)
        return batch

    def forward_step(self, batch):
        if self.args.deepspeed:
            if isinstance(batch, dict):
                model = self.model.module if hasattr(
                    self.model, 'module') else self.model
                named_params = inspect.getargspec(model.forward).args
                if "batch" in named_params:
                    out = self.model(batch)
                else:
                    out = self.model(**batch)
            elif isinstance(batch, tuple):
                out = self.model(*batch)
            else:
                raise TypeError(
                    f"batch is either dict or tuple, {type(batch)}")
        else:
            with T.cuda.amp.autocast(enabled=True):
                if isinstance(batch, dict):
                    model = self.model.module if hasattr(
                        self.model, 'module') else self.model
                    named_params = inspect.getargspec(model.forward).args
                    if "batch" in named_params:
                        out = self.model(batch)
                    else:
                        out = self.model(**batch)
                elif isinstance(batch, tuple):
                    out = self.model(*batch)
                else:
                    raise TypeError(
                        f"batch is either dict or tuple, {type(batch)}")
        return out

    def backward_step(self, loss):
        if self.args.deepspeed:
            self.model.backward(loss)
            self.model.step()
        else:
            self.scaler.scale(loss).backward()
            if self.args.max_grad_norm > 0:
                self.scaler.unscale_(self.optzr)
                # Since the gradients of optimizer's assigned
                # params are unscaled, clips as usual:
                T.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)
            self.scaler.step(self.optzr)
            self.scaler.update()
            self.lr_scheduler.step()
            self.optzr.zero_grad()

    def prepare_dist_model(self):
        if self.args.deepspeed:
            config = get_deepspeed_config(self.args)
            self.model, self.optzr, _, _ = deepspeed.initialize(
                config_params=config,
                model=self.model,
                optimizer=self.optzr,
                lr_scheduler=self.lr_scheduler)
        else:
            self.model = T.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=True)

    def best_epoch(self):
        if not hasattr(self, "log"):
            raise NotImplementedError("no log to find the best epoch")
        if "ac_vl" not in self.log or "ac_ts" not in self.log:
            raise ValueError("calling best_epoch in pretraining, maybe?")
        val_index = np.argmax(self.log["ac_vl"])
        test_index = np.argmax(self.log["ac_ts"])
        val_max = self.log["ac_vl"][val_index]
        test_max = self.log["ac_ts"][test_index]
        return (val_index, val_max), (test_index, test_max)
