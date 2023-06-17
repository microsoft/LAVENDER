from utils.lib import *
from dataset import get_tsv_dls, get_dl, MetaLoader
from model_for_captioning import CaptioningLoss, LAVENDER_Captioning
from utils.args import get_args
from utils.logger import LOGGER, RunningMeter, add_log_to_file
from utils.dist import (
    is_main_process, all_gather,
    get_rank, get_world_size, NoOp)
from main_caption import Dataset_Caption, Agent_Captioning
from main_retrieval_mlm import Dataset_Retrieval_MLM
from main_qamc_mlm import Dataset_QAMC_MLM
from main_qaoe_mlm import Dataset_QAOE_MLM
from main_qaoe_mlm_lsmdc_fib import Dataset_QAOE_LSMDC_TSV
from main_retmc_mlm import Dataset_RetMC_MLM
import copy


def get_meta_dataloaders(args, tokzr):
    meta_dl_tr, meta_dl_ts, meta_dl_vl = {}, {}, {}
    len_dl_tr = 0
    datasets_args = args.datasets
    LOGGER.info(f"In total {len(datasets_args)} datasets: {datasets_args}")
    mc_ans_tok_ids = None
    for d_args in datasets_args:
        d_type = d_args.type
        task = f'{d_type}_{d_args.task}'
        d_full_args = copy.deepcopy(args)
        d_full_args.update(d_args)
        LOGGER.info(f"Loading task {task} with args {d_full_args}")
        if d_type == "retrieval":
            dl_tr, dl_vl, dl_ts = get_tsv_dls(
                d_full_args, Dataset_Retrieval_MLM, tokzr=tokzr)
        elif d_type == "qaoe":
            if 'lsmdc-fib' in task:
                d_cls = Dataset_QAOE_LSMDC_TSV
            else:
                d_cls = Dataset_QAOE_MLM
            d_full_args.size_vocab = -1  # use shared answer vocab
            dl_tr, dl_vl, dl_ts = get_tsv_dls(
                d_full_args, d_cls, tokzr=tokzr)
        elif d_type == "qamc":
            if 'lsmdc-mc' in task:
                d_cls = Dataset_RetMC_MLM
            else:
                d_cls = Dataset_QAMC_MLM
            dl_tr, dl_vl, dl_ts = get_tsv_dls(
                d_full_args, d_cls, tokzr=tokzr)
            if d_cls == Dataset_QAMC_MLM:
                mc_ans_tok_ids = dl_ts.dataset.ans_tok_ids
        elif d_type == "captioning":
            ds_tr = Dataset_Caption(
                d_full_args, d_full_args.train_yaml, 'train', tokzr=tokzr)
            dl_tr = get_dl(
                ds_tr, d_full_args, collate_fn=ds_tr.collate_batch)
            ds_vl = Dataset_Caption(
                d_full_args, d_full_args.val_yaml, 'val', tokzr=tokzr)
            dl_vl = get_dl(
                ds_vl, d_full_args, collate_fn=ds_vl.collate_batch)
            if "test_yaml" in d_full_args:
                ds_ts = Dataset_Caption(
                    d_full_args, d_full_args.test_yaml, 'test', tokzr=tokzr)
                dl_ts = get_dl(
                    ds_ts, d_full_args, collate_fn=ds_vl.collate_batch)
            else:
                dl_ts = None
        else:
            raise NotImplementedError(f"failed to load data for {task}")
        meta_dl_tr[task] = dl_tr
        meta_dl_vl[task] = dl_vl
        len_dl_tr += len(dl_tr)
        if dl_ts is not None:
            meta_dl_ts[task] = dl_ts
    return (
        meta_dl_tr, meta_dl_vl, meta_dl_ts, len_dl_tr,
        mc_ans_tok_ids)


class LAVENDER_Multi_Task(LAVENDER_Captioning):
    def __init__(self, args, tokzr, is_decoder=True):
        super().__init__(args, tokzr, is_decoder)

    def forward(self, batch, is_decode=False):
        batch = defaultdict(lambda: None, batch)
        task = batch["task"]
        batch["attn_mask_type"] = "full"
        if "captioning" in task:
            batch["attn_mask_type"] = "seq2seq"
            out = self.forward_captioning(batch, is_decode=is_decode)
        elif "retrieval" in task:
            out = self.forward_retrieval(batch)
        elif "qamc" in task:
            if 'lsmdc-mc' in task:
                out = self.forward_qamc_ret(batch)
            else:
                out = self.forward_qamc(batch)
        elif "qaoe" in task:
            out = self.forward_qaoe(batch)
        else:
            raise NotImplementedError(f"forward() for {task}")
        if "captioning" not in task:
            return {"out": out[0], "ans": out[1]}
        else:
            return out

    def forward_captioning(self, batch, is_decode=False):
        return super().forward(batch, is_decode=is_decode)

    def forward_retrieval(self, batch):
        img, txt, mask, vid = [
                batch[key] for key in [
                    "img", "txt", "mask", "vid"]]
        (_B, _T, _, _H, _W) = img.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)

        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        mtm_ans = []
        for i in range(_B):
            for j in range(_B):
                mt = mask_txt[j]
                t = txt[j]
                ft = feat_txt[j]
                t, mt, ft = self.prepro_txt_inputs(
                    t, mt, ft, task_name=batch["task_name"],
                    prompt=batch["prompt"])
                pdt_feat_img.append(feat_img[i].unsqueeze(0))
                pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(ft.unsqueeze(0))
                pdt_mask_txt.append(mt.unsqueeze(0))
                gt_txt = T.ones_like(t)*-1
                if vid[i] == vid[j]:
                    gt_txt[-1] = self.true_token_id
                else:
                    gt_txt[-1] = self.false_token_id
                mtm_ans.append(gt_txt.unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt, mtm_ans = [
            T.cat(x, dim=0)
            for x in [pdt_feat_img, pdt_mask_img,
                      pdt_feat_txt, pdt_mask_txt, mtm_ans]
            ]
        out, _ = self.go_cross(
            pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])

        return out, mtm_ans

    def forward_qamc_ret(self, batch):
        img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["mask_ans"]

        (_B, _T, _, _H, _W), (_, _O, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt.flatten(0, 1), mask.flatten(0, 1))
        feat_img, mask_img = [
            feat_img.unsqueeze(1).expand([-1, _O, -1, -1]).flatten(0, 1),
            mask_img.unsqueeze(1).expand([-1, _O, -1]).flatten(0, 1)]
        _B, _O, _L = ans.shape
        ans = ans.flatten(0, 1)
        prompt = batch["prompt"]
        ans, mask_txt, feat_txt = self.prepro_txt_inputs(
            ans, mask_txt, feat_txt, task_name=batch["task_name"],
            prompt=prompt)
        if prompt is not None and self.args.enable_prompt:
            _L = len(prompt[0])
        elif self.args.enable_task_token:
            _L = 1  # for a task token
        else:
            _L = 0  # no added task token or prompt
        ans[:, :_L] = -1
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        ans = ans.view(_B, _O, -1)
        return out, ans

    def forward_qamc(self, batch):
        img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["mask_ans"]

        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt, mask)
        prompt = batch["prompt"]
        ans, mask_txt, feat_txt = self.prepro_txt_inputs(
            ans, mask_txt, feat_txt, task_name=batch["task_name"],
            prompt=prompt)
        if prompt is not None and self.args.enable_prompt:
            _L = len(prompt[0])
        elif self.args.enable_task_token:
            _L = 1  # for a task token
        else:
            _L = 0  # no added task token or prompt
        ans[:, :_L] = -1
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        return out, ans

    def forward_qaoe(self, batch):
        img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["mask_ans"]
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        prompt = batch["prompt"]
        ans, mask_txt, feat_txt = self.prepro_txt_inputs(
            ans, mask_txt, feat_txt, task_name=batch["task_name"],
            prompt=prompt)
        if prompt is not None and self.args.enable_prompt:
            _L = len(prompt[0])
        elif self.args.enable_task_token:
            _L = 1  # for a task token
        else:
            _L = 0  # no added task token or prompt
        ans[:, :_L] = -1
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        return out, ans


class Agent_Multi_Task(Agent_Captioning):
    def __init__(self, args, model, mc_ans_tok_ids):
        super(Agent_Captioning, self).__init__(args, model)
        cap_loss_config = {
            'label_smoothing': getattr(args, 'label_smoothing', 0),
            'drop_worst_ratio': getattr(args, 'drop_worst_ratio', 0),
            'drop_worst_ratio': getattr(args, 'drop_worst_ratio', 0)}
        self.non_cap_loss_func = T.nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.cap_loss_func = CaptioningLoss(cap_loss_config).cuda()
        self.log = defaultdict(list)
        self.task2loss = {}
        self.task2acc = {}
        self.mc_ans_tok_ids = mc_ans_tok_ids

    def meter_loss(self, task, ls):
        key = f'ls_{task}'
        if key not in self.task2loss:
            self.task2loss[key] = RunningMeter(key)
        self.task2loss[key](ls)

    def meter_acc(self, task, acc):
        key = f'ac_{task}'
        if key not in self.task2acc:
            self.task2acc[key] = RunningMeter(key)
        self.task2acc[key](acc)

    def add_prompt_or_task_token(self, batch, dl):
        task = batch["task"]
        if self.args.enable_prompt:
            if isinstance(dl, MetaLoader):
                batch["prompt"] = dl.name2loader[task].dataset.get_prompt()
            else:
                batch["prompt"] = dl.dataset.get_prompt()
        elif self.args.enable_task_token:
            if 'retrieval' in task:
                batch["task_name"] = "vtm"
            elif 'qamc' in task:
                if 'lsmdc-mc' in task:
                    batch["task_name"] = "vtm"
                else:
                    batch["task_name"] = "mc"
            elif 'qaoe' in task:
                batch["task_name"] = "oe"
            elif 'captioning' in task:
                batch["task_name"] = "cap"
            else:
                raise NotImplementedError(f"no task name for {task}")
        return batch

    def cap_evaluate(self, ep, val_dataloader):
        self.model.eval()
        result = super().evaluate(ep, val_dataloader)
        self.model.train()
        return result

    def non_cap_evaluate(self, task, dl):
        self.model.eval()
        ret = defaultdict(list)
        for _, batch in enumerate(dl):
            batch["task"] = task
            batch = self.add_prompt_or_task_token(batch, dl)
            batch = self.prepare_batch(batch)
            r = self.eval_step(batch)
            ret = {
                k: ret[k]+l if isinstance(l, list) else ret[k]+[l]
                for k, l in r.items()}

        gathered_ret = defaultdict(list)
        for ret_per_rank in all_gather(ret):
            for k in ret_per_rank:
                gathered_ret[k].extend(ret_per_rank[k])
        ret_all = {
            k: float(np.average(gathered_ret[k])) for k in ret}
        self.model.train()
        return ret_all

    def get_top_k_acc(self, out, ans, k=5):
        _B = out.shape[0]
        # out_mtm = T.argmax(out, dim=-1)
        ans_mtm = ans[ans != -1].view(-1, 1)
        n_valid_ans = ans_mtm.shape[0]
        out_mtm = out[ans != -1].view(n_valid_ans, -1)
        _, out_mtm_i = T.topk(out_mtm, k=k, dim=-1)
        ac = (out_mtm_i == ans_mtm).any(dim=-1).float().tolist()
        if len(ac) < _B:
            ac += [0.] * (_B - len(ac))
        return ac

    def eval_step(self, batch):
        self.model.eval()
        task = batch["task"]
        out = self.forward_step(batch)
        out, ans = out["out"], out["ans"]
        if "retrieval" in task:
            _B = len(batch["vid"])
            p_true = out[:, :, self.true_token_id]
            p_false = out[:, :, self.false_token_id]
            out_mtm = p_true / (p_true+p_false)
            ans_mtm = ans
            out_mtm = out_mtm[ans_mtm != -1].view(_B, _B)
            ans_mtm = ans_mtm[ans_mtm != -1].view(_B, _B)
            out_mtm = T.argmax(out_mtm, dim=-1)
            ans_mtm_idx = (ans_mtm == self.true_token_id).nonzero()[:, 1]
            ac = (out_mtm == ans_mtm_idx).float().tolist()
            return {'ac': ac}
        elif "qaoe" in task:
            ac_1 = self.get_top_k_acc(out, ans, k=1)
            ac_5 = self.get_top_k_acc(out, ans, k=5)
            return {'ac_1': ac_1, 'ac_5': ac_5}
        elif "qamc" in task:
            if "lsmdc-mc" in task:
                _B, _O, _L = ans.shape
                p_true = out[:, :, self.true_token_id]
                p_false = out[:, :, self.false_token_id]
                out_mtm = p_true / (p_true+p_false)
                ans_mtm = ans.view(_B*_O, _L)
                assert ans_mtm.shape == out_mtm.shape
                out_mtm = out_mtm[ans_mtm != -1].view(_B, _O)
                ans_mtm = ans_mtm[ans_mtm != -1].view(_B, _O)
                out_mtm = T.argmax(out_mtm, dim=-1)
                ans_mtm_idx = (ans_mtm == self.true_token_id).nonzero()[:, 1]
                ac = (out_mtm == ans_mtm_idx).float().tolist()
                return {'ac': ac}
            # other mc
            _B, _ = ans.shape
            p_all_ans_toks = out[:, :, self.mc_ans_tok_ids]
            ans_mtm = ans
            out_mtm = p_all_ans_toks[ans_mtm != -1]
            out_mtm = out_mtm / out_mtm.sum(dim=-1).view(_B, 1)
            out_mtm = out_mtm.view(_B, -1)
            out_mtm = T.argmax(out_mtm, dim=-1)
            ans_idx = batch["ans_idx"]
            ac = (out_mtm == ans_idx).float().tolist()
            return {'ac': ac}

    def evaluate(self, ep, task, dl):
        if "captioning" in task:
            return self.cap_evaluate(ep, dl)
        else:
            return self.non_cap_evaluate(task, dl)

    def train_step(self, batch):
        self.model.train()
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            logits, ans = out["out"], out["ans"]
            task = batch["task"]
            if "captioning" in task:
                ls = self.cap_loss_func(
                    logits[ans != -1].float(), ans[ans != -1])
            else:
                logits = logits.flatten(
                    0, len(logits.shape)-2)
                ans = ans.flatten(
                    0, len(ans.shape)-1)
                ls = self.loss_func(logits, ans)

            self.backward_step(ls)
            pred = T.argmax(logits, dim=-1)
            acc = (
                float((pred == ans).sum() / (ans != -1).sum())
                if (ans != -1).sum() > 0 else 0)
            return {'ls': ls.item(), 'ac': acc}

    def log_train(self, ep, step):
        log_info = self.log_memory(ep, step)
        log_info += "\n\t"
        for task, rm in self.task2loss.items():
            ls_tr = rm.val
            log_info += f" {task}: {ls_tr:.2e}"
        log_info += "\n\t"
        for task, rm in self.task2acc.items():
            ac_tr = rm.val
            log_info += f" {task}: {ac_tr*100:.2f}"
        return log_info

    def run(self, meta_dl_tr, meta_dl_vl, meta_dl_ts):
        LOGGER.info("Start training....")
        step = 0

        for step, (task, batch) in enumerate(meta_dl_tr):
            ep = (step // self.args.iter_per_ep + 1)
            if step % self.args.logging_steps == 0:
                LOGGER.info(self.log_train(ep, step))
            batch["task"] = task
            if "captioning" in task:
                masked_batch = self.masking(
                    batch['txt'], p_mask=self.args.p_mask)
                batch.update(masked_batch)
            batch = self.add_prompt_or_task_token(batch, meta_dl_tr)
            batch = self.prepare_batch(batch)
            out = self.train_step(batch)
            ls, ac = out['ls'], out['ac']
            self.meter_loss(task, ls)
            self.meter_acc(task, ac)
            if step % self.args.iter_per_ep == 0 and step:
                for task, dl_vl in meta_dl_vl.items():
                    res_vl = self.evaluate(ep, task, dl_vl)
                    for k in res_vl:
                        self.log[f'{task}_vl_{k}'].append(res_vl[k])
                    LOGGER.info(f'Ep {ep} {task} vl: {json.dumps(res_vl)}')
                for task, dl_ts in meta_dl_ts.items():
                    res_ts = self.evaluate(ep, task, dl_ts)
                    for k in res_ts:
                        self.log[f'{task}_ts_{k}'].append(res_ts[k])
                    LOGGER.info(f'Ep {ep} {task} ts: {json.dumps(res_ts)}')
                self.save_model(ep)
            if step >= self.args.max_iter:
                break

        if step % self.args.logging_steps != 0:
            LOGGER.info(self.log_train(ep, step))

        if step % self.args.iter_per_ep != 0:
            for task, dl_vl in meta_dl_vl.items():
                res_vl = self.evaluate(step, task, dl_vl)
                for k in res_vl:
                    self.log[f'{task}_vl_{k}'].append(res_vl[k])
                LOGGER.info(
                    f'Last step {step} {task} vl: {json.dumps(res_vl)}')
            for task, dl_ts in meta_dl_ts.items():
                res_ts = self.evaluate(ep, task, dl_ts)
                for k in res_ts:
                    self.log[f'{task}_ts_{k}'].append(res_ts[k])
                LOGGER.info(
                    f'Last step {step} {task} ts: {json.dumps(res_ts)}')
            self.save_model(step)

        for task in meta_dl_vl.keys():
            if 'captioning' in task:
                metric = 'CIDEr'
            elif 'qaoe' in task:
                metric = 'ac_1'
            else:
                metric = 'ac'
            best_vl = self.best_epoch(task, 'vl', metric)
            LOGGER.info(
                f'Best {metric} on {task} val @ ep {best_vl[0]},'
                f' {best_vl[1]*100:.2f}')
        for task in meta_dl_ts.keys():
            if 'captioning' in task:
                metric = 'CIDEr'
            elif 'qaoe' in task:
                metric = 'ac_1'
            else:
                metric = 'ac'
            best_vl = self.best_epoch(task, 'ts', metric)
            LOGGER.info(
                f'Best {metric} on {task} test @ ep {best_vl[0]},'
                f' {best_vl[1]*100:.2f}')
        return

    def best_epoch(self, task, split, metric):
        if not hasattr(self, "log"):
            raise NotImplementedError("no log to find the best epoch")
        val_index = np.argmax(
            self.log[f"{task}_{split}_{metric}"])
        val_max = self.log[f"{task}_{split}_{metric}"][val_index]
        return (val_index, val_max)


if __name__ == '__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    (meta_dl_tr, meta_dl_vl, meta_dl_ts,
     len_dl_tr, mc_ans_tok_ids) = get_meta_dataloaders(args, tokzr=tokzr)
    meta_dl_tr = MetaLoader(meta_dl_tr, distributed=args.distributed)
    if args.size_epoch == 0:
        args.max_iter = 1
    else:
        args.max_iter = (
            len_dl_tr * args.size_epoch)  # estimated
    args.iter_per_ep = len_dl_tr

    model = LAVENDER_Multi_Task(
        args, tokzr,
        is_decoder=getattr(args, 'is_decoder', False))
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed:
        LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                    f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s' % (
        args.path_output, args.task,
        datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_Multi_Task(
        args, model, mc_ans_tok_ids=mc_ans_tok_ids)
    if args.distributed:
        agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process():
        add_log_to_file('%s/stdout.txt' % (args.path_output))
    else:
        LOGGER = NoOp()

    LOGGER.info("Zero shot evaluation ...")
    for task, dl_vl in meta_dl_vl.items():
        res_vl = agent.evaluate(0, task, dl_vl)
        for k in res_vl:
            agent.log[f'{task}_vl_{k}'].append(res_vl[k])
        LOGGER.info(f'Ep 0 {task} vl: {json.dumps(res_vl)}')
    for task, dl_ts in meta_dl_ts.items():
        res_ts = agent.evaluate(0, task, dl_ts)
        for k in res_ts:
            agent.log[f'{task}_ts_{k}'].append(res_ts[k])
        LOGGER.info(f'Ep 0 {task} ts: {json.dumps(res_ts)}')

    if args.size_epoch:
        agent.run(meta_dl_tr, meta_dl_vl, meta_dl_ts)
