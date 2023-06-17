from utils.lib import *
from dataset import get_tsv_dls, get_dl, MetaLoader
from model_for_captioning import CaptioningLoss
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import (
    is_main_process, all_gather,
    get_rank, get_world_size, NoOp)
from main_caption import Dataset_Caption, Agent_Captioning
from main_retrieval_task_specific import Dataset_Retrieval_TS
from main_qamc_task_specific import Dataset_QAMC_TS
from main_retmc_task_specific import Dataset_RetMC_TS
from main_qaoe_task_specific import Dataset_QAOE_TS
import copy
from main_multi_task_mlm import Agent_Multi_Task, LAVENDER_Multi_Task
from agent import NormSoftmaxLoss


def get_meta_dataloaders(args, tokzr):
    meta_dl_tr, meta_dl_ts, meta_dl_vl = {}, {}, {}
    len_dl_tr = 0
    datasets_args = args.datasets
    LOGGER.info(f"In total {len(datasets_args)} datasets: {datasets_args}")
    mc_ans_tok_ids = None
    multi_task_size_vocab = {}
    multi_task_size_option = {}
    for d_args in datasets_args:
        d_type = d_args.type
        task = f'{d_type}_{d_args.task}'
        d_full_args = copy.deepcopy(args)
        d_full_args.update(d_args)
        LOGGER.info(f"Loading task {task} with args {d_full_args}")
        if d_type == "retrieval":
            dl_tr, dl_vl, dl_ts = get_tsv_dls(
                d_full_args, Dataset_Retrieval_TS, tokzr=tokzr)
        elif d_type == "qaoe":
            dl_tr, dl_vl, dl_ts = get_tsv_dls(
                d_full_args, Dataset_QAOE_TS, tokzr=tokzr)
            multi_task_size_vocab[task] = d_full_args.size_vocab
        elif d_type == "qamc":
            if 'lsmdc-mc' in task:
                d_cls = Dataset_RetMC_TS
            else:
                d_cls = Dataset_QAMC_TS                
                multi_task_size_option[task] = d_full_args.size_option
            dl_tr, dl_vl, dl_ts = get_tsv_dls(
                d_full_args, d_cls, tokzr=tokzr)
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
        multi_task_size_vocab, multi_task_size_option)


class LAVENDER_Multi_Task_Multi_Head(LAVENDER_Multi_Task):
    def __init__(self, args, tokzr, is_decoder=True):
        super(LAVENDER_Multi_Task, self).__init__(args, tokzr, is_decoder)
        # mc head
        # retrieval head
        self.fc = T.nn.Sequential(
            *[T.nn.Dropout(0.1),
              T.nn.Linear(self.hidden_size, self.hidden_size*2),
              T.nn.ReLU(inplace=True),
              T.nn.Linear(self.hidden_size*2, 1)])
        # oe head for ablation study, only msvd-qa is used with size vocab=1000
        for key, size_vocab in args.multi_task_size_vocab.items():
            key_attr = key.replace('-', '_')
            new_fc_layer = T.nn.Sequential(
                *[
                    T.nn.Dropout(0.1),
                    T.nn.Linear(self.hidden_size, self.hidden_size*2),
                    T.nn.ReLU(inplace=True),
                    T.nn.Linear(self.hidden_size*2, size_vocab)])
            setattr(self, f'fc_{key_attr}', new_fc_layer)

        for key, size_option in args.multi_task_size_option.items():
            key_attr = key.replace('-', '_')
            new_fc_layer = T.nn.Sequential(
                *[
                    T.nn.Dropout(0.1),
                    T.nn.Linear(self.hidden_size, self.hidden_size*2),
                    T.nn.ReLU(inplace=True),
                    T.nn.Linear(self.hidden_size*2, size_option)])
            setattr(self, f'fc_{key_attr}', new_fc_layer)

    def get_fc_layers(self, key):
        key_attr = key.replace('-', '_')
        return getattr(self, f'fc_{key_attr}')

    def forward_retrieval(self, batch):
        img, txt, mask, vid = [
                batch[key] for key in [
                    "img", "txt", "mask", "vid"]]
        (_B, _T, _, _H, _W) = img.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)

        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            for j in range(_B):
                pdt_feat_img.append(feat_img[i].unsqueeze(0))
                pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0))
                pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [
            T.cat(x, dim=0)
            for x in [pdt_feat_img, pdt_mask_img,
                      pdt_feat_txt, pdt_mask_txt]
            ]
        out, _ = self.go_cross(
            pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        out = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze().view(
            [_B, _B])  # / 0.05

        ans = T.tensor([i for i in range(_B)]).long().cuda()

        return out, ans

    def forward_qamc_ret(self, batch):
        img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["ans"]

        (_B, _T, _, _H, _W), (_, _O, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt.flatten(0, 1), mask.flatten(0, 1))

        feat_img, mask_img = [
            feat_img.unsqueeze(1).expand([-1, _O, -1, -1]).flatten(0, 1),
            mask_img.unsqueeze(1).expand([-1, _O, -1]).flatten(0, 1)]
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze(dim=-1).view([_B, _O])
        return out, ans

    def forward_qamc(self, batch):
        img, txt, mask, ans = [
                batch[key] for key in ["img", "txt", "mask", "ans"]]

        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt, mask)

        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.get_fc_layers(batch["task"])(out[:, (1+_h*_w)*_T, :])
        return out, ans

    def forward_qaoe(self, batch):
        img, txt, mask, ans = [
                batch[key] for key in ["img", "txt", "mask", "ans"]]
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.get_fc_layers(batch["task"])(out[:, (1+_h*_w)*_T, :])
        return out, ans


class Agent_Multi_Task_Multi_Head(Agent_Multi_Task):
    def __init__(self, args, model):
        super(Agent_Captioning, self).__init__(args, model)
        cap_loss_config = {
            'label_smoothing': getattr(args, 'label_smoothing', 0),
            'drop_worst_ratio': getattr(args, 'drop_worst_ratio', 0),
            'drop_worst_ratio': getattr(args, 'drop_worst_ratio', 0)}
        self.other_loss_func = T.nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.ret_loss_func = NormSoftmaxLoss(
            temperature=0.05).cuda()
        self.cap_loss_func = CaptioningLoss(cap_loss_config).cuda()
        self.log = defaultdict(list)
        self.task2loss = {}
        self.task2acc = {}
        assert not self.args.enable_prompt
        assert not self.args.enable_task_token

    def get_top_k_acc(self, out, ans, k=5):
        _B, _O = out.shape
        ans = ans.view(-1, 1)
        _, out_i = T.topk(out, k=k, dim=-1)
        ac = (out_i == ans).any(dim=-1).float().tolist()
        return ac

    def eval_step(self, batch):
        task = batch["task"]
        out = self.forward_step(batch)
        out, ans = out["out"], out["ans"]
        if "retrieval" in task or "qamc" in task:
            out = T.argmax(out, dim=1)
            ac = (out == ans).float().tolist()
            return {'ac': ac}
        elif "qaoe" in task:
            ac_1 = self.get_top_k_acc(out, ans, k=1)
            ac_5 = self.get_top_k_acc(out, ans, k=5)
            return {'ac_1': ac_1, 'ac_5': ac_5}
        else:
            raise ValueError(f"eval_step for {task} not defined")

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
            elif "retrieval" in task:
                ls = self.ret_loss_func(logits)
            else:
                ls = self.loss_func(logits, ans)

            self.backward_step(ls)
            pred = T.argmax(logits, dim=-1)
            acc = (
                float((pred == ans).sum() / (ans != -1).sum())
                if (ans != -1).sum() > 0 else 0)
            return {'ls': ls.item(), 'ac': acc}

    def run(self, meta_dl_tr, meta_dl_vl, meta_dl_ts):
        LOGGER.info("Start training....")
        step = 0

        for step, (task, batch) in enumerate(meta_dl_tr):
            ep = (step // self.args.iter_per_ep) + 1
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


if __name__ == '__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    (meta_dl_tr, meta_dl_vl, meta_dl_ts,
     len_dl_tr, multi_task_size_vocab, multi_task_size_option
     ) = get_meta_dataloaders(args, tokzr=tokzr)
    args.multi_task_size_vocab = multi_task_size_vocab
    args.multi_task_size_option = multi_task_size_option
    meta_dl_tr = MetaLoader(meta_dl_tr, distributed=args.distributed)
    if args.size_epoch == 0:
        args.max_iter = 1
    else:
        args.max_iter = (
            len_dl_tr * args.size_epoch)  # estimated
    args.iter_per_ep = len_dl_tr

    model = LAVENDER_Multi_Task_Multi_Head(
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
    agent = Agent_Multi_Task_Multi_Head(
        args, model)
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
