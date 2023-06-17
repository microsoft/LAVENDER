
from utils.lib import *
from main_pretrain_task_specific import (
    Dataset_Pretrain, LAVENDER_Pretrain,
    Agent_Pretrain, get_dl)
from utils.dist import iter_tqdm
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import (
    is_main_process,
    get_rank, get_world_size, iter_tqdm,
    NoOp)


class Dataset_Pretrain_MLM(Dataset_Pretrain):
    def __init__(self, args, txt, dataset, split,
                 part=None, data_dir=None, tokzr=None):
        super().__init__(
            args, txt, dataset, split, part, data_dir,
            tokzr=tokzr)

    def str2txt(self, s):
        txt, mask = super().str2txt(s)
        txt, mask = self.append_mask_tok2txt(txt, mask)
        return txt, mask

    @property
    def vtm_prompt_text(self):
        return "is the video-text paired, true or false?"

    def get_vtm_prompt(self):
        return self.get_prompt(prompt_text=self.vtm_prompt_text)

    @property
    def cap_prompt_text(self):
        return "write a description about the video."

    def get_cap_prompt(self):
        return self.get_prompt(prompt_text=self.cap_prompt_text)


class LAVENDER_Pretrain_MLM(LAVENDER_Pretrain):
    def __init__(self, args, tokzr=None):
        super(LAVENDER_Pretrain, self).__init__(args, tokzr)
        self.patch_size = args.size_patch
        bert = transformers.AutoModelForMaskedLM.from_pretrained(
            self.args.tokenizer)
        self.fc_mtm = bert.cls
        del bert
        self.vtm_batch = min(self.args.size_batch, 4)
        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(
                0.02*T.randn(10, self.hidden_size))

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        img, txt, mask = [
            batch[key] for key in ["img", "txt", "mask"]]
        vt_mask = batch["vt_mask"]
        ans_mtm = batch["ans_mtm"]
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//self.patch_size, _W//self.patch_size
        _O = min(_B, self.vtm_batch)

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt, mask, vt_mask=vt_mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)

        out_mtm = self.fc_mtm(out[:, (1+_h*_w)*_T:])

        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        ans_vtm = []

        for i in range(_B):
            mt = mask_txt[i]
            t = txt[i]
            ft = feat_txt[i]
            t, mt, ft = self.prepro_txt_inputs(
                t, mt, ft, task_name="vtm",
                prompt=batch["vtm_prompt"])
            # mt[-1] = 1
            pdt_feat_img.append(feat_img[i].unsqueeze(0))
            pdt_mask_img.append(mask_img[i].unsqueeze(0))
            pdt_feat_txt.append(ft.unsqueeze(0))
            pdt_mask_txt.append(mt.unsqueeze(0))
            gt_txt = T.ones_like(t)*-1
            gt_txt[-1] = self.true_token_id
            ans_vtm.append(gt_txt.unsqueeze(0))

            neg = np.random.permutation(
                [j for j in range(_B) if j != i])
            for j in range(_O-1):
                j = neg[j]
                mt = mask_txt[j]
                t = txt[j]
                ft = feat_txt[j]
                t, mt, ft = self.prepro_txt_inputs(
                    t, mt, ft, task_name="vtm",
                    prompt=batch["vtm_prompt"])
                pdt_feat_img.append(feat_img[i].unsqueeze(0))
                pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(ft.unsqueeze(0))
                pdt_mask_txt.append(mt.unsqueeze(0))
                gt_txt = T.ones_like(t)*-1
                gt_txt[-1] = self.false_token_id
                ans_vtm.append(gt_txt.unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt, ans_vtm = [
            T.cat(x, dim=0)
            for x in [
                pdt_feat_img, pdt_mask_img,
                pdt_feat_txt, pdt_mask_txt, ans_vtm]]
        out, _ = self.go_cross(
            pdt_feat_img, pdt_mask_img,
            pdt_feat_txt, pdt_mask_txt)
        out_vtm = self.fc_mtm(out[:, (1+_h*_w)*_T:])

        output = {"out_vtm": out_vtm, "out_mtm": out_mtm,
                  "ans_vtm": ans_vtm, "ans_mtm": ans_mtm}
        return output


class Agent_Pretrain_MLM(Agent_Pretrain):
    def __init__(self, args, model):
        super().__init__(args, model)

    def cal_vtm_loss(self, txt, out, ans, is_train=True):
        if is_train:
            out = out.flatten(0, len(out.shape)-2)
            ans = ans.flatten(0, len(ans.shape)-1)
            ls = self.loss_func(out, ans)
            return ls
        else:
            _B, _ = txt.shape
            p_true = out[:, :, self.true_token_id]
            p_false = out[:, :, self.false_token_id]
            out_vtm = p_true / (p_true+p_false)
            ans_vtm = ans
            out_vtm = out_vtm[ans_vtm != -1].view(_B, -1)
            ans_vtm = ans_vtm[ans_vtm != -1].view(_B, -1)
            out_vtm = T.argmax(out_vtm, dim=-1)
            ans_vtm_idx = (ans_vtm == self.true_token_id).nonzero()[:, 1]
            ac = float((out_vtm == ans_vtm_idx).float().sum() / _B)
            return ac

    def step(self, batch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            (out_mtm, out_vtm) = (
                out[key] for key in [
                    "out_mtm", "out_vtm"])
            (ans_mtm, ans_vtm) = (
                out[key] for key in [
                    "ans_mtm", "ans_vtm"])
            ls_mtm = self.loss_func(
                out_mtm.flatten(0, len(out_mtm.shape)-2),
                ans_mtm.flatten(0, len(ans_mtm.shape)-1))
            ls_vtm = self.cal_vtm_loss(
                batch["txt"], out_vtm, ans_vtm, is_train)
            ls = ls_mtm + ls_vtm
        if is_train:
            self.backward_step(ls)
            return {
                'mtm': ls_mtm.item(),
                'vtm': ls_vtm.item()}
        else:
            out_mtm = T.argmax(out_mtm, dim=-1)

            ac_mtm = (
                float((out_mtm == ans_mtm).sum() / (ans_mtm != -1).sum())
                if (ans_mtm != -1).sum() > 0 else -1)
            res = {'mtm': ac_mtm, 'vtm': ls_vtm}
            return res

    def masking(self,  txt, mask, p_mask=0.15):
        (_B, _X) = txt.shape

        spc_txt = T.logical_or(
            T.logical_or(txt == self.cls_token_id, txt == self.sep_token_id),
            T.logical_or(txt == self.pad_token_id, txt == self.mask_token_id))

        ans_mtm = T.ones(txt.shape).long() * -1

        if p_mask <= 0:
            return {
                "txt": txt, "mask": mask,
                "ans_mtm": ans_mtm}

        for i in range(_B):
            mask_mtm = T.where(T.logical_and(
                T.logical_not(spc_txt[i]), T.rand(_X) < p_mask))[0]

            for p in mask_mtm:
                ans_mtm[i][p], txt[i][p] = txt[i][p], self.mask_token_id

        return {"txt": txt, "mask": mask,
                "ans_mtm": ans_mtm}

    def go_dl(self, ep, dl, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        ret = defaultdict(list)  # {'mtm': [], 'vtm': []}
        idx = 0
        for idx, batch in enumerate(dl):
            batch = defaultdict(lambda: None, batch)
            if idx % self.args.logging_steps == 0 and is_train:
                LOGGER.info(self.log_memory(ep, idx+1))
            img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
            masked_batch = self.masking(txt, mask)
            batch.update(masked_batch)
            if self.args.enable_prompt:
                batch["vtm_prompt"] = dl.dataset.get_vtm_prompt()
                batch["cap_prompt"] = dl.dataset.get_cap_prompt()
            batch = self.prepare_batch(batch)
            r = self.step(batch, is_train)
            ret = {k: ret[k]+[l] for k, l in r.items()}

        if idx % self.args.logging_steps != 0 and is_train:
            LOGGER.info(self.log_memory(ep, idx+1))

        ret = {
            k: self.reduce_mean(
                float(np.average(
                        [v for v in l if not math.isnan(v)])))
            for k, l in ret.items()}
        return ret


if __name__ == '__main__':

    args = get_args()

    for d in args.dataset:
        args.task += f"-{d}"

    args.path_output = '%s/_%s_%s' % (
        args.path_output, args.task,
        datetime.now().strftime('%Y%m%d%H%M%S'))

    LOGGER.info("Loading Data....")
    dataloaders = {}
    txt_data = {}
    dl_tr_len = 0
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    for dataset in args.dataset:
        if isinstance(args.dataset, dict):
            data_dir = args.dataset[dataset]
        else:
            data_dir = args.data_dir
        txt_data[dataset] = json.load(
            open(f'{data_dir}/txt_{dataset}.json', 'r'))

        ds = Dataset_Pretrain_MLM(
            args, txt_data[dataset], dataset, 'val',
            data_dir=data_dir, tokzr=tokzr)
        dataloaders[f"{dataset}-val"] = get_dl(
            ds, args, worker_init_fn=ds.read_tsv, collate_fn=ds.collate_batch)
        size_part = (
            args.size_part
            if isinstance(args.size_part, int)
            else args.size_part[dataset])
        true_token_id = ds.true_token_id
        false_token_id = ds.false_token_id
        ds = Dataset_Pretrain_MLM(
            args, txt_data[dataset], 
            dataset, 'train', 0, data_dir=data_dir)
        dataloaders[f"{dataset}-train-0"] = get_dl(
            ds, args, worker_init_fn=ds.read_tsv,
            collate_fn=ds.collate_batch)
        dl_tr_len += len(dataloaders[f"{dataset}-train-0"]) * size_part
    args.max_iter = dl_tr_len * args.size_epoch  # estimated

    model = LAVENDER_Pretrain_MLM(args, tokzr)
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed:
        LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                    f" world_size: {get_world_size()}")

    agent = Agent_Pretrain_MLM(args, model)
    if args.distributed:
        agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process():
        add_log_to_file('%s/stdout.txt' % (args.path_output))
    else:
        LOGGER = NoOp()
    LOGGER.info("Saved training meta infomation, start training ...")

    for e in iter_tqdm(range(args.size_epoch)):
        for dataset in args.dataset:
            dl_vl = dataloaders[f"{dataset}-val"]
            size_part = (
                args.size_part
                if isinstance(args.size_part, int)
                else args.size_part[dataset])
            for part in iter_tqdm(range(size_part)):
                dl_key = f"{dataset}-train-{part}"
                if dl_key in dataloaders:
                    dl_tr = dataloaders[dl_key]
                else:
                    ds = Dataset_Pretrain_MLM(
                        args, txt_data[dataset],
                        dataset, 'train', part,
                        data_dir=dataloaders[
                            f"{dataset}-train-0"].dataset.data_dir,
                        tokzr=tokzr)
                    dl_tr = get_dl(
                        ds, args, worker_init_fn=ds.read_tsv,
                        collate_fn=ds.collate_batch)
                if args.distributed:
                    dl_tr.sampler.set_epoch(e+1)

                ls_tr = agent.go_dl(e+1, dl_tr, True)

                ac_vl = agent.go_dl(e+1, dl_vl, False)
                for k in ls_tr:
                    agent.log[dataset]['ls_%s' % (k)].append(ls_tr[k])
                    agent.log[dataset]['ac_%s' % (k)].append(ac_vl[k])
                agent.save_model(e+1, dataset, part)
                LOGGER.info(f'Ep {e+1}, dataset {dataset}, part {part}: '
                            f'{json.dumps(ls_tr)}, {json.dumps(ac_vl)}')
