
from utils.lib import *
from dataset import Dataset_Base, get_dl
from model import LAVENDER_Base
from agent import Agent_Base
from utils.dist import iter_tqdm
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import (
    is_main_process,
    get_rank, get_world_size, iter_tqdm,
    NoOp)


class Dataset_Pretrain(Dataset_Base):
    def __init__(self, args, txt, dataset, split,
                 part=None, data_dir=None, tokzr=None):
        super().__init__(args, split=split,
                         size_frame=args.size_frame, tokzr=tokzr)
        if dataset in ["cc3m", "coco", "vg", "cc12m"]:
            self.size_frame = 1
        self.dataset, self.part = dataset, part
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = args.data_dir

        self.txt = txt[self.split]
        if self.dataset == "webvid10m":
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/_webvid10m-tsv_frame4/webvid10m-{self.part+1:03d}.img.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.lineidx', 'r')]
        elif self.dataset == "webvid10m_filtered":
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/image-1{self.part:04d}.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.lineidx', 'r')]
        elif self.dataset == "cc12m":
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/train.{self.part}.62.img.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/cc3m_val.lineidx', 'r')]
        else:
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/{self.dataset}_train_{self.part}.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/{self.dataset}_val.lineidx', 'r')]

    def read_tsv(self, worker_id):
        if self.dataset == "webvid10m":
            self.tsv = open(
                f'{self.data_dir}/_webvid10m-tsv_frame4/webvid10m-{self.part+1:03d}.img.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.tsv', 'r')
        elif self.dataset == "webvid10m_filtered":
            self.tsv = open(
                f'{self.data_dir}/image-1{self.part:04d}.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.tsv', 'r')
        elif self.dataset == "cc12m":
            self.tsv = open(
                f'{self.data_dir}/train.{self.part}.62.img.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/cc3m_val.tsv', 'r')
        else:
            self.tsv = open(
                f'{self.data_dir}/{self.dataset}_train_{self.part}.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/{self.dataset}_val.tsv', 'r')

    def __len__(self):
        return len(self.lineidx)

    def __getitem__(self, idx):
        lineidx = self.lineidx[idx]
        self.tsv.seek(lineidx)
        item = self.tsv.readline().split('\t')

        if self.dataset in [
                "webvid10m", "webvid10m_filtered"
                ] and self.split == "train":
            vid, bufs = item[0], item[2:]
        else:
            vid, bufs = item[0], item[1:]

        if vid in self.txt:
            raw_txt = self.txt[vid][0]
        else:
            print(f"Failed to load txt for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}")
            raw_txt = ""

        try:
            img = self.get_img_or_video(bufs)
            (_T, _, _H, _W) = img.shape
        except Exception as e:
            print(f"Failed to load image binaries for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}, {e}")
            _T = self.args.size_frame
            _H = self.args.size_img
            _W = _H
            _C = 3
            img = T.zeros((_T, _C, _H, _W))

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask

    def collate_batch(self, inputs):
        img, txt, mask = map(list, unzip(inputs))
        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {
            "img": all_imgs, "txt": all_txts,
            "mask": all_masks}
        return batch


class LAVENDER_Pretrain(LAVENDER_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        self.patch_size = args.size_patch
        self.fc = T.nn.Sequential(*[
            T.nn.Dropout(0.1),
            T.nn.Linear(self.hidden_size, self.hidden_size*2),
            T.nn.ReLU(inplace=True),
            T.nn.Linear(self.hidden_size*2, 1)])
        bert = transformers.AutoModelForMaskedLM.from_pretrained(
            self.args.tokenizer)
        self.fc_mtm = bert.cls
        del bert

    def forward(self, img, txt, mask, ans_mtm):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//self.patch_size, _W//self.patch_size
        _O = min(_B, 4)

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)

        out_mtm = self.fc_mtm(out[:, (1+_h*_w)*_T:])

        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            pdt_feat_img.append(feat_img[i].unsqueeze(0))
            pdt_mask_img.append(mask_img[i].unsqueeze(0))
            pdt_feat_txt.append(feat_txt[i].unsqueeze(0))
            pdt_mask_txt.append(mask_txt[i].unsqueeze(0))

            neg = np.random.permutation(
                [j for j in range(_B) if j != i])
            for j in range(_O-1):
                j = neg[j]
                pdt_feat_img.append(feat_img[i].unsqueeze(0))
                pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0))
                pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [
            T.cat(x, dim=0)
            for x in [
                pdt_feat_img, pdt_mask_img,
                pdt_feat_txt, pdt_mask_txt]]
        out, _ = self.go_cross(
            pdt_feat_img, pdt_mask_img,
            pdt_feat_txt, pdt_mask_txt)
        out_vtm = self.fc(
            out[:, (1+_h*_w)*_T, :]).squeeze().view(
                [_B, _O]) / self.args.temp

        ans_vtm = T.tensor([0 for _ in range(_B)]).long().cuda()

        output = {"out_vtm": out_vtm,  "out_mtm": out_mtm,
                  "ans_vtm": ans_vtm, "ans_mtm": ans_mtm}
        return output


class Agent_Pretrain(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.patch_size = self.model.patch_size
        self.log = {
            dataset: defaultdict(list)
            for dataset in self.args.dataset}

    def masking(self, txt, mask, p_mask=0.15):
        (_B, _X) =  txt.shape

        spc_txt = T.logical_or(
            T.logical_or(txt == self.cls_token_id, txt == self.sep_token_id),
            T.logical_or(txt == self.pad_token_id, txt == self.mask_token_id)
            )

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

    def step(self, batch, is_train=True):
        img, txt, mask = [
            batch[key] for key in ["img", "txt", "mask"]]
        ans_mtm = batch["ans_mtm"]
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step((img, txt, mask, ans_mtm))
            (out_mtm, out_vtm) = (
                out[key] for key in [
                    "out_mtm",  "out_vtm"])
            (ans_mtm, ans_vtm) = (
                out[key] for key in [
                    "ans_mtm",  "ans_vtm"])
            ls_mtm = self.loss_func(
                out_mtm.flatten(0, len(out_mtm.shape)-2),
                ans_mtm.flatten(0, len(ans_mtm.shape)-1))
            ls_vtm = self.loss_func(
                out_vtm.flatten(0, len(out_vtm.shape)-2),
                ans_vtm.flatten(0, len(ans_vtm.shape)-1))
            ls = ls_mtm + ls_vtm
        if is_train:
            self.backward_step(ls)
            return {
                'mtm': ls_mtm.item(),
                'vtm': ls_vtm.item()}
        else:
            out_mtm, out_vtm = [
                T.argmax(o, dim=-1)
                for o in [out_mtm, out_vtm]]

            ac_mtm, ac_vtm = [
                float((o == a).sum() / (a != -1).sum())
                if (a != -1).sum() > 0 else -1
                for o, a in zip([out_mtm, out_vtm],
                                [ans_mtm, ans_vtm])]
            res = {'mtm': ac_mtm, 'vtm': ac_vtm}
            return res

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

    def save_model(self, ep, dataset="init", part=0):
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
                os.path.join(
                    f"{self.args.path_output}/"
                    f"ckpt_violet_pretrain_{dataset}_{part}_{ep}.pt"))


if __name__ == '__main__':

    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    model = LAVENDER_Pretrain(args, tokzr)
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed:
        LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                    f" world_size: {get_world_size()}")

    for d in args.dataset:
        args.task += f"-{d}"

    args.path_output = '%s/_%s_%s' % (
        args.path_output, args.task,
        datetime.now().strftime('%Y%m%d%H%M%S'))

    LOGGER.info("Loading Data....")
    dataloaders = {}
    txt_data = {}
    dl_tr_len = 0
    for dataset in args.dataset:
        if isinstance(args.dataset, dict):
            data_dir = args.dataset[dataset]
        else:
            data_dir = args.data_dir
        txt_data[dataset] = json.load(
            open(f'{data_dir}/txt_{dataset}.json', 'r'))
        ds = Dataset_Pretrain(
            args, txt_data[dataset], dataset, 'val',
            data_dir=data_dir, tokzr=tokzr)
        dataloaders[f"{dataset}-val"] = get_dl(
            ds, args, worker_init_fn=ds.read_tsv, collate_fn=ds.collate_batch)
        size_part = (
            args.size_part
            if isinstance(args.size_part, int)
            else args.size_part[dataset])
        # for part in range(size_part):
        ds = Dataset_Pretrain(
            args, txt_data[dataset],
            dataset, 'train', 0, data_dir=data_dir)
        dataloaders[f"{dataset}-train-0"] = get_dl(
            ds, args, worker_init_fn=ds.read_tsv,
            collate_fn=ds.collate_batch)
        dl_tr_len += len(dataloaders[f"{dataset}-train-0"]) * size_part

    args.max_iter = dl_tr_len * args.size_epoch  # estimated

    agent = Agent_Pretrain(args, model)
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
                    ds = Dataset_Pretrain(
                        args, txt_data[dataset],
                        dataset, 'train', part,
                        data_dir=dataloaders[
                            f"{dataset}-train-0"].dataset.data_dir)
                    dl_tr = get_dl(
                        ds, args, worker_init_fn=ds.read_tsv,
                        collate_fn=ds.collate_batch)
                if args.distributed:
                    dl_tr.sampler.set_epoch(e+1)

                ls_tr = agent.go_dl(e+1, dl_tr, True)

                ac_vl = agent.go_dl(e+1, dl_vl, False)
                for k in ls_tr:
                    agent.log[dataset][f'ls_{k}'].append(ls_tr[k])
                for k in ac_vl:
                    agent.log[dataset][f'ac_{k}'].append(ac_vl[k])
                agent.save_model(e+1, dataset, part)
                LOGGER.info(f'Ep {e+1}, dataset {dataset}, part {part}: '
                            f'{json.dumps(ls_tr)}, {json.dumps(ac_vl)}')
                if args.distributed:
                    DIST.barrier()
