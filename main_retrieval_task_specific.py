from utils.lib import *
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from dataset import Dataset_Base, get_tsv_dls
from utils.dist import (
    is_main_process,
    get_rank, get_world_size, iter_tqdm,
    NoOp)
from model import LAVENDER_Base
from agent import Agent_Base, NormSoftmaxLoss


class Dataset_Retrieval_TS(Dataset_Base):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr):
        super().__init__(
            args, split, size_frame=args.size_frame, tokzr=tokzr)
        self.txt = txt[split]
        self.img_tsv_path = img_tsv_path
        self.id2lineidx = id2lineidx
        if args.data_ratio != 1:
            self.get_partial_data()
        self.vid2txt = defaultdict(list)
        for item in self.txt:
            self.vid2txt[item["video"]].append(item)
        if self.split != "train" and len(self.txt) > len(self.vid2txt):
            # use just one caption during evaluation
            first_txt = []
            for vid, list_of_items in self.vid2txt.items():
                first_txt.append(list_of_items[0])
            self.txt = first_txt

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_img_or_video(b)

        raw_txt = item['caption']
        if isinstance(raw_txt, list):
            # random text augmentation from Frozen
            sent_ids = range(len(raw_txt))
            if self.split == "train":
                size_sent = random.randint(1, len(raw_txt))
                sent_ids = random.sample(sent_ids, size_sent)
            raw_txt = " ".join([raw_txt[i] for i in sent_ids])

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, video_id

    def collate_batch(self, inputs):
        img, txt, mask, video_id = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {
            "img": all_imgs, "txt": all_txts,
            "mask": all_masks, "vid": video_id}
        return batch


class LAVENDER_Retrieval_TS(LAVENDER_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        self.fc = T.nn.Sequential(*[
            T.nn.Dropout(0.1),
            T.nn.Linear(self.hidden_size, self.hidden_size*2),
            T.nn.ReLU(inplace=True),
            T.nn.Linear(self.hidden_size*2, 1)])

    def forward(self, batch):
        img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
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


class Agent_Retrieval_TS(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.loss_func = NormSoftmaxLoss(
            temperature=args.temp).cuda()
        self.log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}

    def step(self, batch, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            out, ans = out
            ls = self.loss_func(out)
        if is_train:
            self.backward_step(ls)
            return ls.item()
        else:
            out = T.argmax(out, dim=1)
            ac = (out == ans).float().mean().item()
            return ac

    def go_dl(self, ep, dl, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        ret = []
        idx = 0
        for idx, batch in enumerate(dl):
            if idx % self.args.logging_steps == 0 and is_train:
                LOGGER.info(self.log_memory(ep, idx+1))
            batch = self.prepare_batch(batch)
            curr_ret = self.step(batch, is_train)
            ret.append(curr_ret)

        if idx % self.args.logging_steps != 0 and is_train:
            LOGGER.info(self.log_memory(ep, idx+1))

        ret = float(float(np.average(ret)))
        if self.args.distributed:
            ret = self.reduce_mean(ret)
        return ret


if __name__ == '__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    dl_tr, dl_vl, dl_ts = get_tsv_dls(args, Dataset_Retrieval_TS, tokzr=tokzr)

    args.max_iter = len(dl_tr) * args.size_epoch

    model = LAVENDER_Retrieval_TS(args, tokzr=tokzr)
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed:
        LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                    f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s' % (
        args.path_output, args.task,
        datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_Retrieval_TS(args, model)
    if args.distributed:
        agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process():
        add_log_to_file('%s/stdout.txt' % (args.path_output))
    else:
        LOGGER = NoOp()
    # DIST.barrier()
    LOGGER.info("Saved training meta infomation, start training ...")
    for e in iter_tqdm(range(args.size_epoch)):
        ls_tr = agent.go_dl(e+1, dl_tr, True)

        ac_vl = agent.go_dl(e+1, dl_vl, False)
        ac_ts = agent.go_dl(e+1, dl_ts, False)

        agent.log['ls_tr'].append(ls_tr)
        agent.log['ac_vl'].append(ac_vl)
        agent.log['ac_ts'].append(ac_ts)
        agent.save_model(e+1)
        LOGGER.info('Ep %d: %.6f %.6f %.6f' % (
            e+1, ls_tr, ac_vl, ac_ts))
    best_vl, best_ts = agent.best_epoch()
    LOGGER.info(f'Best val @ ep {best_vl[0]+1}, {best_vl[1]:.6f}')
    LOGGER.info(f'Best test @ ep {best_ts[0]+1}, {best_ts[1]:.6f}')
