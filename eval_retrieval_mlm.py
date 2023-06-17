
from utils.lib import *
from utils.args import get_args
from eval_retrieval_task_specific import (
    Dataset_Product,
    Dataset_RetrievalTsEval)
from dataset import move_to_cuda
from main_retrieval_mlm import LAVENDER_Retrieval_MLM


class LAVENDER_RetrievalMlmEval(LAVENDER_Retrieval_MLM):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(
                0.02*T.randn(10, self.hidden_size))

    def forward(self, typ, batch):
        batch = defaultdict(lambda: None, batch)

        if typ == 'feat':
            img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
            prompt = (batch["prompt_txt"], batch["prompt_mask"])
            _B, _Clips, _T, _C, _H, _W = img.shape
            img = img.view(-1, _T, _C, _H, _W)
            feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
                img, txt, mask)
            _hidden_size = feat_img.shape[-1]
            mean_mask_img = mask_img.view(_B, _Clips, -1)
            mean_feat_img = feat_img.view(_B, _Clips, -1, _hidden_size)
            mean_feat_img = mean_feat_img.mean(dim=1)
            mean_mask_img = mean_mask_img[:, 0, :]
            txt, mask_txt, feat_txt = self.prepro_txt_inputs(
                txt, mask_txt, feat_txt,
                task_name=batch["task_name"], prompt=prompt)
            return mean_feat_img, mean_mask_img, feat_txt, mask_txt, txt

        elif typ == 'cross':
            feat_img, mask_img, feat_txt, mask_txt = [
                batch[key] for key in [
                    "feat_img", "mask_img", "feat_txt", "mask_txt"]]
            txt = batch["txt"]
            out, _ = self.go_cross(
                feat_img, mask_img, feat_txt, mask_txt)
            out = self.fc_mtm(out[:, feat_img.shape[1]:])
            return out, txt


class Dataset_RetrievalMlmEval(Dataset_RetrievalTsEval):
    def __init__(self, args, split):
        super().__init__(
            args, split)

    @property
    def prompt_text(self):
        return "is the video-text paired, true or false?"

    def str2txt(self, s):
        txt, mask = super().str2txt(s)
        txt, mask = self.append_mask_tok2txt(txt, mask)
        return txt, mask

    def collate_batch(self, inputs):
        img, txt, mask, txt_id, video_id = map(list, unzip(inputs))
        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {
            "img": all_imgs, "txt": all_txts, "tid": txt_id,
            "mask": all_masks, "vid": video_id}
        return batch


class Dataset_Product(T.utils.data.Dataset):
    def __init__(self, featv, featt):
        super().__init__()
        self.vids = list(set([item["video"] for key, item in featv.items()]))
        self.vid2idx = {v: i for i, v in enumerate(self.vids)}
        self.tids = list(set([item["tid"] for key, item in featt.items()]))
        self.tid2idx = {t: i for i, t in enumerate(self.tids)}
        self.lst = [[featt[p], featv[q]] for p in featt for q in featv]

        # self.vid2idx = {v: i for i, v in enumerate(feat)}
        # self.lst = [[feat[p], feat[q]] for p in feat for q in feat]

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        p, q = self.lst[idx]

        return (
            p['feat_txt'], p['mask_txt'], p['tid'],
            q['feat_img'], q['mask_img'], q['video'],
            p['txt'])  # (p->text, q->video)

    def collate_batch(self, inputs):
        feat_txt, mask_txt, tid, feat_img, mask_img, vid, txt = map(
            list, unzip(inputs))
        all_feat_txt = T.stack(feat_txt, dim=0)
        all_feat_img = T.stack(feat_img, dim=0)
        all_mask_txt = T.stack(mask_txt, dim=0)
        all_mask_img = T.stack(mask_img, dim=0)
        all_txts = T.stack(txt, dim=0)

        batch = {
            "feat_txt": all_feat_txt, "mask_txt": all_mask_txt,
            "feat_img": all_feat_img, "mask_img": all_mask_img,
            "vid": vid,
            "tid": tid, "txt": all_txts}
        return batch


if __name__ == '__main__':
    args = get_args(distributed=False)
    args.use_checkpoint = False
    args.num_gpus = T.cuda.device_count()
    if args.multi_clip_testing:
        args.size_batch = 10*args.num_gpus
    else:
        if args.size_txt <= 32:
            args.size_batch = 100*args.num_gpus
        else:
            args.size_batch = 80*args.num_gpus
    assert os.path.exists(args.path_ckpt)

    print(args)
    ds_ret = Dataset_RetrievalMlmEval(args, "test")

    log = {}
    model = T.nn.DataParallel(
        LAVENDER_RetrievalMlmEval(args, ds_ret.tokzr).cuda())
    model.module.load_ckpt(args.path_ckpt)
    model.eval()

    for split in ['test']:  # ['val', 'test']:
        ds_ret = Dataset_RetrievalMlmEval(args, split)
        true_token_id = ds_ret.true_token_id
        false_token_id = ds_ret.false_token_id
        mask_token_id = ds_ret.mask_token_id
        dl = T.utils.data.DataLoader(
            ds_ret,
            batch_size=args.size_batch, shuffle=False,
            num_workers=args.n_workers, pin_memory=True,
            worker_init_fn=ds_ret.read_tsv,
            collate_fn=ds_ret.collate_batch)
        featv = {}
        featt = {}
        gt_txt2vid = ds_ret.gt_txt2vid
        for batch in tqdm(dl, ascii=True):
            with T.no_grad():
                if args.enable_prompt:
                    _B, _ = batch["txt"].shape
                    (vtm_prompt_txt, vtm_prompt_mask
                     ) = dl.dataset.get_prompt()
                    vtm_prompt_txt = vtm_prompt_txt.unsqueeze(0).expand(_B, -1)
                    vtm_prompt_mask = vtm_prompt_mask.unsqueeze(
                        0).expand(_B, -1)
                    batch["prompt_mask"] = vtm_prompt_mask
                    batch["prompt_txt"] = vtm_prompt_txt
                elif args.enable_task_token:
                    batch["task_name"] = "vtm"

                batch = move_to_cuda(batch)
                feat_img, mask_img, feat_txt, mask_txt, txt = model(
                    typ='feat', batch=batch)
            for v_idx, v in enumerate(batch["vid"]):
                f_i = feat_img[v_idx]
                f_t = feat_txt[v_idx]
                m_t = mask_txt[v_idx]
                m_i = mask_img[v_idx]
                t = txt[v_idx]
                tid_ = batch["tid"][v_idx]
                if v not in featv:
                    featv[v] = {
                        'video': v, 'feat_img': f_i.cpu(),
                        'mask_img': m_i.cpu()}
                featt[tid_] = {
                    'tid': tid_, 'feat_txt': f_t.cpu(), 'mask_txt': m_t.cpu(),
                    'txt': t.cpu()
                    }
        ds = Dataset_Product(featv, featt)
        dl = T.utils.data.DataLoader(ds,
                                     batch_size=args.size_batch,
                                     shuffle=False,
                                     num_workers=args.n_workers,
                                     pin_memory=True,
                                     collate_fn=ds.collate_batch)
        print(f"number of videos: {len(ds.vid2idx)}")
        print(f"number of queires (by text): {len(ds.tid2idx)}")
        print(f"number of queries (before gathering rank): {len(ds_ret)}")
        rank = {}
        for batch in tqdm(dl, ascii=True):
            # batch = move_to_cuda(batch)
            # (feat_txt, mask_txt, idx_txt, feat_img,
            #     mask_img, idx_vid, txt)
            with T.no_grad():
                out = model(typ='cross', batch=batch)
                # out = T.sigmoid(out).data.cpu().numpy()
                out_ret, txt = out
                p_true = out_ret[:, :, true_token_id]
                p_false = out_ret[:, :, false_token_id]
                out_ret = p_true / (p_true+p_false)
                out_ret = out_ret[txt == mask_token_id].cpu().numpy()
            tid, vid = batch["tid"], batch["vid"]
            assert len(tid) == len(out_ret)
            for tid_, vid_, o in zip(tid, vid, out_ret):
                i_v = ds.vid2idx[vid_]
                i_v, o = int(i_v), float(o)

                if tid_ not in rank:
                    rank[tid_] = {}
                if i_v not in rank[tid_]:
                    rank[tid_][i_v] = o
                else:
                    print(f"repeative entry for {tid_} {i_v}")

        res = {'r@1': 0, 'r@5': 0, 'r@10': 0, 'median': []}
        print(f"number of queries (after gathering rank): {len(rank)}")
        rank = {tid_: [(i_v, o) for i_v, o in item.items()]
                for tid_, item in rank.items()}
        for tid_ in rank:
            tmp = sorted(rank[tid_], key=lambda d: -d[1])
            gt_iv = ds.vid2idx[gt_txt2vid[tid_]]
            p = [d[0] for d in tmp].index(gt_iv)+1

            if p <= 1:
                res['r@1'] += 1.0/len(rank)
            if p <= 5:
                res['r@5'] += 1.0/len(rank)
            if p <= 10:
                res['r@10'] += 1.0/len(rank)
            res['median'].append(p)
        res['median'] = int(np.median(res['median']))
        res = {key: f"{val*100:.2f}"
               if key != 'median'
               else f"{val}"
               for key, val in res.items()}
        log[split] = res
        print(split, res)
    path_ckpt_dir = os.path.dirname(args.path_ckpt)
    filename, _ = os.path.splitext(args.path_ckpt.split("/")[-1])
    log_path = f"{path_ckpt_dir}/{filename}_results.json"
    log["multi_clip_testing"] = args.multi_clip_testing
    json.dump(log, open(log_path, "w"))
