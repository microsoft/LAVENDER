
from utils.lib import *
from utils.args import get_args
from main_retrieval_task_specific import LAVENDER_Retrieval_TS
from dataset import Dataset_Base


class Dataset_RetrievalTsEval(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args, split, size_frame=args.size_frame)

        self.img_tsv_path = f'{args.data_dir}/img_{args.dataset}.tsv'
        self.id2lineidx = pickle.load(open(
            f'{args.data_dir}/img_{args.dataset}.id2lineidx.pkl', 'rb'))
        self.txt = json.load(
            open(f'{args.data_dir}/txt_{args.task}.json',
                 'r'))[split]
        self.gt_txt2vid = {
            idx: item["video"] for idx, item in enumerate(self.txt)}

    def __len__(self):
        return len(self.txt)

    def get_clips_with_temporal_sampling(self, list_of_b):
        max_size_frame = len(list_of_b)

        list_of_sampled_videos = []
        if max_size_frame == 1 or self.size_frame == max_size_frame:
            list_of_sampled_videos.append(
                self.get_img_or_video(list_of_b).unsqueeze(0))
            return T.cat(list_of_sampled_videos, dim=0)

        if max_size_frame < self.size_frame:
            print(f"Error in size_frame",
                  f"\tasked for {size_frame} from {max_size_frame} frames")

        size_frame = min(self.size_frame, max_size_frame)
        size_clips = int(math.ceil(max_size_frame / size_frame))
        if self.args.multi_clip_testing:
            for sampled_start in range(size_clips):
                sampled_end = min(
                    sampled_start + (size_frame - 1) * size_clips,
                    max_size_frame - 1)

                sampled_index = self.sampling(
                    sampled_start, sampled_end, size_frame)
                sampled_video = [list_of_b[i] for i in sampled_index]
                sampled_video = self.get_img_or_video(sampled_video)
                list_of_sampled_videos.append(sampled_video.unsqueeze(0))
        else:
            # uniformly sample frames
            sampled_index = self.sampling(
                    0, max_size_frame - 1, size_frame)
            sampled_video = [list_of_b[i] for i in sampled_index]
            sampled_video = self.get_img_or_video(sampled_video)
            list_of_sampled_videos.append(sampled_video.unsqueeze(0))
        list_of_sampled_videos = T.cat(list_of_sampled_videos, dim=0)
        return list_of_sampled_videos

    def get_img_or_video(self, bufs):
        img = []
        for b in bufs:
            single_img = self.str2img(b)
            if self.args.img_transform == ["vid_rand_crop"]:
                vis_transform = "vid_center_crop"
                img.append(single_img)
            else:
                if self.args.img_transform == ["pad_resize"]:
                    vis_transform = "pad_resize"
                    single_img = self.pad_resize(single_img)
                else:
                    vis_transform = "img_center_crop"
                    single_img = self.img_center_crop(single_img)
                img.append(single_img.unsqueeze(0))
        if vis_transform == "vid_center_crop":
            img = self.vid_center_crop(img)
        else:
            img = T.cat(img, dim=0)
        return img

    def __getitem__(self, idx):
        item = self.txt[idx]

        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_clips_with_temporal_sampling(b)

        raw_txt = item['caption']
        if isinstance(raw_txt, list):
            assert self.split != "train"
            raw_txt = " ".join(raw_txt)

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, idx, item['video']


class Dataset_Product(T.utils.data.Dataset):
    def __init__(self, featv, featt):
        super().__init__()
        self.vids = list(set([item["video"] for key, item in featv.items()]))
        self.vid2idx = {v: i for i, v in enumerate(self.vids)}
        self.tids = list(set([item["tid"] for key, item in featt.items()]))
        self.tid2idx = {t: i for i, t in enumerate(self.tids)}
        self.lst = [[featt[p], featv[q]] for p in featt for q in featv]

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        p, q = self.lst[idx]

        return [p['feat_txt'], p['mask_txt'],
                p['tid'],
                q['feat_img'], q['mask_img'],
                q['video']]  # (p->text, q->video)


class LAVENDER_RetrievalTsEval(LAVENDER_Retrieval_TS):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)

    def forward(
            self, typ,
            img=None, txt=None, mask=None,
            feat_img=None, mask_img=None, feat_txt=None, mask_txt=None):

        if typ == 'feat':
            _B, _Clips, _T, _C, _H, _W = img.shape
            img = img.view(-1, _T, _C, _H, _W)
            feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
                img, txt, mask)
            _hidden_size = feat_img.shape[-1]
            mean_mask_img = mask_img.view(_B, _Clips, -1)
            mean_feat_img = feat_img.view(_B, _Clips, -1, _hidden_size)
            mean_feat_img = mean_feat_img.mean(dim=1)
            mean_mask_img = mean_mask_img[:, 0, :]
            return mean_feat_img, mean_mask_img, feat_txt, mask_txt

        elif typ == 'cross':
            out, _ = self.go_cross(
                feat_img, mask_img, feat_txt, mask_txt)
            out = self.fc(out[:, feat_img.shape[1], :]).squeeze()
            return out


if __name__ == '__main__':
    args = get_args(distributed=False)
    args.use_checkpoint = False
    args.num_gpus = T.cuda.device_count()
    if args.multi_clip_testing:
        args.size_batch = 10*args.num_gpus
    else:
        args.size_batch = 100*args.num_gpus
    assert os.path.exists(args.path_ckpt)

    print(args)
    ds_ret = Dataset_RetrievalTsEval(args, "test")

    log = {}
    model = T.nn.DataParallel(
        LAVENDER_RetrievalTsEval(args, ds_ret.tokzr).cuda())
    model.module.load_ckpt(args.path_ckpt)
    model.eval()

    for split in ['test']:  # ['val', 'test']:
        ds_ret = Dataset_RetrievalTsEval(args, split)
        dl = T.utils.data.DataLoader(
            ds_ret,
            batch_size=args.size_batch, shuffle=False,
            num_workers=args.n_workers, pin_memory=True,
            worker_init_fn=ds_ret.read_tsv)
        featv = {}
        featt = {}
        gt_txt2vid = ds_ret.gt_txt2vid
        for img, txt, mask, tid, vid in tqdm(dl, ascii=True):
            with T.no_grad():
                feat_img, mask_img, feat_txt, mask_txt = model(
                    typ='feat', img=img.cuda(), txt=txt.cuda(),
                    mask=mask.cuda())
            for t, v, f_i, m_i, f_t, m_t in zip(
                    tid, vid, *[
                        d.data.cpu().numpy()
                        for d in [feat_img, mask_img, feat_txt, mask_txt]]
                    ):
                if v not in featv:
                    featv[v] = {
                        'video': v, 'feat_img': f_i,
                        'mask_img': m_i}
                featt[t] = {
                    'tid': t, 'feat_txt': f_t, 'mask_txt': m_t}
        ds = Dataset_Product(featv, featt)
        dl = T.utils.data.DataLoader(ds,
                                     batch_size=args.size_batch,
                                     shuffle=False,
                                     num_workers=args.n_workers,
                                     pin_memory=True)
        print(f"number of videos: {len(ds.vid2idx)}")
        print(f"number of queires (by text): {len(ds.tid2idx)}")
        print(f"number of queries (before gathering rank): {len(ds_ret)}")
        rank = {}
        for (feat_txt, mask_txt, tid, feat_img,
                mask_img, vid) in tqdm(dl, ascii=True):
            with T.no_grad():
                out = model(typ='cross', feat_img=feat_img,
                            mask_img=mask_img,
                            feat_txt=feat_txt, mask_txt=mask_txt)
                out = T.sigmoid(out).data.cpu().numpy()
            for tid_, vid_, o in zip(tid, vid, out):
                i_v = ds.vid2idx[vid_]
                i_v, o = int(i_v), float(o)
                tid_ = tid_.item()

                if tid_ not in rank:
                    rank[tid_] = []
                rank[tid_].append([i_v, o])

        res = {'r@1': 0, 'r@5': 0, 'r@10': 0, 'median': []}
        print(f"number of queries (after gathering rank): {len(rank)}")
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
