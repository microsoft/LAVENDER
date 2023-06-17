from utils.lib import *
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import (
    is_main_process,
    get_rank, get_world_size, iter_tqdm, all_gather,
    NoOp)
from dataset import get_tsv_dls
from main_retrieval_task_specific import (
    Dataset_Retrieval_TS)
from agent import Agent_Base
from model import LAVENDER_Base


class Dataset_Retrieval_MLM(Dataset_Retrieval_TS):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super().__init__(
            args, img_tsv_path, txt, id2lineidx, split, tokzr=tokzr)

    def str2txt(self, s):
        txt, mask = super().str2txt(s)
        txt, mask = self.append_mask_tok2txt(txt, mask)
        return txt, mask

    @property
    def prompt_text(self):
        return "is the video-text paired, true or false?"


class LAVENDER_Retrieval_MLM(LAVENDER_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        # bert = transformers.BertForMaskedLM.from_pretrained(
        #     './_models/huggingface_transformers/bert-base-uncased')
        # config = bert.config
        # self.fc_mtm = BertOnlyMLMHead(config)
        # del bert
        bert = transformers.AutoModelForMaskedLM.from_pretrained(
            self.args.tokenizer)
        if isinstance(bert, transformers.RobertaForMaskedLM):
            self.fc_mtm = bert.lm_head
        else:
            self.fc_mtm = bert.cls
        del bert

        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(
                0.02*T.randn(10, self.hidden_size))

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
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


class Agent_Retrieval_MLM(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}

    def step(self, batch, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            out, ans = out
        if is_train:
            out = out.flatten(0, len(out.shape)-2)
            ans = ans.flatten(0, len(ans.shape)-1)
            ls = self.loss_func(out, ans)
            self.backward_step(ls)
            return ls.item()
        else:
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
            if self.args.enable_prompt:
                batch["prompt"] = dl.dataset.get_prompt()
            elif self.args.enable_task_token:
                batch["task_name"] = "vtm"
            batch = self.prepare_batch(batch)
            curr_ret = self.step(batch, is_train)
            if isinstance(curr_ret, list):
                ret.extend(curr_ret)
            else:
                ret.append(curr_ret)

        if idx % self.args.logging_steps != 0 and is_train:
            LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = []
        for ret_per_rank in all_gather(ret):
            gathered_ret.extend(ret_per_rank)
        ret = float(np.average(gathered_ret))
        return ret


if __name__ == '__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    dl_tr, dl_vl, dl_ts = get_tsv_dls(
        args, Dataset_Retrieval_MLM, tokzr=tokzr)

    if args.size_epoch == 0:
        args.max_iter = 1
    else:
        args.max_iter = len(dl_tr) * args.size_epoch

    model = LAVENDER_Retrieval_MLM(args, tokzr=tokzr)
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed:
        LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                    f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s' % (
        args.path_output, args.task,
        datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_Retrieval_MLM(args, model)
    if args.distributed:
        agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process():
        add_log_to_file('%s/stdout.txt' % (args.path_output))
    else:
        LOGGER = NoOp()
    # DIST.barrier()
    LOGGER.info("Saved training meta infomation, start training ...")

    if os.path.exists(args.path_ckpt):
        LOGGER.info("Zero-shot Evaluation")
        ac_vl = agent.go_dl(0, dl_vl, False)
        ac_ts = agent.go_dl(0, dl_ts, False)
        LOGGER.info('ZS: %.2f %.2f' % (
                ac_vl*100, ac_ts*100))
    else:
        LOGGER.info("No pre-trained weight, skip zero-shot Evaluation")

    if args.size_epoch:
        LOGGER.info("Start training....")
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
