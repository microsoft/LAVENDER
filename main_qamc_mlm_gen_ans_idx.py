from utils.lib import *
from main_qamc_mlm_head import (
    Dataset_QAMC_MLM_Head, VIOLET_QAMC_MLM_Head,
    Agent_QAMC_MLM_Head, get_tsv_dls)
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import (
    NoOp, is_main_process,
    get_rank, get_world_size, iter_tqdm)


class Dataset_QAMC_MLM_Head_GEN(Dataset_QAMC_MLM_Head):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super().__init__(
            args, img_tsv_path, txt, id2lineidx, split,
            tokzr=tokzr)
        self.ans_tok_ids = self.tokzr.convert_tokens_to_ids(
            [f"{i}" for i in range(self.args.size_option)])

    def append_mask(self, tokens, padding_len):
        tokens = [self.tokzr.cls_token] + tokens + [self.tokzr.mask_token] + [
                self.tokzr.sep_token] + [self.tokzr.pad_token] * (padding_len)
        return tokens

    def prepend_mask(self, tokens, padding_len):
        tokens = [self.tokzr.mask_token, self.tokzr.cls_token] + tokens + [
                    self.tokzr.sep_token
                ] + [self.tokzr.pad_token] * (padding_len)
        return tokens

    def replace_cls(self, tokens, padding_len):
        tokens = [self.tokzr.mask_token] + tokens + [
                    self.tokzr.sep_token
                ] + [self.tokzr.pad_token] * (padding_len)
        return tokens

    def insert_mask(self, tokens, padding_len):
        tokens = [self.tokzr.cls_token] + tokens + [
                self.tokzr.sep_token
                ] + [self.tokzr.pad_token] * (padding_len)
        if len(tokens) < 10:
            tokens += [self.tokzr.mask_token]
        else:
            tokens = tokens[:10] + [self.tokzr.mask_token] + tokens[10:]
        return tokens

    def str2txt(self, s):
        # txt, mask = super().str2txt(s)
        # txt, mask = self.append_mask_tok2txt(txt, mask)
        # return txt, mask
        tokens = self.tokzr.tokenize(s)
        tokens = tokens[:self.args.size_txt-1]
        padding_len = self.args.size_txt - len(tokens)
        if self.args.mask_pos == "append":
            tokens = self.append_mask(tokens, padding_len)
        elif self.args.mask_pos == "prepend":
            tokens = self.prepend_mask(tokens, padding_len)
        elif self.args.mask_pos == "insert":
            tokens = self.insert_mask(tokens, padding_len)
        elif self.args.mask_pos == "replace":
            tokens = self.replace_cls(tokens, padding_len)
        txt = self.tokzr.convert_tokens_to_ids(tokens)

        mask = [1 if w != self.pad_token_id else 0 for w in txt]
        mask = T.LongTensor(mask)
        txt = T.LongTensor(txt)
        return txt, mask

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_img_or_video(b)
        ans_idx = item['answer']
        ans_tok_id = self.tokzr.convert_tokens_to_ids([f"{ans_idx}"])[0]
        question = item['question']

        for i in range(self.args.size_option):
            answer = item[f'option_{i}']
            answer = f"option {i}: " + answer
            question = self.concat_txt(question, answer)

        txt, mask = self.str2txt(question)
        mask_ans = T.ones(txt.shape).long() * -1
        mask_ans[txt == self.mask_token_id] = ans_tok_id

        return img, txt, mask, mask_ans, ans_idx

    @property
    def prompt_text(self):
        return "which answer is correct, from " + \
            f"{list(range(self.args.size_option))}?"

    def collate_batch(self, inputs):
        img, txt, mask, mask_ans, ans_idx = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_mask_ans = T.stack(mask_ans, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)
        ans_idx = T.LongTensor(ans_idx)
        batch = {
            "img": all_imgs, "txt": all_txts,
            "mask": all_masks, "mask_ans": all_mask_ans,
            "ans_idx": ans_idx}
        return batch


class VIOLET_QAMC_MLM_Head_GEN(VIOLET_QAMC_MLM_Head):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        img, txt, mask = [
                batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["mask_ans"]

        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            img, txt, mask)
        ans, mask_txt, feat_txt = self.prepro_txt_inputs(
            ans, mask_txt, feat_txt, task_name=batch["task_name"],
            prompt=batch["prompt"])
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if self.args.temporal_fusion == "mean":
            _T = 1
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        return out, ans


class Agent_QAMC_MLM_Head_GEN(Agent_QAMC_MLM_Head):
    def __init__(self, args, model, ans_tok_ids):
        super(Agent_QAMC_MLM_Head, self).__init__(args, model)
        self.ans_tok_ids = ans_tok_ids
        if args.freeze_violet:
            self.model.freeze()

    def step(self, batch, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            out, ans = out
        if is_train:
            ans = ans.flatten(0, 1)
            out = out.flatten(0, len(out.shape)-2)
            ans = ans.flatten(0, len(ans.shape)-1)
            ls = self.loss_func(out, ans)
            self.backward_step(ls)
            return ls.item()
        else:
            _B, _ = ans.shape
            p_all_ans_toks = out[:, :, self.ans_tok_ids]
            ans_mtm = ans
            out_mtm = p_all_ans_toks[ans_mtm != -1]
            out_mtm = out_mtm / out_mtm.sum(dim=-1).view(_B, 1)
            out_mtm = out_mtm.view(_B, -1)
            out_mtm = T.argmax(out_mtm, dim=-1)
            # ans_idx = T.LongTensor(ans_idx, device=out_mtm.device)
            ans_idx = batch["ans_idx"]
            ac = (out_mtm == ans_idx).float().tolist()
            return ac


if __name__ == '__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    dl_tr, dl_vl, dl_ts = get_tsv_dls(
                args, Dataset_QAMC_MLM_Head_GEN, tokzr=tokzr)
    if args.size_epoch == 0:
        args.max_iter = 1
    else:
        args.max_iter = len(dl_tr) * args.size_epoch

    model = VIOLET_QAMC_MLM_Head_GEN(args, tokzr=tokzr)
    model.load_ckpt(args.path_ckpt)
    if args.reinit_head:
        model.reinit_head()
    model.cuda()

    if args.distributed:
        LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                    f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s' % (
        args.path_output, args.task,
        datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_QAMC_MLM_Head_GEN(
        args, model, dl_ts.dataset.ans_tok_ids
    )
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
        agent.setup_wandb()
        LOGGER.info("Start training....")

        for e in iter_tqdm(range(args.size_epoch)):

            ls_tr = agent.go_dl(e+1, dl_tr, True)

            ac_vl = agent.go_dl(e+1, dl_vl, False)
            ac_ts = agent.go_dl(e+1, dl_ts, False)

            agent.log['ls_tr'].append(ls_tr)
            agent.log['ac_vl'].append(ac_vl)
            agent.log['ac_ts'].append(ac_ts)
            agent.log_dict_to_wandb({"ac_vl": ac_vl})
            agent.log_dict_to_wandb({"ac_ts": ac_ts})
            LOGGER.info('Ep %d: %.6f %.2f %.2f' % (
                e+1, ls_tr, ac_vl*100, ac_ts*100))
            agent.save_model(e+1)
        best_vl, best_ts = agent.best_epoch()
        LOGGER.info(f'Best val @ ep {best_vl[0]+1}, {best_vl[1]*100:.2f}')
        LOGGER.info(f'Best test @ ep {best_ts[0]+1}, {best_ts[1]*100:.2f}')
