from utils.lib import *
from dataset import get_tsv_dls
from main_qaoe_task_specific import Dataset_QAOE_TS
from main_qaoe_mlm_lsmdc_fib import Agent_QAOE_MLM_LSMDC
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import (
    NoOp, is_main_process, all_gather,
    get_rank, get_world_size, iter_tqdm)
from main_qaoe_mlm_lsmdc_fib import LAVENDER_QAOE_MLM


class Dataset_QAOE_MLM(Dataset_QAOE_TS):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super().__init__(
            args, img_tsv_path, txt, id2lineidx, split, tokzr=tokzr)

    def str2txt(self, s):
        # txt, mask = super().str2txt(s)
        # txt, mask = self.append_mask_tok2txt(txt, mask)
        # return txt, mask
        tokens = self.tokzr.tokenize(s)
        tokens = tokens[:self.args.size_txt-1]
        padding_len = self.args.size_txt - len(tokens)
        tokens = [self.tokzr.cls_token] + tokens + self.tokzr.tokenize(f"answer: {self.tokzr.mask_token}") + [self.tokzr.sep_token] + [self.tokzr.pad_token] * (padding_len)
        txt = self.tokzr.convert_tokens_to_ids(tokens)
        # if version.parse(transformers.__version__) >= version.parse("4.16.1"):
        #     txt = self.tokzr.encode(s)
        #     txt = txt[:self.args.size_txt-1]
        #     padding_len = self.args.size_txt-len(txt)
        #     txt = txt + [0]*(padding_len)
        # else:
        #     txt = self.tokzr.encode(
        #         s, padding='max_length', max_length=self.args.size_txt,
        #         truncation=True)
        mask = [1 if w != self.pad_token_id else 0 for w in txt]
        mask = T.LongTensor(mask)
        txt = T.LongTensor(txt)
        return txt, mask

    @property
    def prompt_text(self):
        return "answer the question about the video."

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        if video_id in self.id2lineidx:
            lineidx = self.id2lineidx[video_id]
            b = self.seek_img_tsv(lineidx)[2:]
            img = self.get_img_or_video(b)
        else:
            print(f"video missing: {video_id}")
            img = T.zeros(
                (self.args.size_frame, 3,
                 self.args.size_img, self.args.size_img))

        q = item['question']
        txt, mask = self.str2txt(q)

        if self.args.size_vocab > 0:
            # self-defined vocabularies
            ans_id = item['answer']
        else:
            ans = item['answer_text']
            ans_id = self.tokzr.convert_tokens_to_ids([ans])[0]
            if ans_id == self.unk_token_id:
                # handling [UNK]
                ans_id = -1
        if video_id not in self.id2lineidx:
            print(f"video {video_id} not found")
            ans_id = -1
        mask_ans = T.ones(txt.shape).long() * -1
        mask_ans[txt == self.mask_token_id] = ans_id
        return img, txt, mask, mask_ans

    def collate_batch(self, inputs):
        img, txt, mask, mask_ans = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_mask_ans = T.stack(mask_ans, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {
            "img": all_imgs, "txt": all_txts,
            "mask": all_masks, "mask_ans": all_mask_ans}
        return batch

    # def get_prompt(self):
    #     return None


class Agent_QAOE_MLM(Agent_QAOE_MLM_LSMDC):
    def __init__(self, args, model):
        super().__init__(args, model)

    def go_dl(self, ep, dl, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        ret = defaultdict(list)  # {'ls': [], 'ac_1': [], 'ac_5': []}
        idx = 0
        for idx, batch in enumerate(dl):
            if idx % self.args.logging_steps == 0 and is_train:
                LOGGER.info(self.log_memory(ep, idx+1))
            if self.args.enable_prompt:
                batch["prompt"] = dl.dataset.get_prompt()
            elif self.args.enable_task_token:
                batch["task_name"] = "oe"
            batch = self.prepare_batch(batch)
            r = self.step(batch, is_train)
            ret = {
                k: ret[k]+l if isinstance(l, list) else ret[k]+[l]
                for k, l in r.items()}

        if idx % self.args.logging_steps != 0 and is_train:
            LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = defaultdict(list)
        for ret_per_rank in all_gather(ret):
            for k in ret_per_rank:
                gathered_ret[k].extend(ret_per_rank[k])
        ret_all = {
            k: float(np.average(gathered_ret[k])) for k in ret}
        return ret_all


if __name__ == '__main__':
    args = get_args()
    # setting size_vocab to -1, to leverage the pre-trained tokenizer for answer generation
    args.size_vocab = -1
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    dl_tr, dl_vl, dl_ts = get_tsv_dls(
        args, Dataset_QAOE_MLM, tokzr=tokzr)

    if args.size_epoch == 0:
        args.max_iter = 1
    else:
        args.max_iter = len(dl_tr) * args.size_epoch
    args.actual_size_test = len(dl_ts.dataset)

    model = LAVENDER_QAOE_MLM(args, tokzr=tokzr)
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
    agent = Agent_QAOE_MLM(args, model)
    if args.distributed:
        agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process():
        add_log_to_file('%s/stdout.txt' % (args.path_output))
    else:
        LOGGER = NoOp()
    # DIST.barrier()
    LOGGER.info("Saved training meta infomation...")

    if os.path.exists(args.path_ckpt):
        LOGGER.info("Zero-shot Evaluation")
        if len(dl_vl):
            ac_vl = agent.go_dl(0, dl_vl, False)
            LOGGER.info(
                f'ZS (val): {ac_vl["ac_1"]*100:.2f}, {ac_vl["ac_5"]*100:.2f}')
        if len(dl_ts):
            ac_ts = agent.go_dl(0, dl_ts, False)
            LOGGER.info(
                f'ZS (test): {ac_ts["ac_1"]*100:.2f}, {ac_ts["ac_5"]*100:.2f}')
            if (
                    hasattr(args, "size_test") and
                    args.size_test != args.actual_size_test):
                adjusted_ac_ts_1 = ac_ts[
                    'ac_1'] * args.actual_size_test / args.size_test * 100
                adjusted_ac_ts_5 = ac_ts[
                    'ac_5'] * args.actual_size_test / args.size_test * 100
                LOGGER.info(
                    f'ZS (test, adjusted): {adjusted_ac_ts_1:.2f}'
                    f', {adjusted_ac_ts_5:.2f}')
    else:
        LOGGER.info("No pre-trained weight, skip zero-shot Evaluation")

    if args.size_epoch:
        LOGGER.info("Start training....")
        for e in iter_tqdm(range(args.size_epoch)):

            ls_tr = agent.go_dl(e+1, dl_tr, True)
            LOGGER.info(
                f'Ep {e}, Loss (train): {ls_tr["ls"]*100:.4e}')

            if len(dl_vl):
                ac_vl = agent.go_dl(e+1, dl_vl, False)
                for k in ac_vl:
                    agent.log[f'{k}_vl'].append(ac_vl[k])
                LOGGER.info(
                    f'Ep {e}, Acc (val): {ac_vl["ac_1"]*100:.2f}, '
                    f'{ac_vl["ac_5"]*100:.2f}')
            if len(dl_ts):
                ac_ts = agent.go_dl(e+1, dl_ts, False)
                LOGGER.info(
                    f'Ep {e}, Acc (test): {ac_ts["ac_1"]*100:.2f}, '
                    f'{ac_ts["ac_5"]*100:.2f}')
                if (
                        hasattr(args, "size_test") and
                        args.size_test != args.actual_size_test):
                    adjusted_ac_ts_1 = ac_ts[
                        'ac_1'] * args.actual_size_test / args.size_test
                    adjusted_ac_ts_5 = ac_ts[
                        'ac_5'] * args.actual_size_test / args.size_test
                    agent.log['ac_1_ts'].append(adjusted_ac_ts_1)
                    agent.log['ac_5_ts'].append(adjusted_ac_ts_5)
                    LOGGER.info(
                        f'Ep {e}, Acc (test, adjusted): {adjusted_ac_ts_1*100:.2f}'
                        f', {adjusted_ac_ts_5*100:.2f}')
                else:
                    for k in ac_ts:
                        agent.log[f'{k}_ts'].append(ac_ts[k])
            agent.save_model(e+1)
        best_vl, best_ts = agent.best_epoch()
        LOGGER.info(f'Best val @ ep {best_vl[0]+1}, {best_vl[1]*100:.2f}')
        LOGGER.info(f'Best test @ ep {best_ts[0]+1}, {best_ts[1]*100:.2f}'
                    f' (adjusted)')
