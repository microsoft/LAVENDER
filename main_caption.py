from utils.lib import *
from dataset import TsvCompositeDataset, get_dl
from model_for_captioning import CaptioningLoss, LAVENDER_Captioning
from agent import Agent_Base
from utils.args import get_args
from utils.logger import LOGGER, RunningMeter, add_log_to_file
from utils.dist import (
    is_main_process, synchronize, all_gather,
    get_rank, get_world_size, iter_tqdm, NoOp)
from utils.misc import ensure_directory
from utils.tsv_file_ops import (
    tsv_writer,  reorder_tsv_keys)
from evalcap.utils_caption_evaluate import (
        evaluate_on_coco_caption)


class Dataset_Caption(TsvCompositeDataset):
    def __init__(self, args, yaml_file, split, tokzr=None):
        super().__init__(
            args, yaml_file, split, size_frame=args.size_frame, tokzr=tokzr)
        if args.data_ratio != 1:
            self.get_partial_data()

    def __getitem__(self, idx):
        raw_data = self.get_img_txt_pair(idx)
        img = raw_data['img']
        vid = raw_data['img_key']
        raw_txt = raw_data['caption']
        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, vid

    @property
    def prompt_text(self):
        return "write a description about the video."

    def collate_batch(self, inputs):
        img, txt, mask, vid = map(list, unzip(inputs))
        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {
            "img": all_imgs, "txt": all_txts,
            "mask": all_masks, "img_keys": vid}
        return batch


class Agent_Captioning(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        loss_config = {
            'label_smoothing': getattr(args, 'label_smoothing', 0),
            'drop_worst_ratio': getattr(args, 'drop_worst_ratio', 0),
            'drop_worst_ratio': getattr(args, 'drop_worst_ratio', 0)}
        self.loss_func = CaptioningLoss(loss_config).cuda()
        self.log = defaultdict(list)
        self.running_meter = {
            'ls_tr': RunningMeter('ls_tr'),
            'ac_tr': RunningMeter('ac_tr')}

    def masking(self, txt, p_mask=0.15):
        (_B, _X) = txt.shape

        spc_txt = T.logical_or(
            txt == self.pad_token_id, txt == self.mask_token_id)

        ans_mtm = T.ones(txt.shape).long() * -1

        if p_mask <= 0:
            return {"txt": txt, "ans_mtm": ans_mtm}

        for i in range(_B):
            mask_mtm = T.where(T.logical_and(
                T.logical_not(spc_txt[i]), T.rand(_X) < p_mask))[0]

            for p in mask_mtm:
                ans_mtm[i][p], txt[i][p] = txt[i][p], self.mask_token_id
        return {"txt": txt, "ans_mtm": ans_mtm}

    def test(self, test_dataloader, predict_file):
        tokenizer = test_dataloader.dataset.tokzr
        world_size = get_world_size()
        if world_size == 1:
            cache_file = predict_file
        else:
            cache_file = (
                op.splitext(predict_file)[0] +
                f'_{get_rank()}_{world_size}' +
                op.splitext(predict_file)[1])

        self.model.eval()

        # def gen_rows():
        time_meter = 0
        all_preds = []

        with T.no_grad():
            for step, batch in tqdm(
                    enumerate(test_dataloader)):
                batch['task'] = 'captioning_generation'
                batch["attn_mask_type"] = "seq2seq" # self.args.attn_mask_type

                if self.args.enable_prompt:
                    batch["prompt"] = test_dataloader.dataset.get_prompt()
                batch = self.prepare_batch(batch)
                tic = time.time()
                # captions, logprobs
                outputs = self.model(batch, is_decode=True)
                time_meter += time.time() - tic
                # batch_size * num_keep_best * max_len
                all_caps = outputs[0]
                all_confs = T.exp(outputs[1])
                img_keys = batch["img_keys"]

                for img_key, caps, confs in zip(
                        img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(
                            cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, T.Tensor):
                        img_key = img_key.item()
                    # yield img_key, json.dumps(res)
                    all_preds.append([img_key, json.dumps(res)])

            LOGGER.info(
                f"Inference model computing time: "
                f"{time_meter / (step+1)} seconds per batch")

        # tsv_writer(gen_rows(), cache_file)
        # tsv_writer(all_preds, cache_file)
        # cache_ready = os.path.exists(cache_file)
        # cache_ready_all = all_gather(cache_ready)
        # cache_ready = os.path.exists(cache_file)
        gathered_all_preds = []
        for preds in all_gather(all_preds):
            gathered_all_preds.extend(preds)

        if is_main_process():
            tsv_writer(gathered_all_preds, predict_file)
            # if world_size > 1:
            #     cache_files = [
            #         op.splitext(predict_file)[0] + f'_{i}_{world_size}' +
            #         op.splitext(predict_file)[1]
            #         for i in range(world_size)]
            #     # cache_ready_all = []
            #     while not all(cache_ready_all):
            #         print(f"Cache not ready, retrying..., {cache_ready_all}")
            #         cache_ready_all = [
            #             os.path.exists(cf)
            #             for cf in cache_files]
            #     concat_tsv_files(cache_files, predict_file)
            #     delete_tsv_files(cache_files)
            reorder_tsv_keys(
                predict_file,
                test_dataloader.dataset.image_keys, predict_file)
        synchronize()

    def get_predict_file(self, ep, output_dir, data_yaml_file):
        args = self.args
        cc = [f'ep{ep}_pred']
        # example data_yaml_file: _datasets/coco_caption/test.yaml
        data = data_yaml_file.split('/')[-2]
        if data != 'coco_caption':
            cc.append(data)
        cc.append(op.splitext(op.basename(data_yaml_file))[0])
        if hasattr(args, 'num_beams'):
            cc.append('beam{}'.format(args.num_beams))
        cc.append('max{}'.format(args.max_gen_length))
        if hasattr(args, 'use_asr') and args.use_asr:
            cc.append('w_asr')
        if hasattr(args, 'num_keep_best') and args.num_keep_best != 1:
            cc.append('best{}'.format(args.num_keep_best))
        return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))

    def get_evaluate_file(self, predict_file):
        assert predict_file.endswith('.tsv')
        return op.splitext(predict_file)[0] + '.eval.json'

    def evaluate(self, ep, val_dataloader):
        self.model.eval()
        objects = [None]
        output_dir = op.join(
            self.args.path_output, "caption_predictions")
        if is_main_process():
            ensure_directory(output_dir)
        predict_file = self.get_predict_file(
            ep, output_dir,
            val_dataloader.dataset.yaml_file)
        if op.isfile(predict_file):
            LOGGER.info('Skip predict. {} already exists'.format(predict_file))
        else:
            self.test(val_dataloader, predict_file)

        synchronize()
        evaluate_file = self.get_evaluate_file(predict_file)
        if is_main_process():
            caption_file = (
                val_dataloader.dataset.get_caption_file_in_coco_format())
            if op.isfile(evaluate_file):
                LOGGER.info(f'Skip evaluation. {evaluate_file} already exists')
            else:
                data = val_dataloader.dataset.yaml_file.split('/')[-2]
                if 'nocaps' not in data:
                    result = evaluate_on_coco_caption(
                        predict_file, caption_file, outfile=evaluate_file)
                    # LOGGER.info(f'evaluation result: {str(result)}')
                    LOGGER.info(f'evaluation result saved to {evaluate_file}')
                    objects = [result]
        world_size = get_world_size()
        if world_size > 1:
            DIST.broadcast_object_list(
                objects, src=0)
        result = objects[0]
        return result

    def train_step(self, batch):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            logits, ans = out["out"], out["ans"]
            # masked_ids = ans[ans != -1]   # remove padded target
            # logits = logits[ans != -1]
            ls = self.loss_func(logits[ans != -1].float(), ans[ans != -1])
            self.backward_step(ls)
            pred = T.argmax(logits, dim=-1)
            acc = (
                float((pred == ans).sum() / (ans != -1).sum())
                if (ans != -1).sum() > 0 else 0)
            return ls.item(), acc

    def log_train(self):
        ls_tr = self.running_meter['ls_tr'].val
        ac_tr = self.running_meter['ac_tr'].val
        log_info = self.log_memory()
        if ls_tr is not None and ac_tr is not None:
            log_info += f" ls_tr: {ls_tr:.2e} ac_tr: {ac_tr*100:.2f}"
        return log_info

    def train(self, ep, dl):
        self.model.train()
        ret_ls = []
        ret_ac = []
        for idx, batch in enumerate(dl):
            self.global_step += 1
            if idx % self.args.logging_steps == 0:
                LOGGER.info(self.log_train())
            masked_batch = self.masking(
                batch['txt'], p_mask=self.args.p_mask)
            batch.update(masked_batch)
            if self.args.enable_prompt:
                batch["prompt"] = dl.dataset.get_prompt()
            batch = self.prepare_batch(batch)
            batch["attn_mask_type"] = "seq2seq"  
            ls, ac = self.train_step(batch)
            self.running_meter['ls_tr'](ls)
            self.running_meter['ac_tr'](ac)
            ret_ls.append(ls)
            ret_ac.append(ac)

        if idx % self.args.logging_steps != 0:
            LOGGER.info(self.log_train())

        ret_ls = float(float(np.average(ret_ls)))
        ret_ac = float(float(np.average(ret_ac)))
        if self.args.distributed:
            ret_ls = self.reduce_mean(ret_ls)
            ret_ac = self.reduce_mean(ret_ac)
        return {'loss': ret_ls, 'acc': ret_ac}

    def best_epoch(self, metric, split):
        if not hasattr(self, "log"):
            raise NotImplementedError("no log to find the best epoch")
        val_index = np.argmax(
            self.log[f"{split}_{metric}"])
        val_max = self.log[f"{split}_{metric}"][val_index]
        return (val_index, val_max)


if __name__ == '__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    ds_tr = Dataset_Caption(args, args.train_yaml, 'train', tokzr=tokzr)
    dl_tr = get_dl(ds_tr, args, collate_fn=ds_tr.collate_batch)
    ds_vl = Dataset_Caption(args, args.val_yaml, 'val', tokzr=tokzr)
    dl_vl = get_dl(ds_vl, args, collate_fn=ds_vl.collate_batch)
    log_data_len = f"data_ratio: {args.data_ratio}"
    log_data_len += f", train: {len(ds_tr)}"
    log_data_len += f", val: {len(ds_vl)}"
    if "test_yaml" in args:
        ds_ts = Dataset_Caption(args, args.test_yaml, 'test', tokzr=tokzr)
        dl_ts = get_dl(ds_ts, args, collate_fn=ds_ts.collate_batch)
        log_data_len += f", test: {len(ds_ts)}"
    else:
        dl_ts = None
    LOGGER.info(log_data_len)

    if args.size_epoch == 0:
        args.max_iter = 1
    else:
        args.max_iter = len(dl_tr) * args.size_epoch

    model = LAVENDER_Captioning(
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
    agent = Agent_Captioning(args, model)
    if args.distributed:
        agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process():
        add_log_to_file('%s/stdout.txt' % (args.path_output))
    else:
        LOGGER = NoOp()

    LOGGER.info("Zero shot evaluation ...")
    res_vl = agent.evaluate(0, dl_vl)
    for k in res_vl:
        agent.log[f'vl_{k}'].append(res_vl[k])
    LOGGER.info(f'Ep 0, val: {json.dumps(res_vl)}')
    if dl_ts is not None:
        res_ts = agent.evaluate(0, dl_ts)
        for k in res_ts:
            agent.log[f'ts_{k}'].append(res_ts[k])
        LOGGER.info(f'Ep 0, test: {json.dumps(res_ts)}')

    if args.size_epoch:
        LOGGER.info("Saved training meta infomation, start training....")

        for e in iter_tqdm(range(args.size_epoch)):
            res_tr = agent.train(e+1, dl_tr)

            res_vl = agent.evaluate(e+1, dl_vl)
            for k in res_tr:
                agent.log[f'tr_{k}'].append(res_tr[k])
            for k in res_vl:
                agent.log[f'vl_{k}'].append(res_vl[k])
            if dl_ts is not None:
                res_ts = agent.evaluate(e+1, dl_ts)
                for k in res_ts:
                    agent.log[f'ts_{k}'].append(res_ts[k])
            LOGGER.info(f'Ep {e+1}: '
                        f'train: {json.dumps(res_tr)}, '
                        f'val: {json.dumps(res_vl)}')
            if dl_ts is not None:
                LOGGER.info(f'/t/t test: {json.dumps(res_ts)}')
            agent.save_model(e+1)
        metric = 'CIDEr'
        best_vl = agent.best_epoch(metric, "vl")
        LOGGER.info(
            f'Best {metric} on val @ ep {best_vl[0]}, {best_vl[1]*100:.2f}')
        if dl_ts is not None:
            best_tst = agent.best_epoch(metric, "ts")
            LOGGER.info(
                f'Best {metric} on test @ ep {best_tst[0]}, {best_tst[1]*100:.2f}')
