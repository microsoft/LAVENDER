'''
Adapted from SwinBERT
'''
from utils.lib import *
from model import LAVENDER_Base
import torch.nn.functional as F
import torch.nn as nn


class CaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = T.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = T.topk(
                loss,
                k=int(loss.shape[0] * (1-self.drop_worst_ratio)),
                largest=False)

        loss = loss.mean()

        return loss


class LAVENDER_Captioning(LAVENDER_Base):
    def __init__(self, args, tokzr, is_decoder=True):
        super().__init__(args, tokzr)
        self.config.is_decoder = is_decoder
        bert = transformers.AutoModelForMaskedLM.from_pretrained(
            self.args.tokenizer, config=self.config)
        self.fc_mtm = bert.cls
        del bert

        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(
                0.02*T.randn(10, self.hidden_size))
        self.cap_prompt_txt_L = 0

    def forward(self, batch, is_decode=False):
        batch = defaultdict(lambda: None, batch)
        if is_decode:
            return self.generate(batch)
        else:
            return self.encode_forward(batch)

    def encode_forward(self, batch):
        if batch["input_ids"] is None:
            img, txt, mask = batch["img"], batch["txt"], batch["mask"]
            ans_mtm = batch["ans_mtm"]
            prompt = batch["prompt"]
            (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
            _h, _w = _H//32, _W//32

            feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
                img, txt, mask)
            ans_mtm, _, feat_txt = self.prepro_txt_inputs(
                ans_mtm, mask_txt, feat_txt, task_name="cap",
                prompt=prompt)
            if prompt is not None and self.args.enable_prompt:
                _L = len(prompt[0])
                self.cap_prompt_txt_L = _L
            elif self.args.enable_task_token:
                _L = 1  # for a task token
                self.cap_prompt_txt_L = _L
            else:
                _L = 0
                assert self.cap_prompt_txt_L == _L
            ans_mtm[:, :_L] = -1
            if _L > 0:
                mask_pretxt = T.ones_like(mask_txt)[:, :_L]
            else:
                mask_pretxt = None
            out, _ = self.go_cross(
                feat_img, mask_img, feat_txt,
                mask_txt, attn_mask_type=batch["attn_mask_type"],
                mask_pretxt=mask_pretxt)
            out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
            return {"out": out, "ans": ans_mtm}
        else:
            input_ids = batch["input_ids"]
            feat_txt = self.enc_txt(
                input_ids,
                attn_mask_type='seq2seq')
            txt_seq_len = input_ids.shape[-1]
            feat_img = batch["feat_img"]
            mask = batch["attention_mask"]
            cap_pretxt_feat = batch["cap_pretxt_feat"]
            # prompt = batch["prompt"]
            if feat_img is not None:
                _, img_seq_len, _ = feat_img.shape
                if cap_pretxt_feat is not None:
                    feat = T.cat([feat_img, cap_pretxt_feat, feat_txt], dim=1)
                    out_seq_start = img_seq_len + self.cap_prompt_txt_L
                    out_seq_end = (
                        img_seq_len + self.cap_prompt_txt_L + txt_seq_len)
                else:
                    feat = T.cat([feat_img, feat_txt], dim=1)
                    out_seq_start = img_seq_len
                    out_seq_end = img_seq_len+txt_seq_len
                mask = self.mask_ext(mask, mask.shape, mask.device)
                # safeguard fp16
                mask = mask.to(dtype=feat_txt.dtype)
                outputs = self.trsfr(
                    feat, mask, output_attentions=True, use_cache=True)
                sequence_output = outputs["last_hidden_state"][
                    :, out_seq_start:out_seq_end, :]
            else:
                raise NotImplementedError("Fast decoding with past key-values is not validated")
                assert not self.args.enable_task_token
                assert not self.args.enable_prompt
                feat = feat_txt
                past = batch["past_key_values"]
                mask = self.mask_ext(mask, mask.shape, mask.device)
                outputs = self.trsfr(
                    feat, mask, output_attentions=True, use_cache=True,
                    past_key_values=past)
                sequence_output = outputs["last_hidden_state"][
                    :, :txt_seq_len, :]
            class_logits = self.fc_mtm(sequence_output)
            past_key_values = outputs["past_key_values"]
            return {"logits": class_logits, "past": past_key_values}

    def generate(self, batch):
        """ Generates captions given image features
        """
        in_img, in_txt, in_mask = batch["img"], batch["txt"], batch["mask"]
        bos_token_id = self.cls_token_id
        pad_token_id = self.pad_token_id
        eos_token_ids = [self.sep_token_id]

        # default settings
        # # NOTE: num_keep_best is not equavilant to num_return_sequences
        # # num_keep_best is the number of hypotheses to keep in beam search
        # # num_return_sequences is the repeating times of input, coupled with
        # # do_sample=True can generate more than one samples per image
        self.num_keep_best = self.args.get('num_keep_best', 1)
        num_beams = self.args.get('num_beams', 1)
        num_return_sequences = self.args.get('num_return_sequences', 1)
        num_fsm_states = self.args.get('num_fsm_states', 1)
        do_sample = self.args.get('do_sample', False)
        temperature = self.args.get('gen_temperature', 1.)
        top_k = self.args.get('top_k', 0)
        top_p = self.args.get('top_p', 1)
        repetition_penalty = self.args.get('repetition_penalty', 1)

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(
            in_img, in_txt, in_mask)
        prompt = batch["prompt"]
        _, cap_pretxt_mask, cap_pretxt_feat = self.get_pretxt(
            mask_txt, task_name="cap", prompt=prompt)
        if cap_pretxt_mask is not None:
            self.cap_prompt_txt_L = cap_pretxt_mask.shape[1]
        else:
            # _L = 0  # no added task token or prompt
            assert self.cap_prompt_txt_L == 0
        self.cap_pretxt_feat = cap_pretxt_feat

        attention_mask = self.get_attn_mask(
            mask_img, mask_txt,
            attn_mask_type=batch["attn_mask_type"],
            mask_pretxt=cap_pretxt_mask)
        max_length = self.args.get(
            'max_gen_length', 20)
        batch_size = feat_img.shape[0]
        self.img_seq_len = feat_img.shape[1]
        self.max_seq_len = feat_txt.shape[1]
        # max(self.args.size_txt, feat_txt.shape[1])
        self.past_key_values = None

        assert feat_txt.shape == (
            batch_size, self.max_seq_len, self.hidden_size)
        input_ids = None

        if input_ids is None:
            input_ids = T.full(
                (batch_size, 1), bos_token_id, dtype=T.long,
                device=next(self.parameters()).device
            )
            # input_ids = T.cat([prompted_in_txt[:, :_L], input_ids], dim=1)
        else:
            assert input_ids.dim() == 2,\
                f"Input prompt of shape {input_ids.shape()}" +\
                "should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size,\
                f"Input prompt of shape {input_ids.shape()}" +\
                f"Input batch size must match image batch size {batch_size}"

        cur_len = input_ids.shape[1]
        assert num_return_sequences == 1,\
            "Only supporting num_return_sequences == 1, but got " +\
            f"{num_return_sequences} instead"
        effective_batch_size = batch_size

        num_expand = num_beams * num_fsm_states * num_return_sequences
        self.img_feats = self._expand_for_beams(feat_img, num_expand)
        if self.cap_pretxt_feat is not None:
            self.cap_pretxt_feat = self._expand_for_beams(
                self.cap_pretxt_feat, num_expand)
        self.full_attention_mask = self._expand_for_beams(
            attention_mask, num_expand)

        output = self._generate_no_beam_search(
            input_ids, cur_len, max_length, do_sample,
            temperature, top_k, top_p,
            repetition_penalty, pad_token_id,
            eos_token_ids, effective_batch_size)
        # if self.cap_prompt_txt_L > 0:
        #     output = (
        #         output[0][:, :, self.cap_prompt_txt_L:], output[1])
        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view(
            [input_shape[0] * num_expand] + input_shape[1:])
        return x

    def prepare_inputs_for_generation(self, curr_ids, past=None):

        # NOTE: if attention is on,
        # it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = T.full(
            (batch_size, 1), mask_token_id, dtype=T.long,
            device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len)
            return T.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = T.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            cap_pretxt_feat = self.cap_pretxt_feat
            if cap_pretxt_feat is None:
                assert self.cap_prompt_txt_L == 0
            full_len = (
                self.max_seq_len + self.cap_prompt_txt_L + self.img_seq_len)
            assert self.full_attention_mask.shape == (
                batch_size, full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = T.cat([T.cat([t00, t01], dim=2), T.cat([t10, t11],
                            dim=2)], dim=1)
                assert res.shape == (
                    t.shape[0], t.shape[1]-row_end+row_start,
                    t.shape[2]-col_end+col_start)
                return res

            img_feats = self.img_feats
            seq_end = self.img_seq_len + self.cap_prompt_txt_L + curr_len

            # seq_start = curr_len
            # seq_end = self.max_seq_len
            # attention_mask = _remove_rows_cols(
            #     self.full_attention_mask, seq_start,
            #     seq_end, seq_start, seq_end)
            attention_mask = self.full_attention_mask[
                :, :seq_end, :seq_end]
            past_key_values = None
        else:
            raise NotImplementedError(
                "Fast decoding with past key-value is not validated")
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = T.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]

            img_feats = None
            assert past[0][0].shape[2] == 1 + self.img_seq_len + start_pos
            past_key_values = []
            for layer_idx in range(len(past)):
                key, value = past[layer_idx]
                past_key_values.append((
                    key[:, :, :self.img_seq_len + start_pos],
                    value[:, :, :self.img_seq_len + start_pos],
                    ))

            # assert past[0].shape[0] == batch_size
            # past is a list of key values
            # key value of shape [
            #   batch_size, number_heads, seq_len, attn_hidden_size]
            # if self.past_key_values is None:
            #     assert start_pos == 1  # the first token after BOS
            #     assert past[0][0].shape[2] == 2 + self.img_seq_len
            #     self.past_key_values = []
            #     for layer_idx in range(len(past)):
            #         key, value = past[layer_idx]
            #         self.past_key_values.append((
            #             key[:, :, :self.img_seq_len + start_pos],
            #             value[:, :, :self.img_seq_len + start_pos],
            #             ))
            #     # # reorder to [img_feats, sentence]
            #     # self.prev_encoded_layers = [
            #     #         T.cat([x[:, 2:, :], x[:, :start_pos, :]], dim=1)
            #     #         for x in past]
            #     # i2i = self.full_attention_mask[
            #     #     :, :self.img_seq_len, :self.img_seq_len]
            #     # i2s = self.full_attention_mask[
            #     #     :, :self.img_seq_len, self.img_seq_len:]
            #     # s2i = self.full_attention_mask[
            #     #     :, self.img_seq_len:, :self.img_seq_len]
            #     # s2s = self.full_attention_mask[
            #     #     :, self.img_seq_len:, self.img_seq_len:]
            #     # self.full_attention_mask = T.cat(
            #     #         [T.cat([i2i, i2s], dim=2),
            #     #          T.cat([s2i, s2s], dim=2)], dim=1)
            # else:
            #     assert start_pos > 1
            #     assert past[0][0].shape[2] == 2
            #     # self.prev_encoded_layers = [
            #     #     T.cat([x, p[:, :-1, :]], dim=1)
            #     #     for x, p in zip(self.prev_encoded_layers, past)]
            #     for layer_idx in range(len(past)):
            #         key, value = past[layer_idx]
            #         p_key, p_value = self.past_key_values[layer_idx]
            #         new_key = T.cat([p_key, key[:, :, :-1, :]], dim=2)
            #         new_value = T.cat([p_value, value[:, :, :-1, :]], dim=2)
            #         self.past_key_values[layer_idx] = (new_key, new_value)

            attention_mask = self.full_attention_mask[
                :,
                self.img_seq_len+start_pos:self.img_seq_len+end_pos,
                :self.img_seq_len+end_pos]

        return {
            'input_ids': input_ids, 'feat_img': img_feats,
            'cap_pretxt_feat': cap_pretxt_feat,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'task': 'captioning_generation'}

    def _do_output_past(self):
        return self.config.is_decoder

    def _generate_no_beam_search(
            self, input_ids, cur_len, max_length,
            do_sample, temperature, top_k, top_p,
            repetition_penalty, pad_token_id, eos_token_ids, batch_size):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1,\
            'cannot generate >1 sentences in greedy search'
        # current position / max lengths /
        # length of generated sentences / unfinished sentences
        unfinished_sents = []
        if T._C._get_tracing_state():
            cur_unfinished = T.ones(1, dtype=input_ids)
        else:
            cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past)
            outputs = self(model_inputs)
            logits = outputs["logits"]

            if cur_len == 1:
                token_len = 2
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past():
                    token_len = cur_len + 1
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1

            assert logits.shape[1] == token_len
            next_token_logits = logits[:, next_token_idx, :]

            # if model has past, then set
            # the past variable to speed up decoding
            if self._do_output_past():
                past = outputs["past"]

            # repetition penalty from CTRL paper
            # (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to
                        # multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[
                                i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[
                                i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature =>
                #   more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = T.multinomial(
                    F.softmax(next_token_logits, dim=-1),
                    num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = T.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(
                next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = T.gather(
                _scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (
                1 - cur_unfinished)
            input_ids = T.cat(
                [input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            # for t in input_ids:
            #     print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(
                    tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence,
            # or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(
                cur_unfinished.to(dtype=T.bool), eos_token_ids[0])

        logprobs = T.cat(logprobs, dim=1)
        unfinished_sents = T.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(
                batch_size, pad_len).fill_(pad_token_id)
            input_ids = T.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)



def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < T.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = T.sort(logits, descending=True)
        cumulative_probs = T.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
