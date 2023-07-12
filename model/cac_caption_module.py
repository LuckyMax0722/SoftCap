import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from utils.box_util import box3d_iou_batch_tensor


class CACModule(nn.Module):
    def __init__(self, vocabulary, embeddings, emb_size=300, feat_size=32, hidden_size=300, num_proposals=128,
                 use_relation=True):
        super().__init__()

        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        self.num_locals = CONF.graph_module.num_locals

        self.use_relation = use_relation

        # top-down recurrent module
        self.map_topdown = nn.Sequential(
            nn.Linear(hidden_size + feat_size + emb_size, emb_size),
            nn.ReLU()
        )

        self.recurrent_cell_1 = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )

        # top-down attention module
        self.map_feat = nn.Linear(feat_size, hidden_size, bias=False)
        self.map_hidd = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attend = nn.Linear(hidden_size, 1, bias=False)

        # language recurrent module
        self.map_lang = nn.Sequential(
            nn.Linear(feat_size + hidden_size, emb_size),
            nn.ReLU()
        )

        self.recurrent_cell_2 = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )

        # 输出分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, self.num_vocabs),
        )

    def step(self, step_input, target_feat, obj_feats, hidden_1, hidden_2, object_masks):
        '''
                    recurrent step

                    Args:
                        step_input: current word embedding, (batch_size, emb_size)
                        target_feat: object feature of the target object, (batch_size, feat_size)
                        obj_feats: object features of all detected objects, (batch_size, num_proposals, feat_size)
                        hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                        hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)

                    Returns:
                        hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                        hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)
                        masks: attention masks on proposals, (batch_size, num_proposals, 1)
                '''

        # fuse inputs for top-down module
        step_input = torch.cat([step_input, hidden_2, target_feat], dim=-1)
        step_input = self.map_topdown(step_input)

        # top-down recurrence
        hidden_1 = self.recurrent_cell_1(step_input, hidden_1)

        # top-down attention
        combined = self.map_feat(obj_feats)  # batch_size, num_proposals, hidden_size
        combined += self.map_hidd(hidden_1).unsqueeze(1)  # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.attend(combined)  # batch_size, num_proposals, 1
        scores.masked_fill_(object_masks == 0, float('-1e30'))

        masks = F.softmax(scores, dim=1)  # batch_size, num_proposals, 1
        attended = obj_feats * masks
        attended = attended.sum(1)  # batch_size, feat_size

        # fuse inputs for language module
        lang_input = torch.cat([attended, hidden_1], dim=-1)
        lang_input = self.map_lang(lang_input)

        # language recurrence
        hidden_2 = self.recurrent_cell_2(lang_input, hidden_2)  # batch_size, hidden_size

        return hidden_1, hidden_2, masks

    def step_val(self, step_input, target_feat, obj_feats, hidden_1, hidden_2, object_masks):
        '''
                    recurrent step

                    Args:
                        step_input: current word embedding, (emb_size)
                        target_feat: object feature of the target object, (feat_size)
                        obj_feats: object features of all detected objects, (num_proposals, feat_size)
                        hidden_1: hidden state of top-down recurrent unit, (hidden_size)
                        hidden_2: hidden state of language recurrent unit, (hidden_size)

                    Returns:
                        hidden_1: hidden state of top-down recurrent unit, (hidden_size)
                        hidden_2: hidden state of language recurrent unit, (hidden_size)
                        masks: attention masks on proposals, (num_proposals, 1)
                '''

        # fuse inputs for top-down module
        step_input = torch.cat([step_input, hidden_2, target_feat], dim=-1)
        step_input = self.map_topdown(step_input)

        # top-down recurrence
        hidden_1 = self.recurrent_cell_1(step_input, hidden_1)

        # top-down attention
        combined = self.map_feat(obj_feats)  # num_proposals, hidden_size
        combined += self.map_hidd(hidden_1).unsqueeze(0)  # num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.attend(combined)  # num_proposals, 1
        scores.masked_fill_(object_masks == 0, float('-1e30'))

        masks = F.softmax(scores, dim=0)  # num_proposals, 1
        attended = obj_feats * masks
        attended = attended.sum(0)  # feat_size

        # fuse inputs for language module
        lang_input = torch.cat([attended, hidden_1], dim=-1)
        lang_input = self.map_lang(lang_input)

        # language recurrence
        hidden_2 = self.recurrent_cell_2(lang_input, hidden_2)  # hidden_size

        return hidden_1, hidden_2, masks

    def step_mc(self, step_word_idx, hiddens, target_feat, obj_feats, object_masks):

        # split hidden states
        hidden_1, hidden_2 = hiddens[0], hiddens[1]
        batch_size, _ = hidden_1.shape

        # expand

        # embed input word
        step_input = torch.zeros(batch_size, self.emb_size).type_as(hidden_1)  # batch_size, emb_size
        for i in range(batch_size):
            step_input[i] = torch.FloatTensor(
                self.embeddings[self.vocabulary["idx2word"][str(step_word_idx[i].item())]]).cuda()

        # fuse inputs for top-down module
        step_input = torch.cat([step_input, hidden_2, target_feat], dim=-1)
        step_input = self.map_topdown(step_input)

        # top-down recurrence
        hidden_1 = self.recurrent_cell_1(step_input, hidden_1)

        # top-down attention
        combined = self.map_feat(obj_feats)  # batch_size, num_proposals, hidden_size
        combined += self.map_hidd(hidden_1).unsqueeze(1)  # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.attend(combined)  # batch_size, num_proposals, 1
        scores.masked_fill_(object_masks == 0, 0)

        masks = F.softmax(scores, dim=1)  # batch_size, num_proposals, 1
        attended = obj_feats * masks
        attended = attended.sum(1)  # batch_size, feat_size

        # fuse inputs for language module
        lang_input = torch.cat([attended, hidden_1], dim=-1)
        lang_input = self.map_lang(lang_input)

        # language recurrence
        hidden_2 = self.recurrent_cell_2(lang_input, hidden_2)  # num_proposals, hidden_size

        # language classification
        step_output = self.classifier(hidden_2)  # batch_size, num_vocabs
        # logprobs = F.log_softmax(step_output, dim=-1)
        logprobs = step_output.clone()

        # stack hidden states
        hiddens = (hidden_1, hidden_2)

        return step_output, logprobs, hiddens, masks

    def compute_loss(self, data_dict):
        pred_caps = data_dict['lang_cap']  # (B, num_words - 1, num_vocabs)
        num_words = data_dict['lang_len'].max()
        target_caps = data_dict['lang_ids_tensor'][:, 1:num_words]  # (B, num_words - 1)

        _, _, num_vocabs = pred_caps.shape

        # caption loss
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

        # mask out bad boxes
        good_bbox_masks = data_dict["good_clu_masks"].unsqueeze(1).repeat(1, num_words - 1)  # (B, num_words - 1)
        good_bbox_masks = good_bbox_masks.reshape(-1)  # (B * num_words - 1)
        cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)

        num_good_bbox = data_dict["good_clu_masks"].sum()

        if num_good_bbox > 0:  # only apply loss on the good boxes
            pred_caps = pred_caps[data_dict["good_clu_masks"]]  # num_good_bbox
            target_caps = target_caps[data_dict["good_clu_masks"]]  # num_good_bbox

            # caption acc
            pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # num_good_bbox * (num_words - 1)
            target_caps = target_caps.reshape(-1)  # num_good_bbox * (num_words - 1)
            masks = target_caps != 0
            masked_pred_caps = pred_caps[masks]
            masked_target_caps = target_caps[masks]
            cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
        else:  # zero placeholder if there is no good box
            cap_acc = torch.zeros(1)[0].cuda()

        return cap_loss, cap_acc

    def forward(self, data_dict, mode=''):
        if mode == 'train':
            data_dict = self.forward_sample_batch(data_dict)
            data_dict['cap_loss'], data_dict['cap_acc'] = self.compute_loss(data_dict)
        if mode == 'val':
            data_dict = self.forward_scene_val(data_dict)

        return data_dict

    def forward_sample_batch(self, data_dict):
        """
        generate descriptions based on input tokens and object features
        """
        if self.use_relation:
            target_feats = data_dict["enhanced_feats"].cuda()  # batch_size, feat_size
            obj_feats = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size
            valid_masks = data_dict["valid_mask"]  # batch_size, num_proposals
        else:
            target_feats = data_dict["select_feats"].cuda()  # batch_size, feat_size
            obj_feats = data_dict["object_feats"]  # batch_size, num_proposals, feat_size
            valid_masks = data_dict['object_mask']  # batch_size, num_proposals

        # unpack
        word_embs = data_dict["lang_feat"].cuda()  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"].cuda()  # batch_size

        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        hidden_1 = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size, emb_size

        while True:
            # feed
            hidden_1, hidden_2, step_mask = self.step(step_input, target_feats, obj_feats, hidden_1, hidden_2,
                                                      valid_masks.unsqueeze(-1))
            step_output = self.classifier(hidden_2)  # batch_size, num_vocabs

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)
            masks.append(step_mask)  # batch_size, num_proposals, 1

            # next step
            step_id += 1
            if step_id == num_words - 1:
                break  # exit for train mode
            step_input = word_embs[:, step_id]  # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=-1)  # batch_size, num_proposals, num_words - 1/max_len

        # store
        data_dict["lang_cap"] = outputs
        data_dict["topdown_attn"] = masks
        # print(masks)

        return data_dict

    def forward_scene_val(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN):  # batch size must be 1
        if self.use_relation:
            all_target_feats = data_dict["bbox_feature"].cuda()  # 1, num_proposals, feat_size
        else:
            all_target_feats = data_dict["object_feats"].cuda()  # 1, num_proposals, feat_size

        word_embs = data_dict["lang_feat"].cuda()  # 1, emb_size
        num_proposal = all_target_feats.shape[1]
        final_lang = []

        for i in range(num_proposal):
            target_feats = all_target_feats[0][i].cuda()  # feat_size
            if self.use_relation:
                obj_feats = data_dict['rel_bbox_feature'][0][i]  # num_proposals, feat_size
                valid_masks = data_dict['adjacent_mat'][0][i]  # num_proposals
            else:
                obj_feats = data_dict['object_feats'][0]  # num_proposals, feat_size
                valid_masks = data_dict['object_mask'][0]  # num_proposals

            # recurrent from 0 to max_len - 2
            outputs = []
            hidden_1 = torch.zeros(self.hidden_size).cuda()  # hidden_size
            hidden_2 = torch.zeros(self.hidden_size).cuda()  # hidden_size
            step_id = 0
            step_input = word_embs[0][step_id]  # emb_size
            # print(step_input.shape)
            while True:
                # feed
                hidden_1, hidden_2, step_mask = self.step_val(step_input, target_feats, obj_feats, hidden_1, hidden_2,
                                                              valid_masks.unsqueeze(-1))
                step_output = self.classifier(hidden_2)  # batch_size, num_vocabs

                idx = step_output.argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.FloatTensor(self.embeddings[word]).cuda()  # emb_size
                step_preds = emb  # emb_size

                # store
                step_output = step_output.unsqueeze(0)  # 1, num_vocabs
                outputs.append(step_output)

                # next step
                step_id += 1
                if step_id == max_len - 1:
                    break  # exit for no_tf_val mode

                step_input = step_preds  # input_size

            outputs = torch.cat(outputs, dim=0)  # max_len, num_vocabs
            final_lang.append(outputs.unsqueeze(0))  # 1, max_len, num_vocabs

        final_lang = torch.cat(final_lang, dim=0)  # num_proposal, max_len, num_vocabs
        data_dict["final_lang"] = final_lang
        return data_dict

    def forward_sample_greedy(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN):
        if self.use_relation:
            target_feats = data_dict["enhanced_feats"].cuda()  # batch_size, feat_size
            obj_feats = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size
            valid_masks = data_dict["valid_mask"]  # batch_size, num_proposals
        else:
            target_feats = data_dict["select_feats"].cuda()  # batch_size, feat_size
            obj_feats = data_dict["object_feats"]  # batch_size, num_proposals, feat_size
            valid_masks = data_dict['object_mask']  # batch_size, num_proposals

        # unpack
        word_embs = data_dict["lang_feat"].cuda()  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"].cuda()  # batch_size

        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        hidden_1 = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size, emb_size

        while True:
            # feed
            hidden_1, hidden_2, step_mask = self.step(step_input, target_feats, obj_feats, hidden_1, hidden_2,
                                                      valid_masks.unsqueeze(-1))
            step_output = self.classifier(hidden_2)  # batch_size, num_vocabs

            # predicted word
            step_preds = []
            for batch_id in range(batch_size):
                idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda()  # 1, emb_size
                step_preds.append(emb)

            step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)
            masks.append(step_mask)  # batch_size, num_proposals, 1

            # next step
            step_id += 1

            if step_id == max_len - 1:
                break  # exit for no_tf_val mode

            step_input = step_preds  # batch_size, input_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs
        seqLogprobs, _ = F.log_softmax(outputs, dim=2).max(2)
        seq = outputs.argmax(-1)  # batch_size, num_words - 1/max_len 单词下标

        # print(seq)

        return seq, seqLogprobs

    def beam_search(self, init_state, init_logprobs, seq_length, target_feat, obj_feats, masks, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]  # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1),
                                            change.new_ones(batch_size, 1))

                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda

            return logprobs, unaug_logprobs

        # does one step of classical beam search

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobs: probabilities augmented after diversity N*bxV
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions Nxbxl
            # beam_seq_logprobs : log-probability of each decision made, NxbxlxV
            # beam_logprobs_sum : joint log-probability of each beam Nxb

            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # beam_logprobs_sum Nxb logprobs is NxbxV
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size  # Nxb which beam
            selected_ix = ix % vocab_size  # Nxb # which world
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(
                -1)  # N*b which in Nxb beams

            if t > 0:
                # gather according to beam_ix
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
                        beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))

                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1,
                                                                                                                   -1,
                                                                                                                   vocab_size))  # NxbxV
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                # new_state[_ix] = state[_ix][:, state_ix]
                new_state[_ix] = state[_ix][state_ix]

            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)  # This should not affect beam search, but will affect dbs
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        # length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size  # beam per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, len(self.vocabulary["word2idx"])).to(device)
                                   for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        # state_table = list(zip(*[_.reshape(-1, batch_size * bdash, group_size, *_.shape[2:]).chunk(group_size, 2) for _ in init_state]))
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        # logprobs_table = list(init_logprobs.reshape(batch_size * bdash, group_size, -1).chunk(group_size, 0))
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        # END INIT

        # # Chunk elements in the args
        # args = list(args)
        # args = utils.split_tensors(group_size, args) # For each arg, turn (Bbg)x... to (Bb)x(g)x...
        # if self.__class__.__name__ == 'AttEnsemble':
        #     args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name
        # else:
        #     args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= seq_length + divm - 1:
                    # add diversity
                    logprobs = logprobs_table[divm]

                    # suppress previous word
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1).to(device),
                                          float('-inf'))
                    if remove_bad_endings and t - divm > 0:
                        logprobs[torch.from_numpy(np.isin(beam_seq_table[divm][:, :, t - divm - 1].cpu().numpy(),
                                                          self.bad_endings_ix)).reshape(-1), 0] = float('-inf')
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocabulary') and self.vocabulary[
                        str(logprobs.size(1) - 1)] == 'unk':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000
                        # diversity is added here
                    # the function directly modifies the logprobs values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)

                    # infer new beams
                    beam_seq_table[divm], \
                        beam_seq_logprobs_table[divm], \
                        beam_logprobs_sum_table[divm], \
                        state_table[divm] = beam_step(logprobs,
                                                      unaug_logprobs,
                                                      bdash,
                                                      t - divm,
                                                      beam_seq_table[divm],
                                                      beam_seq_logprobs_table[divm],
                                                      beam_logprobs_sum_table[divm],
                                                      state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t - divm] == self.vocabulary["word2idx"]["eos"]
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1
                        if t == seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(),
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                # final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    # move the current group one step forward in time
                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1).to(logprobs.device)

                    inter_ids = torch.arange(batch_size).repeat_interleave(beam_size)
                    _, logprobs_table[divm], state_table[divm], _ = self.step_mc(it, state_table[divm],
                                                                                 target_feat[inter_ids],
                                                                                 obj_feats[inter_ids], masks[inter_ids])
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
                            for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]

        return done_beams

    def forward_sample_mc(self, data_dict, beam_size=1, max_len=CONF.TRAIN.MAX_DES_LEN):
        # unpack
        word_embs = data_dict["lang_feat"].cuda()  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"].cuda()  # batch_size
        batch_size = des_lens.shape[0]

        if self.use_relation:
            target_feats = data_dict["enhanced_feats"].cuda()  # batch_size, feat_size
            obj_feats = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size
            valid_masks = data_dict["valid_mask"]  # batch_size, num_proposals
        else:
            target_feats = data_dict["select_feats"].cuda()  # batch_size, feat_size
            obj_feats = data_dict["object_feats"]  # batch_size, num_proposals, feat_size
            valid_masks = data_dict['object_mask']  # batch_size, num_proposals

        if beam_size > 1:
            # start
            start_word_idx = self.vocabulary["word2idx"]["sos"]
            start_word_idx = torch.Tensor([start_word_idx]).type_as(target_feats).long().repeat(batch_size)

            # init hiddens
            hidden_1 = torch.zeros(batch_size, self.hidden_size).type_as(obj_feats)  # batch_size, hidden_size
            hidden_2 = torch.zeros(batch_size, self.hidden_size).type_as(obj_feats)  # batch_size, hidden_size
            hiddens = (hidden_1, hidden_2)

            valid_masks = valid_masks.unsqueeze(-1)
            _, start_logprobs, hiddens, _ = self.step_mc(start_word_idx, hiddens, target_feats, obj_feats, valid_masks)
            start_logprobs = F.log_softmax(start_logprobs, dim=-1)

            # beam search
            done_beams = self.beam_search(init_state=hiddens, init_logprobs=start_logprobs, seq_length=max_len - 1,
                                          target_feat=target_feats, obj_feats=obj_feats, masks=valid_masks,
                                          opt={"beam_size": beam_size})

            outputs = []  # output captions
            logprobs = []  # output logprobs

            sample_topn = 2
            # aggregate outputs
            for batch_id in range(batch_size):
                batch_outputs = []
                batch_logprobs = []
                for beam_id in range(sample_topn):
                    seq = done_beams[batch_id][beam_id]["seq"]
                    logps = done_beams[batch_id][beam_id]["logps"].gather(1, seq.unsqueeze(1)).squeeze(1)
                    batch_outputs.append(seq)
                    batch_logprobs.append(logps)

                outputs.append(batch_outputs)
                logprobs.append(batch_logprobs)

            seq = torch.zeros(batch_size * sample_topn, max_len - 1).long().cuda()
            seqLogprobs = torch.zeros(batch_size * sample_topn, max_len - 1).cuda()
            cur_idx = 0
            for batch_id in range(batch_size):
                for beam_id in range(sample_topn):
                    cap_len = outputs[batch_id][beam_id].shape[0]
                    seq[cur_idx, :cap_len] = outputs[batch_id][beam_id]
                    seqLogprobs[cur_idx, :cap_len] = logprobs[batch_id][beam_id]
                    cur_idx = cur_idx + 1

            return seq, seqLogprobs

        if beam_size == 1:
            # recurrent from 0 to max_len - 2
            outputs = []
            step_id = 0
            hidden_1 = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
            hidden_2 = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
            step_input = word_embs[:, step_id]  # batch_size, emb_size

            seq = torch.zeros(batch_size, max_len - 1).long().cuda()
            # seqLogprobs = torch.zeros(batch_size, max_len, self.num_vocabs).cuda()
            seqLogprobs = torch.zeros(batch_size, max_len - 1).cuda()
            while True:
                # feed
                hidden_1, hidden_2, step_mask = self.step(step_input, target_feats, obj_feats, hidden_1, hidden_2,
                                                          valid_masks.unsqueeze(-1))
                step_output = self.classifier(hidden_2)  # batch_size, num_vocabs

                logprobs = F.log_softmax(step_output, dim=1)  # batch_size, num_vocabs
                prob_prev = torch.exp(logprobs.data).cpu()  # batch_size, num_vocabs

                # predicted word
                it = torch.multinomial(prob_prev, 1).cuda()  # 0 ~ num_vocabs, 维度为 batch_size,1
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

                step_preds = []
                for batch_id in range(batch_size):
                    idx = it[batch_id]  # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(int(idx.cpu()))]
                    emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda()  # 1, emb_size
                    step_preds.append(emb)

                step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

                # store
                step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
                outputs.append(step_output)

                if step_id == 0:
                    unfinished = it != 3
                    it = it * unfinished.type_as(it)
                    seq[:, step_id] = it
                else:
                    it = it * unfinished.type_as(it)
                    seq[:, step_id] = it
                    unfinished = unfinished & (it != 3)

                seqLogprobs[:, step_id] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

                # next step
                step_id += 1

                if step_id == max_len - 1:
                    break  # exit for no_tf_val mode

                step_input = step_preds  # batch_size, input_size

            return seq, seqLogprobs
