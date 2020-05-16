from .BasicModule import BasicModule
from pytorch_pretrained_bert import BertForTokenClassification, BertModel, BertConfig
import torch.nn as nn
from metrics import get_entities
import json
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from modules import Encoder
from IPython import embed
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class BERT_REL(BasicModule):
    def __init__(self, opt):
        super(BERT_REL, self).__init__()
        self.opt = opt
        self.bertForToken = BertForTokenClassification.from_pretrained(self.opt.bert_model_dir, num_labels=self.opt.tag_nums)
        # tag分类
        self.num_labels = self.opt.tag_nums

        # 关系分类
        self.rel_bert = BertModel.from_pretrained(self.opt.bert_model_dir)
        self.rel_fc = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, self.opt.rel_nums))

        self.id2tag = json.loads(open(opt.id2tag_dir, 'r').readline())
        self.type2types = json.loads(open(opt.type2types_dir, 'r').readline())
        self.sep1 = torch.LongTensor([1]).to("cuda")
        self.sep2 = torch.LongTensor([2]).to("cuda")
        self.init_weights()
    def init_weights(self):
        for name, param in self.rel_fc.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def match_entities(self, tags_lists):
        """
        返回Batch个句子中所有可能存在关系的实体对
        [[s1, e1, s2, e2, r],
         [s1, e1, s2, e2, 0]
         ...]]
        """
        tags_lists = torch.max(tags_lists,2)[1]
        if self.opt.use_gpu:
            tags_lists = tags_lists.cpu()
        tags_lists = tags_lists.tolist()
        all_entitys = []
        for tags_list in tags_lists:
            all_entity = []
            tags_list = [self.id2tag[str(i)] for i in tags_list]
            ent_and_position = get_entities(tags_list)
            for ent1 in ent_and_position:
                for ent2 in ent_and_position:
                    if ent2 == ent1:
                        continue
                    ent2_for_ent1 = self.type2types.get(ent1[0],[])
                    if ent2[0]not in ent2_for_ent1:
                        continue
                    all_entity.append([ent1[1], ent1[2], ent2[1], ent2[2], 0])
            all_entitys.append(all_entity)
        return all_entitys


    def get_ent_pair_matrix(self, positions, sen_matrix):
        '''
        position: [s1, e1, s2, e2]
        sent_matrix: 对应句子的bert输出
        返回，该pair的cnn输入句子
        '''
        s1, e1, s2, e2 = positions
        type_obj = torch.zeros(e1-s1+1).long()
        type_sbj = torch.ones(e2-s2+1).long()
        if e1 < s2:
            m_s = e1+1
            m_e = s2-1
        else:
            m_s = e2+1
            m_e = s1-1
        type_mid = torch.ones(m_e-m_s+1).long() * 2
        if self.opt.use_gpu:
            type_obj = type_obj.cuda()
            type_sbj = type_sbj.cuda()
            type_mid = type_mid.cuda()
        obj_vecs = sen_matrix[s1:e1+1,:] + self.type_emb(type_obj)
        sbj_vecs = sen_matrix[s2:e2+1,:] + self.type_emb(type_sbj)
        mid_vecs = sen_matrix[m_s:m_e+1,:] + self.type_emb(type_mid)

        sample_matrix = torch.cat([obj_vecs, sbj_vecs, mid_vecs], 0)
        if sample_matrix.size(0) < self.opt.tuple_max_len:
            length = sample_matrix.size(0)
            pad = torch.zeros(self.opt.tuple_max_len - sample_matrix.size(0), self.opt.bert_hidden_size)
            if self.opt.use_gpu:
                pad = pad.cuda()
            sample_matrix = torch.cat([sample_matrix, pad], 0)
        else:
            length = self.opt.tuple_max_len
            sample_matrix = sample_matrix[:self.opt.tuple_max_len,:]
        return sample_matrix, length

    def forward(self, batch_data, tags=None, entRels=None):
        batch_masks = batch_data.gt(0)   # 用于长度的mask
        '''
        B: 批大小, L: 句子最大, N: Toekn的类别数
        input
            - batch_data: torch.LongTensor (B,L) 输入数据
            - token_type_ids: torch.LongTensor 两句话时才有， 标记是那一句话的词
            - attention_mask: torch.LongTensor (B,L) 用来对长度的mask
            - tags: torch.LongTensor (B)数据标签
            - entRels: [[[s1,e1,s2,e2,r], []]; [[],[]]]
        output
            train : tags的损失和关系的损失
            predict: [[[s1, e1, s2, e2, r], []],[[],[]]]
        '''
        # 训练
        sequence_output, _ = self.bertForToken.bert(batch_data, token_type_ids=None, attention_mask=batch_masks, output_all_encoded_layers=False)
        sequence_output = self.bertForToken.dropout(sequence_output)  # (B, L, H)
        logits = self.bertForToken.classifier(sequence_output)

        all_rels, all_seqs = [], []
        if tags is None:
            entRels = self.match_entities(logits)
        for idx, sen_ent_rels in enumerate(entRels):
            sen_matrix = sequence_output[idx,:,:]
            for sample in sen_ent_rels:
                # sample [s1, e1, s2, e2, r]
                s1, e1, s2, e2, r = sample
                obj_span = batch_data[idx][s1:e1+1]
                sub_span = batch_data[idx][s2:e2+1]
                obj_sub_doc = torch.cat([obj_span, self.sep1, sub_span, self.sep2, batch_data[idx]])
                all_seqs.append(obj_sub_doc)
                if tags is not None:
                    all_rels.append(r)

        all_seqs = pad_sequence(all_seqs).permute(1,0)
        if all_seqs.size(0) > 12 and not(tags is None):
            reserved = np.random.choice(range(all_seqs.size(0)), 12, replace=True)
            all_seqs = all_seqs[reserved]
            all_rels = np.array(all_rels)[reserved].tolist()
        #embed()
        seq_masks = all_seqs.gt(0)
        rel_feats, _ = self.rel_bert(all_seqs, attention_mask=seq_masks)
        out = self.rel_fc(rel_feats[-1][:,0,:])

        if tags is not None:
            loss_fct = CrossEntropyLoss()
            loss_tags = loss_fct(logits.view(-1, self.num_labels), tags.view(-1))
            all_rels = torch.LongTensor(all_rels)
            if self.opt.use_gpu:
                all_rels = all_rels.cuda()
            loss_rels = loss_fct(out, all_rels)
            return loss_tags, loss_rels

        all_out = []
        idx = 0
        sum_case = sum([len(i) for i in entRels])
        out = torch.max(out, 1)[1]
        assert sum_case == len(out)
        for entRel in entRels:
            sen_pair = []
            for t, e in enumerate(entRel):
                entPostion = entRel[t]
                entPostion[-1] = out[idx].item()
                idx+=1
                if entPostion[-1] == self.opt.rel_nums-1:
                    continue
                sen_pair.append(entPostion)
            all_out.append(sen_pair)
        return logits, all_out
