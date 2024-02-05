import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.distributed as dist
from IPython import embed
from transformers import RobertaForSequenceClassification, BertPreTrainedModel, BertModel, BertForSequenceClassification
from sentence_transformers import SentenceTransformer as SBert, util
import sys

def load_model(model_path, **kwargs):
    if 'quantization_config' not in kwargs:
        kwargs['quantization_config'] = None
    if 'device_map' not in kwargs:
        kwargs['device_map'] = None
    if 'trust_remote_code' not in kwargs:
        kwargs['trust_remote_code'] = False
    if 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = None
    if "ance" in model_path:
        model = ANCE.from_pretrained(model_path,
                                     quantization_config=kwargs['quantization_config'],
                                     device_map=kwargs['device_map'],
                                     trust_remote_code=kwargs['trust_remote_code'],
                                     torch_dtype=kwargs['torch_dtype'])
    else:
        model = None
    return model


class ANCE(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    
    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)
    
    def forward_for_pretraining(self, input_ids, attention_mask, wrap_pooler=False):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def compute_augCL_loss(self, query_embs, aug_input_encodings1, aug_input_encodings2, temperature, neg_aug_input_encodings = None):
        cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        input_ids1 = aug_input_encodings1["input_ids"]
        attention_mask1 = aug_input_encodings1["attention_mask"]
        bert_inputs1 = {'input_ids': input_ids1, 'attention_mask': attention_mask1}
        outputs1 = self.forward_for_pretraining(**bert_inputs1)
        sent_rep1 = outputs1

        input_ids2 = aug_input_encodings2["input_ids"]
        attention_mask2 = aug_input_encodings2["attention_mask"]
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2}
        outputs2 = self.forward_for_pretraining(**bert_inputs2)
        sent_rep2 = outputs2
        batch_size = sent_rep1.size(0)

        sent_norm1 = sent_rep1.norm(dim=-1, keepdim=True)  # [batch]
        sent_norm2 = sent_rep2.norm(dim=-1, keepdim=True)  # [batch]
        batch_self_11 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm1) + 1e-6)  # [batch, batch]
        batch_cross_12 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm2) + 1e-6)  # [batch, batch]
        batch_self_11 = batch_self_11 / temperature
        batch_cross_12 = batch_cross_12 / temperature
        batch_first = torch.cat([batch_self_11, batch_cross_12], dim=-1)  # [batch, batch * 2]
        batch_arange = torch.arange(batch_size).to(torch.cuda.current_device())
        mask = F.one_hot(batch_arange, num_classes=batch_size * 2) * -1e10
        batch_first += mask
        batch_label1 = batch_arange + batch_size  # [batch]

        batch_self_22 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm2) + 1e-6)  # [batch, batch]
        batch_cross_21 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm1) + 1e-6)  # [batch, batch]
        batch_self_22 = batch_self_22 / temperature
        batch_cross_21 = batch_cross_21 / temperature
        batch_second = torch.cat([batch_self_22, batch_cross_21], dim=-1)  # [batch, batch * 2]
        batch_second += mask
        batch_label2 = batch_arange + batch_size  # [batch]
        
        if neg_aug_input_encodings is not None:
            neg_input_ids = neg_aug_input_encodings["input_ids"]
            neg_attention_mask = neg_aug_input_encodings["attention_mask"]
            neg_ratio = int(neg_input_ids.shape[0] / input_ids1.shape[0])
            neg_bert_inputs = {'input_ids': neg_input_ids, 'attention_mask': neg_attention_mask}
            neg_sent_rep = self.forward_for_pretraining(**neg_bert_inputs)
            neg_sent_rep = neg_sent_rep.view(batch_size, neg_ratio, -1)
                        
            neg_sent_norm = neg_sent_rep.norm(dim=-1, keepdim=True)
            batch_neg1 = torch.sum(sent_rep1.unsqueeze(1) * neg_sent_rep, dim = -1)
            neg_norm1 = torch.sum(sent_norm1.unsqueeze(1) * neg_sent_norm, dim = -1) + 1e-6
            batch_neg1 = batch_neg1 / neg_norm1
            batch_neg1 = batch_neg1 / (temperature)
            batch_first = torch.cat([batch_first, batch_neg1], dim=-1)  # [batch, batch * 2 + neg_ratio]

            batch_neg2 = torch.sum(sent_rep2.unsqueeze(1) * neg_sent_rep, dim = -1)
            neg_norm2 = torch.sum(sent_norm2.unsqueeze(1) * neg_sent_norm, dim = -1) + 1e-6
            batch_neg2 = batch_neg2 / neg_norm2
            batch_neg2 = batch_neg2 / (temperature)
            batch_second = torch.cat([batch_second, batch_neg2], dim=-1)  # [batch, batch * 2 + neg_ratio]
        

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]
        
        
        contras_loss = cl_loss(batch_predict, batch_label)

        batch_logit = batch_predict.argmax(dim=-1)
        acc = torch.sum(batch_logit == batch_label).float() / (batch_size * 2)

        return contras_loss, acc
