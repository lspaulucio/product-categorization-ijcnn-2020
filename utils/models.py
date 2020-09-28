# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel

class Ensemble(nn.Module):
    def __init__(self, folder, hidden_size=768, num_classes=1588, num_models=10):
        super(Ensemble, self).__init__()

        self.fcs = [nn.Linear(hidden_size, num_classes) for _ in range(num_models)]
        for i in range(num_models):
            checkpoint = torch.load(folder+'fc_{}.pt'.format(i))
            self.fcs[i].load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        output = self.fcs[0](x)
        for i in range(1, len(self.fcs)):
            output += self.fcs[i](x)
        return output
    
    def cuda(self):
        for fc in self.fcs:
            fc.cuda()

class BertSequenceEmbedding(nn.Module):
    def __init__(self, normalize=False, dropout=0.1, path=None):
        super(BertSequenceEmbedding, self).__init__()
        self.normalize = normalize
        
        if path is None:
            model = 'bert-base-multilingual-cased'
        else:
            model = path

        self.bert = BertModel.from_pretrained(model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        if self.normalize:
            pooled_output = nn.functional.normalize(pooled_output)
        dropout_output = self.dropout(pooled_output)

        return dropout_output