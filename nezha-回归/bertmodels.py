import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
class Config(object):
    def __init__(self,
                initializer_range=0.02,#1
                embedding_size=768,#4
                project_embeddings_with_bias=True,#5
                vocab_size=21128,#6
                hidden_dropout=0.1,#10
                extra_tokens_vocab_size=None,#11
                project_position_embeddings=True,#12
                mask_zero=False,#13
                adapter_size=None,#14
                hidden_act='gelu',#15
                adapter_init_scale=0.001,#16
                num_attention_heads=12,#17
                size_per_head=None,#18
                attention_probs_dropout_prob=0.1,#22
                negative_infinity=-10000.0,#23
                intermediate_size=3072,#24
                intermediate_activation='gelu',#25
                num_layers=12,#26
                #获取对应的切分分割字符内容
                directionality = 'bidi',
                pooler_fc_size = 768,
                pooler_num_attention_heads = 12,
                pooler_num_fc_layers = 3,
                pooler_size_per_head = 128,
                pooler_type = "first_token_transform",
                type_vocab_size = 2,
                with_mlm = False,
                mlm_activation = 'softmax',
                mode = 'bert',
                #max_relative_position = 512,
                solution = 'seq2seq',
                max_relative_position = 64,
                with_pooler = False,
                max_position_embeddings = 512,
                layer_norm_eps = 1e-12,
                *args, **kwargs):
        self.initializer_range = initializer_range#1
        self.embedding_size = embedding_size#4
        self.project_embeddings_with_bias = project_embeddings_with_bias#5
        self.vocab_size = vocab_size#6
        self.token_type_vocab_size = 2#9
        self.hidden_dropout = hidden_dropout#10
        self.extra_tokens_vocab_size = extra_tokens_vocab_size#11
        self.project_position_embeddings = project_position_embeddings#12
        self.mask_zero = mask_zero#13
        self.adapter_size = adapter_size#14
        self.adapter_init_scale = adapter_init_scale#16
        self.num_attention_heads = num_attention_heads#17注意力头数，需指定
        assert embedding_size%num_attention_heads == 0,"size_per_head必须能够整除num_attention_heads"
        self.size_per_head = embedding_size//num_attention_heads#18
        self.attention_probs_dropout_prob = attention_probs_dropout_prob#22
        self.negative_infinity = negative_infinity#23
        self.intermediate_size = intermediate_size#24
        self.intermediate_activation = intermediate_activation#25
        self.num_layers = num_layers#26 attention层数，需指定
        self.directionality = directionality
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.with_mlm = with_mlm
        self.mlm_activation = mlm_activation
        self.mode = mode
        self.solution = solution
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.with_pooler = with_pooler
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

class Bert(nn.Module):
    def __init__(self,config):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        #super(Nezha, self).__init__()
        super(Bert,self).__init__()
        print('---__init__ Nezha')
        self.config = config
        self.bertembeddings = Embeddings(config)
        self.bert_encoder_layer = nn.ModuleList()
        for layer_ndx in range(config.num_layers):
            encoder_layer = Transformer(config)
            self.bert_encoder_layer.append(encoder_layer)
        if config.with_pooler:
            self.bert_pooler = nn.Linear(config.embedding_size,config.embedding_size)
        if config.with_mlm:
            self.mlm_dense0 = nn.Linear(config.embedding_size,config.embedding_size)
            self.mlm_norm = nn.LayerNorm(config.embedding_size,eps=1e-12)
            self.mlm_dense1 = nn.Linear(config.embedding_size,config.vocab_size)
        #print('self.config.initializer_range = ')
        #print(self.config.initializer_range)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            #print('888self.config.initializer_range = 888')
            #print(self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,segment_ids=None,mask_ids=None):
        if segment_ids == None:
            segment_ids = torch.zeros_like(input_ids)
        #print('^^^input_ids = ^^^')
        #print(input_ids)
        outputs = self.bertembeddings(input_ids,segment_ids,mask_ids)
        #outputs = self.bert_encoder_layer[0](outputs)
        for layer_ndx in self.bert_encoder_layer:
            outputs = layer_ndx(outputs)
            #print('outputs1 = ')
            #print(outputs)
        if self.config.with_pooler:
            outputs = self.bert_pooler(outputs)
        #print('outputs2 = ')
        #print(outputs)
        if self.config.with_mlm:
            outputs = self.mlm_dense0(outputs)
            outputs = self.mlm_norm(outputs)
            outputs = self.mlm_dense1(outputs)
        #print('outputs3 = ')
        #print(outputs)
        return outputs

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, embedding_size, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_size))
        self.beta  = nn.Parameter(torch.zeros(embedding_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        #keep_dim=True:保持输出的维度,keepdim=False:输出在求范数的维度上
        #元素个数变为1
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    def __init__(self,config):
        #之前__init__之中少写了一个self,报错multiple initialize
        #先能跑通一个网络层，再往里面加入网络层
        super(Embeddings, self).__init__()
        self.word_embeddings_layer = nn.Embedding(config.vocab_size,config.embedding_size)
        self.segment_embeddings_layer = nn.Embedding(config.token_type_vocab_size,config.embedding_size)
        self.position_embeddings_layer = nn.Embedding(config.max_position_embeddings,config.embedding_size)
        self.layer_normalization = nn.LayerNorm(config.embedding_size,eps=1e-12)
        self.dropout_layer = nn.Dropout(config.hidden_dropout)

    def forward(self,input_ids,segment_ids,mask_ids=None):
        if segment_ids == None:
            segment_ids = torch.zeros_like(input_ids)
        seq_len = input_ids.size(1)
        #长度为(batch_size,seq_len)
        position_ids = torch.arange(seq_len,dtype=torch.long,device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        results = self.word_embeddings_layer(input_ids)+self.segment_embeddings_layer(segment_ids)+self.position_embeddings_layer(position_ids)
        results = self.layer_normalization(results)
        results = self.dropout_layer(results)
        return results
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_activation(activation):
    if activation == 'gelu':
        return F.gelu
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return torch.tanh

class Transformer(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(Transformer,self).__init__()
        self.attention = AttentionLayer(config)
        self.dense0 = nn.Linear(config.embedding_size,config.embedding_size)
        self.dropout0 = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm0 = nn.LayerNorm(config.embedding_size,eps=1e-12)
        self.dense = nn.Linear(config.embedding_size,config.intermediate_size)
        self.activation = get_activation(config.hidden_act)
        self.dense1 = nn.Linear(config.intermediate_size,config.embedding_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm1 = nn.LayerNorm(config.embedding_size,eps=1e-12)
        
    
    def forward(self,inputs,masks=None,**kwargs):
        residual = inputs
        embedding_output = inputs
        embedding_output = self.attention(inputs)
        embedding_output = self.dense0(embedding_output)
        embedding_output = self.dropout0(embedding_output)
        
        embedding_output = self.layer_norm0(residual+embedding_output)
        residual = embedding_output
        embedding_output = self.dense(embedding_output)
        embedding_output = self.activation(embedding_output)
        embedding_output = self.dense1(embedding_output)
        embedding_output = self.dropout1(embedding_output)
        embedding_output = self.layer_norm1(residual+embedding_output)
        
        return embedding_output

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    print('s = ')
    print(s)
    assert n_dims > 1 and n_dims < len(s)
    print('s[:-n_dims] = ')
    print(*s[:-n_dims])
    return x.view(*s[:-n_dims], -1)

class AttentionLayer(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(AttentionLayer,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.embedding_size)
        self.key_layer = nn.Linear(config.embedding_size,config.embedding_size)
        self.value_layer = nn.Linear(config.embedding_size,config.embedding_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.config = config
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,inputs,mask=None,**kwargs):
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        batch_size,seq_len,embedding_size = inputs.size(0),inputs.size(1),inputs.size(2)
        query = query.view([batch_size,seq_len,self.config.num_attention_heads,self.config.size_per_head])
        #query = (1,5,12,64)
        query = query.permute(0,2,1,3)
        key = key.view([batch_size,seq_len,self.config.num_attention_heads,self.config.size_per_head])
        key = key.permute(0,2,1,3)
        attention_scores = torch.matmul(query,key.transpose(-1,-2))
        #attention_scores = [1,12,5,64]*[1,12,64,5] = [1,12,5,5]
        attention_scores = attention_scores/math.sqrt(float(self.config.size_per_head))
        attention_scores = F.softmax(attention_scores,dim=-1)
        value = value.view(batch_size,seq_len,self.config.num_attention_heads,self.config.size_per_head)
        value = value.permute(0,2,1,3)
        context_layer = torch.matmul(attention_scores,value)
        context_layer = context_layer.permute(0,2,1,3)
        context_layer = context_layer.contiguous().view(batch_size,seq_len,self.config.num_attention_heads*self.config.size_per_head)
        return context_layer