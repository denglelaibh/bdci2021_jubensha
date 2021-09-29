import torch
def load_bert_data(model,resolved_archive_file):
    print('555load Pytorch model555')
    state_dict = None
    if state_dict is None:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
            #print('state_dict = ')
            #print(state_dict.keys())
            #print('model state_dict = ')
            #print(model.state_dict())
            file_name = list(state_dict.keys())
            model_dict = model.state_dict()
        except Exception:
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                f"at '{resolved_archive_file}'"
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )
        transformer_dicts = {
           'bertembeddings.word_embeddings_layer.weight':'module.bertembeddings.word_embeddings_layer.weight',
           'bertembeddings.position_embeddings_layer.weight':'module.bertembeddings.position_embeddings_layer.weight',
           'bertembeddings.segment_embeddings_layer.weight':'module.bertembeddings.segment_embeddings_layer.weight',
           'bertembeddings.layer_normalization.weight':'module.bertembeddings.layer_normalization.weight',
           'bertembeddings.layer_normalization.bias':'module.bertembeddings.layer_normalization.bias',
           'bertembeddings.layer_normalization.gamma':'module.bertembeddings.layer_normalization.gamma',
           'bertembeddings.layer_normalization.beta':'module.bertembeddings.layer_normalization.beta',
           'bert_pooler.weight':'module.bert_pooler.weight',
           'bert_pooler.bias':'module.bert_pooler.bias',
           'mlm_dense0.weight':'module.mlm_dense0.weight',
           'mlm_dense0.bias':'module.mlm_dense0.bias',
           'mlm_norm.weight':'module.mlm_norm.weight',
           'mlm_norm.bias':'module.mlm_norm.bias',
           'mlm_dense1.weight':'module.mlm_dense1.weight',
           'mlm_dense1.bias':'module.mlm_dense1.bias'
        }
        #由自己的权重名称去找原先的权重名称
        for layer_ndx in range(model.config.num_layers):
            transformer_dicts.update({
                'bert_encoder_layer.%d.attention.query_layer.weight'%(layer_ndx):'module.bert_encoder_layer.%d.attention.query_layer.weight'%(layer_ndx),
                #注意中间有冒号，两边要分开进行赋值
                'bert_encoder_layer.%d.attention.query_layer.bias'%(layer_ndx):'module.bert_encoder_layer.%d.attention.query_layer.bias'%(layer_ndx),
                'bert_encoder_layer.%d.attention.key_layer.weight'%(layer_ndx):'module.bert_encoder_layer.%d.attention.key_layer.weight'%(layer_ndx),
                'bert_encoder_layer.%d.attention.key_layer.bias'%(layer_ndx):'module.bert_encoder_layer.%d.attention.key_layer.bias'%(layer_ndx),
                'bert_encoder_layer.%d.attention.value_layer.weight'%(layer_ndx):'module.bert_encoder_layer.%d.attention.value_layer.weight'%(layer_ndx),
                'bert_encoder_layer.%d.attention.value_layer.bias'%(layer_ndx):'module.bert_encoder_layer.%d.attention.value_layer.bias'%(layer_ndx),
                
                'bert_encoder_layer.%d.dense0.weight'%(layer_ndx):'module.bert_encoder_layer.%d.dense0.weight'%(layer_ndx),
                'bert_encoder_layer.%d.dense0.bias'%(layer_ndx):'module.bert_encoder_layer.%d.dense0.bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.weight'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm0.weight'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.bias'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm0.bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.gamma'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm0.gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.beta'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm0.beta'%(layer_ndx),

                
                'bert_encoder_layer.%d.dense.weight'%(layer_ndx):'module.bert_encoder_layer.%d.dense.weight'%(layer_ndx),
                'bert_encoder_layer.%d.dense.bias'%(layer_ndx):'module.bert_encoder_layer.%d.dense.bias'%(layer_ndx),

                'bert_encoder_layer.%d.dense1.weight'%(layer_ndx):'module.bert_encoder_layer.%d.dense1.weight'%(layer_ndx),
                'bert_encoder_layer.%d.dense1.bias'%(layer_ndx):'module.bert_encoder_layer.%d.dense1.bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.gamma'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm1.gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.beta'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm1.beta'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.weight'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm1.weight'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.bias'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm1.bias'%(layer_ndx),
            })
        model_name = model.state_dict().keys()
        weight_value_tuples = []
        skipped_weight_value_tuples = []
        skip_count = 0
        loaded_weights = []
        used_name = []
        for param_name in model_name:
            stock_name = transformer_dicts[param_name]
            if stock_name in file_name:
                stock_value = state_dict[stock_name]
                param_value = model_dict[param_name]
                if stock_name == 'module.bertembeddings.word_embeddings_layer.weight':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.bias':
                    stock_value = stock_value[:param_value.shape[0]]
                #if param_name == 'mlm_dense1.weight':
                #     stock_value = stock_value.permute(0,1)
                if param_value.shape != stock_value.shape:
                    print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                          "with the checkpoint:[{}] shape:{}".format(param_name, param_value.shape,
                                                                 stock_name, stock_value.shape))
                    skipped_weight_value_tuples.append((param_name,stock_value))
                    continue
                used_name.append(stock_name)
                model_dict[param_name] = stock_value
                weight_value_tuples.append((param_value,stock_value))
            else:
                print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param_name, stock_name, resolved_archive_file))
                skip_count += 1

    model.load_state_dict(model_dict)
    print("Done loading {} NEZHA weights from: {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), resolved_archive_file,skip_count, len(skipped_weight_value_tuples)))

    #print("Unused weights from checkpoint:",
    #      "\n\t" + "\n\t".join(sorted(file_name.difference(used_name))))
    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(set(file_name).difference(set(used_name))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return model