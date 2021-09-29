import torch
def load_bert_data(model,resolved_archive_file):
    print('!!!load Pytorch model!!!')
    state_dict = None
    if state_dict is None:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
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
           'bertembeddings.word_embeddings_layer.weight':'bert.embeddings.word_embeddings.weight',
           'bertembeddings.position_embeddings_layer.weight':'bert.embeddings.position_embeddings.weight',
           'bertembeddings.segment_embeddings_layer.weight':'bert.embeddings.token_type_embeddings.weight',
           'bertembeddings.layer_normalization.weight':'bert.embeddings.LayerNorm.gamma',
           'bertembeddings.layer_normalization.bias':'bert.embeddings.LayerNorm.beta',
           'bertembeddings.layer_normalization.gamma':'bert.embeddings.LayerNorm.gamma',
           'bertembeddings.layer_normalization.beta':'bert.embeddings.LayerNorm.beta',
           'bertembeddings.layer_normalization.weight':'bert.embeddings.LayerNorm.weight',
           'bertembeddings.layer_normalization.bias':'bert.embeddings.LayerNorm.bias',
           'bert_pooler.weight':'bert.pooler.dense.weight',
           'bert_pooler.bias':'bert.pooler.dense.bias',
           'mlm_dense0.weight':'cls.predictions.transform.dense.weight',
           'mlm_dense0.bias':'cls.predictions.transform.dense.bias',
           'mlm_norm.gamma':'cls.predictions.transform.LayerNorm.gamma',
           'mlm_norm.beta':'cls.predictions.transform.LayerNorm.beta',
           #'mlm_norm.weight':'cls.predictions.transform.LayerNorm.gamma',
           #'mlm_norm.bias':'cls.predictions.transform.LayerNorm.beta',
           'mlm_norm.weight':'cls.predictions.transform.LayerNorm.weight',
           'mlm_norm.bias':'cls.predictions.transform.LayerNorm.bias',
           'mlm_dense1.weight':'bert.embeddings.word_embeddings.weight',
           'mlm_dense1.bias':'cls.predictions.bias'
        }
        #由自己的权重名称去找原先的权重名称
        for layer_ndx in range(model.config.num_layers):
            transformer_dicts.update({
                'bert_encoder_layer.%d.attention.query_layer.weight'%(layer_ndx):'bert.encoder.layer.%d.attention.self.query.weight'%(layer_ndx),
                #注意中间有冒号，两边要分开进行赋值
                'bert_encoder_layer.%d.attention.query_layer.bias'%(layer_ndx):'bert.encoder.layer.%d.attention.self.query.bias'%(layer_ndx),
                'bert_encoder_layer.%d.attention.key_layer.weight'%(layer_ndx):'bert.encoder.layer.%d.attention.self.key.weight'%(layer_ndx),
                'bert_encoder_layer.%d.attention.key_layer.bias'%(layer_ndx):'bert.encoder.layer.%d.attention.self.key.bias'%(layer_ndx),
                'bert_encoder_layer.%d.attention.value_layer.weight'%(layer_ndx):'bert.encoder.layer.%d.attention.self.value.weight'%(layer_ndx),
                'bert_encoder_layer.%d.attention.value_layer.bias'%(layer_ndx):'bert.encoder.layer.%d.attention.self.value.bias'%(layer_ndx),
                
                'bert_encoder_layer.%d.dense0.weight'%(layer_ndx):'bert.encoder.layer.%d.attention.output.dense.weight'%(layer_ndx),
                'bert_encoder_layer.%d.dense0.bias'%(layer_ndx):'bert.encoder.layer.%d.attention.output.dense.bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.weight'%(layer_ndx):'bert.encoder.layer.%d.attention.output.LayerNorm.gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.bias'%(layer_ndx):'bert.encoder.layer.%d.attention.output.LayerNorm.beta'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.gamma'%(layer_ndx):'bert.encoder.layer.%d.attention.output.LayerNorm.gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.beta'%(layer_ndx):'bert.encoder.layer.%d.attention.output.LayerNorm.beta'%(layer_ndx),
                 'bert_encoder_layer.%d.layer_norm0.weight'%(layer_ndx):'bert.encoder.layer.%d.attention.output.LayerNorm.weight'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.bias'%(layer_ndx):'bert.encoder.layer.%d.attention.output.LayerNorm.bias'%(layer_ndx),
                
                'bert_encoder_layer.%d.dense.weight'%(layer_ndx):'bert.encoder.layer.%d.intermediate.dense.weight'%(layer_ndx),
                'bert_encoder_layer.%d.dense.bias'%(layer_ndx):'bert.encoder.layer.%d.intermediate.dense.bias'%(layer_ndx),

                'bert_encoder_layer.%d.dense1.weight'%(layer_ndx):'bert.encoder.layer.%d.output.dense.weight'%(layer_ndx),
                'bert_encoder_layer.%d.dense1.bias'%(layer_ndx):'bert.encoder.layer.%d.output.dense.bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.weight'%(layer_ndx):'bert.encoder.layer.%d.output.LayerNorm.gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.bias'%(layer_ndx):'bert.encoder.layer.%d.output.LayerNorm.beta'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.gamma'%(layer_ndx):'bert.encoder.layer.%d.output.LayerNorm.gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.beta'%(layer_ndx):'bert.encoder.layer.%d.output.LayerNorm.beta'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.weight'%(layer_ndx):'bert.encoder.layer.%d.output.LayerNorm.weight'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.bias'%(layer_ndx):'bert.encoder.layer.%d.output.LayerNorm.bias'%(layer_ndx),
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
                if stock_name == 'bert.embeddings.word_embeddings.weight':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.bias':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.weight':
                     stock_value = stock_value.permute(0,1)
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

def _load_state_dict_into_model(model,state_dict,pretrained_model_name_or_path,_fast_init=True):
    new_state_dict = model.state_dict()
    print('new_state_dict = ')
    print(new_state_dict)
    return None,None,None,None