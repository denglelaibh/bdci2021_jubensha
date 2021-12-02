import pandas as pd
train=pd.read_csv('/home/xiaoguzai/程序/剧本角色情感识别/数据/train_dataset_v2.tsv', sep='\t')
seed = 1
split_n = 10
#多模折数

import pandas as pd
import itertools
import json
import math
from pythonicforbert import get_model_function,FullTokenizer
vocab_file = r'/home/xiaoguzai/模型/nezha-base/vocab.txt'
vocab_size = len(open(vocab_file,'r').readlines()) 
with open('/home/xiaoguzai/模型/nezha-base/config.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
json_data['vocab_size'] = vocab_size

content_dict = {}
play_dict = {}
for index in range(len(train['content'])):
    currentindex = train['id'][index]
    currentindex = currentindex.split('_')
    str1 = currentindex[0]
    while len(str1) < 5:
        str1 = '0'+str1
    str4 = currentindex[3]
    while len(str4) < 4:
        str4 = '0'+str4
    resultindex = str1+'_'+currentindex[1]+'_'+currentindex[2]+'_'+str4
    train['id'][index] = currentindex
    if str1 not in play_dict:
        play_dict[str1] = 0
    else:
        play_dict[str1] = play_dict[str1]+1
    content_dict[resultindex] = train['content'][index]
    
    
def sortedDictValues(adict): 
    keys = adict.keys() 
    keys = sorted(keys)
    new_content_dict = {}
    for data in keys:
        new_content_dict[data] = content_dict[data]
    return new_content_dict 
new_content_dict = sortedDictValues(content_dict)

inv_new_content_dict = {}
for data in new_content_dict:
    inv_new_content_dict[new_content_dict[data]] = data

keys = new_content_dict.keys()
keys = list(keys)
dict_keys_index = {}
for index in range(len(keys)):
    #dict_keys_index[index] = keys[index]
    dict_keys_index[keys[index]] = index

content,emotions = [],[]
dicts = {}
inv_dicts = {}
label_num = 0
characters = []
ids = []
label1,label2,label3,label4,label5,label6 = [],[],[],[],[],[]
for index in range(len(train['content'])):
    if pd.isna(train['emotions'][index]) == False:
        ids.append(train['id'][index])
        content.append(train['content'][index])
        emotions.append(train['emotions'][index])
        characters.append(train['character'][index])
        current_emotion = train['emotions'][index].split(',')
        label1.append(int(current_emotion[0]))
        label2.append(int(current_emotion[1]))
        label3.append(int(current_emotion[2]))
        label4.append(int(current_emotion[3]))
        label5.append(int(current_emotion[4]))
        label6.append(int(current_emotion[5]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
tokenizer = FullTokenizer(vocab_file=vocab_file)
from tqdm import tqdm
import os
import random
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(seed)

play_name = []
for data in play_dict.keys():
    if data != '34945':
        play_name.append(data)
        
class ClassificationDataset(Dataset):
    def __init__(self,text,characters,label1,label2,label3,label4,label5,label6,maxlen,flag):
        self.text = text
        self.maxlen = maxlen
        self.characters = characters
        token_data,token_id,segment_id,mask_id = [],[],[],[]
        #sequence填充可以最后统一实现
        for index in tqdm(range(len(self.text))):
            current_text = text[index]
            current_character = characters[index]
            current_str_index = inv_new_content_dict[current_text]
            current_str_index_bak = inv_new_content_dict[current_text]
            current_key_index = dict_keys_index[current_str_index]
            current_key_index_bak = dict_keys_index[current_str_index]
            # 同一幕中所有上文拼在一起
            lcs_lst = []

            pre_current_key_index = current_key_index
            pre_current_key_text = current_text
            pre_content = ''
            #前面一个语句对应的索引以及文本内容
            num = 0
            while True:
                new_current_key_index = pre_current_key_index
                new_current_key_text = pre_current_key_text
                while new_current_key_index != -1 and new_content_dict[keys[new_current_key_index]] == new_current_key_text:
                    new_current_key_index = new_current_key_index-1
                if new_current_key_index == -1:
                    pre_content = ''
                    break
                # 只在同一幕中找
                new_str_index = keys[new_current_key_index]
                current_str_indexs = current_str_index.split('_')
                new_str_indexs = new_str_index.split('_')
                # 跨剧本，停止寻找
                if current_str_indexs[0] != new_str_indexs[0]:
                    break
                # 跨幕，停止寻找
                if current_str_indexs[1] != new_str_indexs[1]:
                    break
                new_pre_content = new_content_dict[keys[new_current_key_index]]
                if str(new_pre_content) == 'nan':
                    new_pre_content = '无'
                if str(current_character) == 'nan':
                    break

                lcs_lst.append(new_pre_content)
                num = num+1
                
                pre_current_key_index = new_current_key_index
                pre_current_key_text = new_pre_content
                if num == 5:
                    break
            lcs_lst.reverse()
            if not lcs_lst:
                pre_content = '无'
            else:
                pre_content = ''.join(lcs_lst)
            # 统计有效上文长度
            
            if str(current_character) == 'nan':
                current_character = '无'
                
            current_character_token = tokenizer.tokenize(current_character)
            pre_token = tokenizer.tokenize(pre_content)
            current_token = tokenizer.tokenize(current_text)

            current_token = ["[CLS]"]+current_character_token+["[SEP]"]+pre_token+["[SEP]"]+current_token+["[SEP]"]
            current_id = tokenizer.convert_tokens_to_ids(current_token)
            current_id = self.sequence_padding(current_id)
            token_data.append(current_token)
            token_id.append(current_id)
        self.token_data = token_data
        self.token_id = token_id
        
        self.label = [label1,label2,label3,label4,label5,label6]
        self.tensors = [torch.tensor(self.token_id,dtype=torch.long),
                 torch.tensor(self.label[0],dtype=torch.long),
                 torch.tensor(self.label[1],dtype=torch.long),
                 torch.tensor(self.label[2],dtype=torch.long),
                 torch.tensor(self.label[3],dtype=torch.long),
                 torch.tensor(self.label[4],dtype=torch.long),
                 torch.tensor(self.label[5],dtype=torch.long)]
        
    def __len__(self):
        return len(self.token_id)
    
    def __getitem__(self,index):
        return tuple(tensor[index] for tensor in self.tensors)

    def sequence_padding(self,inputs,padding = 0):
        length = self.maxlen
        if len(inputs) > length:
            abondon = len(inputs)-length
            inputs = inputs[0:2]+inputs[3+abondon-1:]
        outputs = []
        pad_width = (0,length-len(inputs))
        x = np.pad(inputs,pad_width,'constant',constant_values=padding)
        return x

text = content
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np

class ClassificationModel(nn.Module):
    def __init__(self,model,config,n_labels):
        super(ClassificationModel,self).__init__()
        #self.embedding = nn.Embedding(30522,768)
        self.model = model
        self.fc = nn.Linear(config.embedding_size,n_labels)
        
    def forward(self,input_ids):
        #outputs = self.embedding(input_ids)
        mask_ids = torch.not_equal(input_ids,0)
        output = self.model(input_ids)
        #[64,128,768]
        if mask_ids is not None:
            mask_ids = mask_ids[:,:,None].float()
            output -= 1e-12*(1.0-mask_ids)
        output = output[:,0]
        output = self.fc(output)
        return output
    #之前这里少量return outputs返回值为None

def compute_multilabel_loss(x,model,label):
    logit = model(x)
    mseloss = 0
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    logit = torch.transpose(logit, 0, 1)
    mseloss = loss_fn(logit,label)
    return mseloss

from tqdm import tqdm
from colorama import Fore
n_label = 6
bestpoint = []
for currentsplit in range(split_n):
    bestpoint.append(0.0)
needcount = 30//split_n
#random.seed(seed)
current_list = []
for index in range(30):
    current_list.append(index)
test_index = random.sample(current_list,30)
for current_split in range(split_n):
    train_text,test_text,train_ids,test_ids = [],[],[],[]
    train_characters,test_characters = [],[]
    train_label1,train_label2,train_label3,train_label4,train_label5,train_label6 = [],[],[],[],[],[]
    test_label1,test_label2,test_label3,test_label4,test_label5,test_label6 = [],[],[],[],[],[]
    needcount = 30//split_n
    
    test_str = []
    for index in test_index[needcount*current_split:min(needcount*(current_split+1),30)]:
        test_str.append(play_name[index])
        if play_name[index] == '02388':
            test_str.append('34945')
    for index in range(len(ids)):
        current_id = ids[index]
        current_str = ''
        for data1 in current_id:
            current_str = current_str+data1
        if  current_id[0] in test_str:
            test_ids.append(current_str)
            test_text.append(text[index])
            test_characters.append(characters[index])
            test_label1.append(label1[index])
            test_label2.append(label2[index])
            test_label3.append(label3[index])
            test_label4.append(label4[index])
            test_label5.append(label5[index])
            test_label6.append(label6[index])
        else:
            train_ids.append(current_str)
            train_text.append(text[index])
            train_characters.append(characters[index])
            train_label1.append(label1[index])
            train_label2.append(label2[index])
            train_label3.append(label3[index])
            train_label4.append(label4[index])
            train_label5.append(label5[index])
            train_label6.append(label6[index])

    train_dataset = ClassificationDataset(train_text,train_characters,train_label1,train_label2,train_label3,train_label4,train_label5,train_label6,maxlen=250,flag=True)
    test_dataset = ClassificationDataset(test_text,test_characters,test_label1,test_label2,test_label3,test_label4,test_label5,test_label6,maxlen=250,flag=False)
    #到里面的classificationdataset才进行字符的切割以及划分
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=16)

    
    #加載官方预训练模型
    nezha,nezhaconfig,get_data = get_model_function('nezha-base')
    config = nezhaconfig(**json_data)
    config.with_mlm = False
    model = ClassificationModel(nezha,config,n_label)
    nezha = nezha(config)
    nezha = get_data(nezha,'/home/xiaoguzai/模型/nezha-base/pytorch_model.bin')
    model = ClassificationModel(nezha,config,n_label)
    #初始化模型:易漏
    
    #加載我方预训练模型
    #nezha,nezhaconfig,get_data = get_model_function('nezha-pretrain')
    #config = nezhaconfig(**json_data)
    #config.with_mlm = False
    #model = ClassificationModel(nezha,config,n_label)
    #nezha = nezha(config)
    #nezha = get_data(nezha,'/home/xiaoguzai/程序/剧本角色情感识别/预训练文件/labeled_data+model_epoch=60.pth')
    #model = ClassificationModel(nezha,config,n_label)
    
    
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import LambdaLR
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(),lr=9e-6)
    def lr_lambda(epoch):
        if epoch > 5:
            return 1
        else:
            return 2/(epoch+1)
    scheduler = LambdaLR(optimizer, lr_lambda)
    print("初始化的学习率：", optimizer.defaults['lr'])
    for epoch in range(3):
        print('epoch {}'.format(epoch))
        train_loss = 0
        train_acc = 0

        model.train()

        model = model.to(device)
        model = nn.DataParallel(model)
        loss_fn = torch.nn.MSELoss(reduce=True,size_average=True)

        for batch_token_ids,batch_label1,batch_label2,batch_label3,batch_label4,batch_label5,batch_label6 in tqdm(train_loader):
            torch.set_printoptions(edgeitems=768)# 设置输出矩阵维度为768
            #print('batch_token_ids')
            #print(batch_token_ids)
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = [batch_label1.numpy(),batch_label2.numpy(),batch_label3.numpy(),\
                            batch_label4.numpy(),batch_label5.numpy(),batch_label6.numpy()]
            batch_labels = torch.tensor(batch_labels,dtype=torch.float)
            batch_labels = batch_labels.to(device)
            #for index in range(len(batch_labels)):
            #    batch_labels[index] = batch_labels[index].to(device)
            optimizer.zero_grad()
            loss = compute_multilabel_loss(batch_token_ids,model,batch_labels)
            train_loss = train_loss+loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        #注意必须从epoch=1开始，否则第0个没有学习率
        print('Train Loss: {:.6f}'.format(train_loss))

        model = model.to(device)
        model.eval()

        eval_true_label = [[],[],[],[],[],[]]
        eval_predict_label = [[],[],[],[],[],[]]

        eval_label_loss = [[0,0,0,0],\
                           [0,0,0,0],\
                           [0,0,0,0],\
                           [0,0,0,0]]
        #for batch_token_ids,batch_labels in tqdm(test_loader,bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
        for batch_token_ids,batch_label1,batch_label2,batch_label3,batch_label4,batch_label5,batch_label6 in tqdm(test_loader):
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = [batch_label1,batch_label2,batch_label3,batch_label4,batch_label5,batch_label6]
            with torch.no_grad():
                output = model(batch_token_ids)
            for index in range(len(output)):
                #总的数据
                for index1 in range(len(output[index])):
                    #对应的类别概率0~6
                    abs0 = abs(output[index][index1]-0)
                    abs1 = abs(output[index][index1]-1)
                    abs2 = abs(output[index][index1]-2)
                    abs3 = abs(output[index][index1]-3)
                    currentdata = [abs0,abs1,abs2,abs3]
                    current_label = currentdata.index(min(currentdata))
                    #eval_predict_label[index1].append(current_label)
                    current_predict = output[index][index1].item()
                    if current_predict < 0.00:
                        current_predict = 0
                    elif current_predict > 3:
                        current_predict = 3
                    #当前类别的分类结果,这里append(output[index][index1])
                    #直接放入对应概率值时效果最好,这里直接放入0,1,2,3对应的数值
                    eval_predict_label[index1].append(current_predict)
            for index in range(len(batch_labels)):
                current_label = np.array(batch_labels[index].cpu()).tolist()
                eval_true_label[index].extend(current_label)
        criterion = nn.MSELoss()
        totalloss = 0

        for index in range(len(eval_true_label)):
            inputs = torch.autograd.Variable(torch.from_numpy(np.array(eval_predict_label[index])))
            target = torch.autograd.Variable(torch.from_numpy(np.array(eval_true_label[index])))
            for index1 in range(len(inputs)):
                abs0 = abs(inputs[index1]-0)
                abs1 = abs(inputs[index1]-1)
                abs2 = abs(inputs[index1]-2)
                abs3 = abs(inputs[index1]-3)
                currentdata = [abs0,abs1,abs2,abs3]
                current_label = currentdata.index(min(currentdata))
                true_label = target[index1].item()
                if current_label != true_label:
                    eval_label_loss[true_label][current_label] = eval_label_loss[true_label][current_label]+1
                    #对的预测为错误的
            current_loss = criterion(inputs.float(),target.float())
            current_loss = current_loss.item()
            print('index = %d,current_loss = %f'%(index,current_loss))
            totalloss = totalloss+current_loss

        #totalloss = totalloss/len(eval_predict_label)
        print('totalloss = ')
        print(totalloss)
        totalloss = totalloss/6
        totalloss = math.sqrt(totalloss)
        currentpoint = 1/(1+totalloss)
        #currentpoint = 1/(1+current_loss)
        print('current_point = ')
        print(currentpoint)
        if currentpoint >= bestpoint[current_split]:
            bestpoint[current_split] = currentpoint
            torch.save(model,'best_score='+str(bestpoint[current_split])+'split='+str(current_split)+'.pth')
        print('eval_label_loss = ')
        print(eval_label_loss)

test=pd.read_csv('/home/xiaoguzai/程序/剧本角色情感识别/数据/test_dataset.tsv', sep='\t')

content_dict = {}
for index in range(len(test['content'])):
    currentindex = test['id'][index]
    currentindex = currentindex.split('_')
    str1 = currentindex[0]
    while len(str1) < 5:
        str1 = '0'+str1
    str4 = currentindex[3]
    while len(str4) < 4:
        str4 = '0'+str4
    resultindex = str1+'_'+currentindex[1]+'_'+currentindex[2]+'_'+str4
    content_dict[resultindex] = test['content'][index]
	
def sortedDictValues(adict): 
    keys = adict.keys() 
    keys = sorted(keys)
    print('keys = ')
    print(keys[0:10])
    new_content_dict = {}
    for data in keys:
        new_content_dict[data] = content_dict[data]
    return new_content_dict 
new_content_dict = sortedDictValues(content_dict)

inv_new_content_dict = {}
for data in new_content_dict:
    inv_new_content_dict[new_content_dict[data]] = data
	
keys = new_content_dict.keys()
keys = list(keys)
dict_keys_index = {}
for index in range(len(keys)):
    #dict_keys_index[index] = keys[index]
    dict_keys_index[keys[index]] = index
	
content,emotions = [],[]
dicts = {}
inv_dicts = {}
label_num = 0
characters = []
for index in range(len(test['content'])):
        content.append(test['content'][index])
        characters.append(test['character'][index])
		
testtext = test['content']
testid = test['id']
testcharacter = test['character']
class TestDataset(Dataset):
    def __init__(self,text,character,maxlen):
        self.text = text
        self.maxlen = maxlen
        token_data,token_id,segment_id,mask_id = [],[],[],[]
        #sequence填充可以最后统一实现
        for index in tqdm(range(len(self.text))):
            current_text = text[index]
            #print('current_text:\t', current_text, len(current_text))
            current_character = characters[index]
            current_str_index = inv_new_content_dict[current_text]
            current_str_index_bak = inv_new_content_dict[current_text]
            current_key_index = dict_keys_index[current_str_index]
            current_key_index_bak = dict_keys_index[current_str_index]
            # 同一幕中所有上文拼在一起
            lcs_lst = []

            pre_current_key_index = current_key_index
            pre_current_key_text = current_text
            pre_content = ''
            #前面一个语句对应的索引以及文本内容
            num = 0
            while True:
                new_current_key_index = pre_current_key_index
                new_current_key_text = pre_current_key_text
                while new_current_key_index != -1 and new_content_dict[keys[new_current_key_index]] == new_current_key_text:
                    new_current_key_index = new_current_key_index-1
                if new_current_key_index == -1:
                    pre_content = ''
                    break
                # 只在同一幕中找
                new_str_index = keys[new_current_key_index]
                current_str_indexs = current_str_index.split('_')
                new_str_indexs = new_str_index.split('_')
                # 跨剧本，停止寻找
                if current_str_indexs[1] != new_str_indexs[1]:
                    break
                new_pre_content = new_content_dict[keys[new_current_key_index]]
                if str(new_pre_content) == 'nan':
                    new_pre_content = '无'
                if str(current_character) == 'nan':
                    break

                lcs_lst.append(new_pre_content)
                num = num+1
                
                pre_current_key_index = new_current_key_index
                pre_current_key_text = new_pre_content
                if num == 5:
                    break
            lcs_lst.reverse()
            if not lcs_lst:
                pre_content = '无'
            else:
                pre_content = ''.join(lcs_lst)
            # 统计有效上文长度
            
            if str(current_character) == 'nan':
                current_character = '无'

            current_character_token = tokenizer.tokenize(current_character)
            pre_token = tokenizer.tokenize(pre_content)
            current_token = tokenizer.tokenize(current_text)
            current_token = ["[CLS]"]+current_character_token+["[SEP]"]+pre_token+["[SEP]"]+current_token+["[SEP]"]
            current_id = tokenizer.convert_tokens_to_ids(current_token)            
            current_id = self.sequence_padding(current_id)
            token_data.append(current_token)
            token_id.append(current_id)
        self.token_data = token_data
        self.token_id = token_id
        #self.segment_id = sequence_padding(self.segment_id,maxlen)
        #self.mask_id = sequence_padding(self.mask_id,maxlen)
        self.tensors = [torch.tensor(self.token_id,dtype=torch.long)]
        
    def __len__(self):
        return len(self.token_id)
    
    def __getitem__(self,index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def sequence_padding(self,inputs,padding = 0):
        length = self.maxlen
        if len(inputs) > length:
            #print('inputs = ')
            #print(len(inputs))
            abondon = len(inputs)-length
            inputs = inputs[0:2]+inputs[3+abondon-1:]
            #print('now current_token = ')
            #print(len(inputs))
        outputs = []
        pad_width = (0,length-len(inputs))
        x = np.pad(inputs,pad_width,'constant',constant_values=padding)
        return x
		
test_dataset = TestDataset(testtext,testcharacter,maxlen=250)
test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)

for current_split in range(split_n):
    model = torch.load('best_score='+str(bestpoint[current_split])+'split='+str(current_split)+'.pth')
    
    model = model.to(device)
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    eval_predict_label = []
    index = []
    pred = [[],[],[],[],[],[]]
    current_index = 0
    for batch_token_ids in tqdm(test_loader):
        batch_token_ids = batch_token_ids[0].to(device)
        with torch.no_grad():
            output = model(batch_token_ids)
            for index in range(len(output)):
                #总的数据
                for index1 in range(len(output[index])):
                    #对应的类别概率0~6
                    abs0 = abs(output[index][index1]-0)
                    abs1 = abs(output[index][index1]-1)
                    abs2 = abs(output[index][index1]-2)
                    abs3 = abs(output[index][index1]-3)
                    currentdata = [abs0,abs1,abs2,abs3]
                    current_label = currentdata.index(min(currentdata))
                    #eval_predict_label[index1].append(current_label)
                    current_predict = output[index][index1].item()

                    if current_predict < 0.00:
                        current_predict = 0.0
                    elif current_predict > 3:
                        current_predict = 3.0

                    pred[index1].append(current_predict)
                    #eval_predict_label[index1].append(current_predict)
                    #当前类别的分类结果,这里append(output[index][index1])
                    #直接放入对应概率值时效果最好,这里直接放入0,1,2,3对应的数值
    for index in range(len(pred[0])):
        eval_predict_label.append(str(pred[0][index])+','+str(pred[1][index])+','+str(pred[2][index])+','+str(pred[3][index])+','+str(pred[4][index])+','+str(pred[5][index]))
    result_data = []
    for index in range(len(testid)):
        result_data.append([testid[index],eval_predict_label[index]])
    #pd.DataFrame({"id":testid,"label":eval_predict_label}).to_csv("/home/xiaoguzai/代码/剧本角色情感识别/数据集/crossentropy"+str(bestpoint)+"result.csv",index=False)
    import csv
    with open(r'/home/xiaoguzai/程序/剧本角色情感识别/训练数据/多折叠_上文+五句话_nezha'+str(seed)+'_best_point='+str(bestpoint[current_split])+"result.csv", 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow(['id', 'emotion'])  # 单行写入
        tsv_w.writerows(result_data)  # 多行写入

#模型融合
import pandas as pd
result = []
for current_split in range(split_n):
    current_result = pd.read_csv('/home/xiaoguzai/程序/剧本角色情感识别/训练数据/多折叠_上文+五句话_nezha'+str(seed)+'_best_point='+str(bestpoint[current_split])+"result.csv",sep='\t')
    result.append(current_result)

final_result = [[],[],[],[],[],[]]
test_id = []
for index in range(len(result[0]['emotion'])):
    test_id.append(result[0]['id'][index])
    currentdata = [0,0,0,0,0,0]
    for index1 in range(split_n):
        newdata = result[index1]['emotion'][index].split(',')
        currentdata[0] = currentdata[0]+float(newdata[0])
        currentdata[1] = currentdata[1]+float(newdata[1])
        currentdata[2] = currentdata[2]+float(newdata[2])
        currentdata[3] = currentdata[3]+float(newdata[3])
        currentdata[4] = currentdata[4]+float(newdata[4])
        currentdata[5] = currentdata[5]+float(newdata[5])
    currentdata[0] = currentdata[0]/split_n
    currentdata[1] = currentdata[1]/split_n
    currentdata[2] = currentdata[2]/split_n
    currentdata[3] = currentdata[3]/split_n
    currentdata[4] = currentdata[4]/split_n
    currentdata[5] = currentdata[5]/split_n
    #final_result.append(currentdata)
    final_result[0].append(currentdata[0])
    final_result[1].append(currentdata[1])
    final_result[2].append(currentdata[2])
    final_result[3].append(currentdata[3])
    final_result[4].append(currentdata[4])
    final_result[5].append(currentdata[5])
eval_predict_label = []
for index in range(len(pred[0])):
    eval_predict_label.append(str(final_result[0][index])+','+str(final_result[1][index])+','+str(final_result[2][index]) \
                               +','+str(final_result[3][index])+','+str(final_result[4][index])+','+str(final_result[5][index]))
result = []
for index in range(len(test_id)):
    result.append([test_id[index],eval_predict_label[index]])
with open(r'/home/xiaoguzai/程序/剧本角色情感识别/训练数据/多折'+str(split_n)+'折融合nezha+seed='+str(seed)+'+result1.csv','w') as f:
    csv_w = csv.writer(f,delimiter='\t')
    csv_w.writerow(['id','emotion'])
    csv_w.writerows(result)
