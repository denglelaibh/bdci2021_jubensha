import jieba
import numpy as np
import tensorflow as tf
import os
from tokenization import FullTokenizer
from tqdm import tqdm
import copy
from functools import partial
from p_tqdm import p_map
#tokenizer = FullTokenizer(vocab_file = '/home/xiaoguzai/代码/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别/bert-base-count3-HMCN-F/pretrain/bert_model/vocab.txt')
def nezha_pretraining_store_id_data(text_ids,text_tokens,vocab_file,begin,end):
    input_ids,labels = [],[]
    result_tuples = p_map(partial(random_mask,vocab_file=vocab_file,begin=begin,end=end),text_ids,text_tokens)
    def get_tuple0(input_ids):
        return input_ids[0]
    def get_tuple1(input_ids):
        return input_ids[1]
    input_ids = p_map(get_tuple0,result_tuples)
    labels = p_map(get_tuple1,result_tuples)
    return input_ids,labels

def random_mask(text_ids,text_tokens,vocab_file,begin,end):
    tokenizer = FullTokenizer(vocab_file = vocab_file)
    #对于只有id的脱敏数据进行ngram-mask
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    idx=0
    while idx<len(rands):
        if rands[idx]<0.15:#需要mask
            ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
            if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                ngram=2
            if ngram==2 and len(rands)<4:
                ngram=1
            L=idx+1
            R=idx+ngram#最终需要mask的右边界（开）
            while L<R and L<len(rands):
                rands[L]=np.random.random()*0.15#强制mask
                L+=1
            idx=R
            if idx<len(rands):
                rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
        idx+=1

    sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
    sep_id = sep_id[0]
    cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
    cls_id = cls_id[0]
    mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])
    mask_id = mask_id[0]
    for index in range(len(text_ids)):
        r = rands[index]
        i = text_ids[index]
        if text_tokens[index] in ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',
                                  '\uff0c','\u3302','\uff1f','\uff01']:
            input_ids.append(i)
            output_ids.append(-100)
            #保持原样不预测
            continue
        if r < 0.15 * 0.8:
            input_ids.append(mask_id)
            output_ids.append(i)#mask预测自己
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)#自己预测自己
        elif r < 0.15:
            input_ids.append(np.random.randint(begin,end))
            output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
        else:
            input_ids.append(i)
            output_ids.append(-100)#保持原样不预测
    return input_ids, output_ids

def nezha_pretraining_store_data(question_id,segment_id,question_text,mask_id,vocab_size):
    #对于全部数据进行bert-wwm的mask操作预先训练(有待改进)
    question_segment_pos = []
    question_segment_text = []
    for index in tqdm(range(len(question_text))):
        currents = jieba.cut(question_text[index])
        #currents = jieba.cut('$河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元$桐和街chang六liuliu$')
        def get_data(currents):
            current_list = []
            for data in currents:
                if not('\u4e00' <= data[0] <= '\u9fff'):
                    #print('data = ')
                    #print(data)
                    token_id1 = tokenizer.tokenize(data)
                    #print('token_id1 = ')
                    #print(token_id1)
                    current_list.extend(token_id1)
                #'345'拆成'3','4','5'
                else:
                    current_list.append(data)
                r"""
                if data[0] == '33':
                    if not('\u4e00' <= data[0] <= '\u9fff'):
                        token_id1 = tokenizer.tokenize(data)
                        print('999token_id1 = 999')
                        print(token_id1)
                        current_list.extend(token_id1)
                """
            return current_list
        current_list = get_data(currents)
        #print('999current_list = 999')
        #print(current_list)
        question_segment_text.append(current_list)
        current_pos = 0
        current_pos_list = []
        for data in current_list:
            if data == '$':
                current_pos_list.append([-1])
                current_pos = current_pos+1
            elif '\u4e00' <= data[0] <= '\u9fff':
                current_pos_list.append([index1 for index1 in range(current_pos,current_pos+len(data))])
                current_pos = current_pos+len(data)
            else:
                current_pos_list.append([current_pos])
                current_pos = current_pos+1
        question_segment_pos.append(current_pos_list)
    r"""
    question_segment_pos标记词语切分的位置
    比如$河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元$巩义市桐和街$
    切分出来的内容为
    ['$', '河南省', '巩义市', '新华路', '街道', '办事处', '桐和街', '6', '号', '钢苑', '新区', '3', '号楼', '一', '单元', '$', '巩义市', '桐和街', '$']
    接下来标记对应的位置数组
    [[-1],[1,2,3],[4,5,6],  [7,8,9], [10,11],..........]
    注意这里循环完成之后index的值会保留为45
    """
    question_new_id,question_new_segment = [],[]
    #question_origin_id = question_id.deepcopy()
    question_origin_id = copy.deepcopy(question_id)
    #!!!注意python中的等于为浅拷贝，copy()为深拷贝
    for cycle in range(4):
        #进行mask的次数
        #question_id = question_origin_id.deepcopy()
        question_id = copy.deepcopy(question_origin_id)
        for index in tqdm(range(len(question_segment_pos))):
            #注意这里的origin_data也需要深度拷贝
            origin_data = copy.deepcopy(question_id[index])
            for data in question_segment_pos[index]:
            #开头结尾的标志不变
                if len(data) == 1 and data[0] == -1:
                    continue
                rand1 = np.random.random()
                if rand1 <= 0.85:
                #以0.15的概率替换
                    continue
                else:
                    rand2 = np.random.random()
                    #print('rand2 = ')
                    #print(rand2)
                    #0.8概率用mask，0.1概率随机替换，0.1概率保持不变
                    if rand2 <= 0.8:
                        for index1 in data:
                            #print('index1 = ')
                            #print(index1)
                            try:
                                question_id[index][index1] = mask_id
                            except:
                            #抛出错误的原因:31号店先被jieba分词切分成了3,1号店
                            #接着又被tokenizer切分成了3,1,号店
                            #而实际上直接切分的时候bert会将31号店切分成31,号,店
                                r"""
                                print('situation2')
                                print('index = ')
                                print(index)
                                print('question_id = ')
                                #print(origin_data)
                                print(question_origin_id[index])
                                print('question_segment_text = ')
                                print(question_segment_text[index])
                                print('question_segment_pos = ')
                                print(question_segment_pos[index])
                                print('data = ')
                                print(data)
                                """
                                continue
                    elif rand2 <= 0.9:
                        for index1 in data:
                            try:
                                question_id[index][index1] = np.random.randint(0,vocab_size)
                            except:
                                r"""
                                print('situation3')
                                print('index = ')
                                print(index)
                                print('question_id = ')
                                print(origin_data)
                                print('question_segment_text = ')
                                print(question_segment_text[index])
                                print('question_segment_pos = ')
                                print(question_segment_pos[index])
                                print('data = ')
                                print(data)
                                """
                                continue
                    else:
                        continue
        question_new_id.extend(question_id)
        question_new_segment.extend(segment_id)
        serialized_instances = []
        for data in question_id:
            features = { 'label':tf.train.Feature(int64_list=tf.train.Int64List(value=data))}
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialized_instance = tf_example.SerializeToString()
            serialized_instances.append(serialized_instance)
        record_name = '/home/xiaoguzai/数据/预训练临时数据/corpus.tfrecord'+str(cycle)
        #注意这里不能加不存在的文件夹
        writer = tf.io.TFRecordWriter(record_name)
        for seralized_instance in serialized_instances:
            writer.write(seralized_instance)
        writer.close()
    #print('question_new_id = ')
    #print(question_new_id)
    return question_new_id,question_new_segment

def file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):  
        #print(root) #当前目录路径  
        #print(dirs) #当前路径下所有子目录  
        #print(files) #当前路径下所有非目录子文件
        file_list.append(files)
    return file_list
def nezha_pretraining_get_data():
    file_list = file_name('/home/xiaoguzai/数据/预训练临时数据/')
    def parse_function(serialized):
        features = {
            'label':tf.io.FixedLenFeature([],tf.int64),
        }
        features = tf.io.parse_single_example(serialized,features)
        token_ids = features['label']
        print('parse_function token_ids = ')
        print(token_ids)
        return token_ids
    dataset = tf.data.TFRecordDataset(file_list[0])
    dataset = dataset.map(parse_function)
    print('dataset = ')
    print(dataset)

def nezha_pretraining_get_new_data():
    file_list = file_name('/home/xiaoguzai/数据/预训练临时数据/')
    filename_queue = tf.train.string_input_producer(file_list)
    #根据文件名生成一个队列
    reader = tf.TFRecorderReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features = {
                                          'label':tf.io.FixedLenFeature([],tf.int64)
                                           })
    token_ids = tf.decode_raw(features['label'],tf.int)
    print('token_ids = ')
    print(token_ids)