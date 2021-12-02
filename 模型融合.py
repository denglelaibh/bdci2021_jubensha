import pandas as pd
result1 = pd.read_csv('/home/xiaoguzai/程序/剧本角色情感识别/训练数据/多折换主语为中文10折融合nezha+seed=1+result2.csv',sep='\t')
data1 = result1['emotion']
result2 = pd.read_csv('home/xiaoguzai/程序/剧本角色情感识别/训练数据/多折10折融合nezha+seed=1+result1.csv',sep='\t')
data2 = result2['emotion']

final_result = []
test_id = []
for index in range(len(result2['emotion'])):
    currentdata1 = result1['emotion'][index].split(',')
    currentdata2 = result2['emotion'][index].split(',')
    for index1 in range(len(currentdata1)):
        currentdata1[index1] = 0.5*float(currentdata1[index1])+0.5*float(currentdata2[index1])
    final_result.append(str(currentdata1[0])+','+str(currentdata1[1])+','+str(currentdata1[2])\
                                       +','+str(currentdata1[3])+','+str(currentdata1[4])+','+str(currentdata1[5]))
    test_id.append(result2['id'][index])

result = []
for index in range(len(test_id)):
    result.append([test_id[index],final_result[index]])

import csv
with open(r'/home/xiaoguzai/程序/剧本角色情感识别/数据/FINAL.csv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['id', 'emotion'])  # 单行写入
    tsv_w.writerows(result)  # 多行写入