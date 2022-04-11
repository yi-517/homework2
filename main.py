import pandas as pd
import tqdm
from matplotlib import pyplot as plt
import json


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def Create_C1(dataset):
    C1 = []
    for data in tqdm.tqdm(dataset):
        for item in data:
            if [item] not in C1:
                C1.append([item])
    print("Created success")
    return [frozenset(item) for item in C1]


def scan_D(dataset, Ck):
    Ck_count = dict()
    for data in dataset:
        for cand in Ck:
            if cand.issubset(data):
                if cand not in Ck_count:
                    Ck_count[cand] = 1
                else:
                    Ck_count[cand] += 1

    num_items = float(len(dataset))
    return_list = []
    support_data = dict()
    # 过滤非频繁项集
    for key in Ck_count:
        support = Ck_count[key] / num_items
        if support >= 0.1:
            return_list.insert(0, key)
        support_data[key] = support
    return return_list, support_data


def apriori_gen(Lk, k):
    return_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i + 1, len_Lk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                return_list.append(Lk[i] | Lk[j])
    return return_list


def apriori(dataset):
    C1 = Create_C1(dataset)
    dataset = [set(data) for data in dataset]
    L1, support_data = scan_D(dataset, C1)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = apriori_gen(L[k - 2], k)
        Lk, support_k = scan_D(dataset, Ck)
        support_data.update(support_k)
        L.append(Lk)
        k += 1
    return L, support_data


def generate_rules(L, support_data):
    big_rules_list = []
    for i in range(1, len(L)):
        for freq_set in L[i]:
            H1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_conseq(freq_set, H1, support_data, big_rules_list)
            else:
                cal_conf(freq_set, H1, support_data, big_rules_list)
    return big_rules_list


def rules_from_conseq(freq_set, H, support_data, big_rules_list):
    m = len(H[0])
    if len(freq_set) > (m + 1):
        Hmp1 = apriori_gen(H, m + 1)
        Hmp1 = cal_conf(freq_set, Hmp1, support_data, big_rules_list)
        if len(Hmp1) > 1:
            rules_from_conseq(freq_set, Hmp1, support_data, big_rules_list)


def cal_conf(freq_set, H, support_data, big_rules_list):
    prunedH = []
    for conseq in H:
        sup = support_data[freq_set]
        conf = sup / support_data[freq_set - conseq]
        lift = conf / support_data[freq_set - conseq]
        jaccard = sup / (support_data[freq_set - conseq] + support_data[conseq] - sup)
        if conf >= 0.5:
            big_rules_list.append((freq_set - conseq, conseq, sup, conf, lift, jaccard))
            prunedH.append(conseq)
    return prunedH


if __name__ == '__main__':
    print_hi('PyCharm')

    # 读取数据并合并
    column_names = ['country', 'description', 'designation', 'points', 'price',
                    'province', 'region_1', 'region_2', 'variety', 'winery']
    wine_data1 = pd.read_csv('./Wine Reviews/winemag-data_first150k.csv',
                             usecols=column_names)
    wine_data2 = pd.read_csv('./Wine Reviews/winemag-data-130k-v2.csv',
                             usecols=column_names)
    wine_data = pd.concat([wine_data2, wine_data1], ignore_index=True, sort=False)
    print(wine_data.info())
    print(wine_data.isnull().sum())

    # 数据缺失值处理
    df1 = wine_data.dropna(how='any')
    wine_data.dropna(axis=0, inplace=False)
    print(df1.isnull().sum())

    # 转换成适合进行关联规则挖掘的形式
    data_list = df1.values.tolist()
    dataset = []
    for item in data_list:
        ds = []
        for i, value in enumerate(item):
            if not value:
                ds.append((column_names[i], 'NA'))
            else:
                ds.append((column_names[i], value))
        dataset.append(ds)
    print(dataset[0])

    # 频繁模式挖掘
    freq_set, sup_rata = apriori(dataset)
    sup_rata_out = sorted(sup_rata.items(), key=lambda d: d[1], reverse=True)
    freq_set_file = open("fterms.json", 'w')
    for (key, value) in sup_rata_out:
        result_dict = {'set': None, 'sup': None}
        set_result = list(key)
        sup_result = value
        if sup_result < 0.1:
            continue
        result_dict['set'] = set_result
        result_dict['sup'] = sup_result
        json_str = json.dumps(result_dict, ensure_ascii=False)
        freq_set_file.write(json_str + '\n')
    freq_set_file.close()
    with open("fterms.json") as f1:
        freq = [json.loads(each) for each in f1.readlines()]
        freq_sup = [each["sup"] for each in freq]
        plt.figure()
        plt.title("Frequent item")
        plt.boxplot(freq_sup)
        plt.show()
    print(freq)

    # 关联规则挖掘, 计算支持度和置信度, 并使用lift、jaccard对规则进行评价
    strong_rules_list = generate_rules(freq_set, sup_rata)
    strong_rules_list = sorted(strong_rules_list, key=lambda x: x[3], reverse=True)
    rules_file = open('rules.json', 'w')
    for result in strong_rules_list:
        result_dict = {'X_set': None, 'Y_set': None, 'sup': None, 'conf': None,
                       'lift': None, 'jaccard': None}
        X_set, Y_set, sup, conf, lift, jaccard = result
        result_dict['X_set'] = list(X_set)
        result_dict['Y_set'] = list(Y_set)
        result_dict['sup'] = sup
        result_dict['conf'] = conf
        result_dict['lift'] = lift
        result_dict['jaccard'] = jaccard
        json_str = json.dumps(result_dict, ensure_ascii=False)
        rules_file.write(json_str + '\n')
    rules_file.close()
    with open("rules.json") as f2:
        rules = [json.loads(each) for each in f2.readlines()]
        rules_sup = [each["sup"] for each in rules]
        rules_conf = [each["conf"] for each in rules]
        fig = plt.figure("rule")
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title("Rules")
        ax.scatter(rules_sup, rules_conf, marker='o', color='red')
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        plt.show()
    # print(rules)
