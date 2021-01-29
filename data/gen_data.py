import loader
import random
import numpy as np
random.seed(2020)

root = 'hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet/train'
set_lt = 'data/imagenet_longtail_train.txt'
set_val = 'data/imagenet_longtail_val.txt'
set_100k = 'data/imagenet_100k_train.txt'

class KVDataset(loader.KVDataset):
    def get_percls_names(self):
        percls_names = [[] for _ in range(self.num_classes)]
        for key in self.keys:
            percls_names[self.key2target(key)].append(key)
        return percls_names

def gen_100k():
    fullset = KVDataset(root, 32, None, use_reader=True)
    ltset = KVDataset(root, 32, None, subset=set_lt, use_reader=True)
    valset = KVDataset(root, 32, None, subset=set_val, use_reader=True)
    full_names = fullset.get_percls_names()
    lt_names = ltset.get_percls_names()
    val_names = valset.get_percls_names()

    # Remove val names 
    for i in range(1000):
        for k in val_names[i]:
            full_names[i].remove(k)

    num_perc = len(ltset) // 1000
    perc_names = [[]] * 1000

    print('=> number of examples percls: ', num_perc)

    for i in range(1000):
        if len(lt_names[i]) >= num_perc:
            perc_names[i] = random.sample(lt_names[i], num_perc)
        else:
            for k in lt_names[i]:
                full_names[i].remove(k)
            perc_names[i] = lt_names[i] + \
                random.sample(full_names[i], num_perc-len(lt_names[i])) 

    total_names = sorted(sum(perc_names, []))

    with open(set_100k, 'w') as f:
        for i, name in enumerate(total_names):
            if i % 1000 == 0:
                print('{}/{}'.format(i, len(total_names)))
            f.write(name + ' ' + str(fullset.key2target(name)) + '\n')

def gen_100kp():
    set_100kp = 'data/imagenet_100kp_train.txt'
    fullset = KVDataset(root, 32, None, use_reader=True)
    ltset = KVDataset(root, 32, None, subset=set_lt, use_reader=True)
    valset = KVDataset(root, 32, None, subset=set_val, use_reader=True)
    set100k = KVDataset(root, 32, None, subset=set_100k, use_reader=True)
    full_names = fullset.get_percls_names()
    lt_names = ltset.get_percls_names()
    val_names = valset.get_percls_names()
    k100_names = set100k.get_percls_names()

    # Remove val names 
    for i in range(1000):
        for k in val_names[i]:
            full_names[i].remove(k)
    
        for k in k100_names[i]:
            # remove 100k names in fullset and ltset 
            if k in full_names[i]:
                full_names[i].remove(k)
            if k in lt_names[i]:
                lt_names[i].remove(k)

    num_perc = 116
    perc_names = [[]] * 1000

    print('=> number of examples percls: ', num_perc)

    for i in range(1000):
        perc_names[i] = k100_names[i].copy()
        if len(lt_names[i]) > 0:
            perc_names[i] += random.sample(lt_names[i], 1)
        else:
            perc_names[i] += random.sample(full_names[i], 1)


    total_names = sorted(sum(perc_names, []))
    print('Total number: ', len(total_names))

    with open(set_100kp, 'w') as f:
        for i, name in enumerate(total_names):
            if i % 1000 == 0:
                print('{}/{}'.format(i, len(total_names)))
            f.write(name + ' ' + str(fullset.key2target(name)) + '\n')

def gen_clsset():
    set1 = 'data/imagenet_clsset1.txt'
    set2 = 'data/imagenet_clsset2.txt'
    set1_comp = 'data/imagenet_clsset1_comp.txt'
    set2_comp = 'data/imagenet_clsset2_comp.txt'
    ltset = KVDataset(root, 32, None, subset=set_lt, use_reader=True)
    classes = ltset.classes

    with open('data/imagenet_clsfull.txt', 'w') as f:
        for c in classes:
            f.write(c+'\n')
    return 

    # with open(set1, 'w') as f:
    #     for c in classes[:500]:
    #         f.write(c+'\n')
    # with open(set1_comp, 'w') as f:
    #     for c in classes[500:]:
    #         f.write(c+'\n')   
    
    # random.seed(2020)
    # random.shuffle(classes)

    # with open(set2, 'w') as f:
    #     for c in sorted(classes[:500]):
    #         f.write(c+'\n')
    # with open(set2_comp, 'w') as f:
    #     for c in sorted(classes[500:]):
    #         f.write(c+'\n')

def gen_skewed():
    fullset = KVDataset(root, 32, None, use_reader=True)
    ltset = KVDataset(root, 32, None, subset=set_lt, use_reader=True)
    valset = KVDataset(root, 32, None, subset=set_val, use_reader=True)
    balset = KVDataset(root, 32, None, subset=set_100k, use_reader=True)
    full_names = fullset.get_percls_names()
    lt_names = ltset.get_percls_names()
    val_names = valset.get_percls_names()
    bal_names = balset.get_percls_names()

    # Remove val names 
    for i in range(1000):
        for k in val_names[i]:
            full_names[i].remove(k)

    # Get LT cls cnts 
    cnts = ltset.get_cls_cnts()
    def smooth(x, alpha):
        return np.power(x, alpha)

    def rescale(x):
        x *= cnts.mean() / x.mean()
        return np.round(x).astype(int)

    base_names = lt_names
    for a in [8, 6, 4, 2]:
        key = str(a)
        cur_cnts = rescale(smooth(cnts, a*0.1))

        cur_names = [[]] * 1000
        random.seed(2020)
        for i in range(1000):
            if len(base_names[i]) >= cur_cnts[i]:
                cur_names[i] = random.sample(base_names[i], cur_cnts[i])
            elif len(bal_names[i]) >= cur_cnts[i]:
                # import pdb; pdb.set_trace()
                cur_names[i] = random.sample(bal_names[i], cur_cnts[i])
            else:
                print('> class:', i)
                if len(base_names[i]) > len(bal_names[i]):
                    exist_names = base_names[i]
                else:
                    exist_names = bal_names[i]
                c_names = full_names[i].copy()
                for k in exist_names:
                    c_names.remove(k)
                cur_names[i] = exist_names + \
                    random.sample(c_names, cur_cnts[i]-len(exist_names))

        total_names = sorted(sum(cur_names, []))
        fname = 'data/imagenet_lt{}_train.txt'.format(key)
        with open(fname, 'w') as f:
            print('=> writing to ', fname)
            print('=> number of images:', len(total_names), 'max:',
                  cur_cnts.max(), 'min:', cur_cnts.min())
            for i, name in enumerate(total_names):
                if i % 1000 == 0:
                    print('{}/{}'.format(i, len(total_names)))
                f.write(name + ' ' + str(fullset.key2target(name)) + '\n')
        
        # reset base names 
        base_names = cur_names


def test_clsset():
    fullset = KVDataset(root, 32, None, use_reader=True,
                        clsset='data/imagenet_clsset1.txt')
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    # gen_clsset()
    # gen_skewed()
    gen_100kp()