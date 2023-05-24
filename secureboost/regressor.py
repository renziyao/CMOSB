import time
import random
import math
import copy
import pdb
import numpy as np
import IPython
from phe import paillier
from tqdm import trange
from sklearn.metrics import roc_auc_score
from sklearn.cluster import SpectralClustering

SEED = 42

class SecureBoostPlusRegressor():
    def __init__(
            self, 
            n_clients, 
            tree_num=30, 
            bin_num=32, 
            height=3, 
            key_size=1024, 
            lr=0.3, 
            top_rate=0.0, 
            other_rate=1.0,
            purity_threshold=1.0, 
            non_fed_tree_num=0, 
        ):
        # HE
        self.enable_encrypt = False # change here to use HE
        self.key_size = key_size
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=self.key_size)
        # HE compression
        self.r = 12
        self.er = 2**self.r
        self.M = 32
        self.width = 2**self.M
        # DT
        self.bin_num = bin_num
        self.tree_num = tree_num
        self.height = height
        self.lr = lr
        self.lamb = 0.1
        # BOOST - GOSS
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.class_num = 2
        # Privacy
        self.purity_threshold = purity_threshold
        self.non_fed_tree_num = non_fed_tree_num
        # TIME
        self.estimate_time = {
            'encrypt': [0.009771301489999999, 1], 
            'addition': [7.896339633333346e-06, 1], 
            'decrypt': [7.788100000141184e-05, 1], 
        }
        # Fed
        self.n_clients = n_clients
        class Client():
            ...
        self.clients = {
            i: Client() for i in range(n_clients)
        }
        self.metrics = {
            'accuracy': [], 
            'auc': [], 
            'loss': [], 
            'leaf_purity': [], 
            'training_time': [], 
            'leaf_cnt': [], 
            'attack_acc': [], 
        }
        self.clients[0].GHs = [None for _ in range(1 << (self.height + 1))]
        self.clients[0].GHnums = [None for _ in range(1 << (self.height + 1))]
        self.all_sample = None
        self.mat = None
        self.rnd = random.Random()
        self.rnd.seed(SEED)

    def calculate_histogram(self, I):
        l_k = self.bin_num
        GHs = [None for _ in range(self.n_clients)]
        GHnums = [None for _ in range(self.n_clients)]
        cnt = 0
        t1 = time.process_time()
        for cid in range(self.n_clients):
            gh_vec = self.clients[cid].enc_gh
            if cid == 0: gh_vec = self.clients[0].raw_gh
            GH_vec = [[0 for _ in range(l_k + 1)] for _ in range(self.clients[cid].features.shape[1])]
            GH_num = [[0 for _ in range(l_k + 1)] for _ in range(self.clients[cid].features.shape[1])]
            for i in range(self.clients[cid].features.shape[1]):
                for j in I:
                    GH_vec[i][self.clients[cid].features_bin[j][i]] += gh_vec[j]
                    GH_num[i][self.clients[cid].features_bin[j][i]] += 1
                    if cid > 0: cnt += 1
            GHs[cid] = GH_vec
            GHnums[cid] = GH_num
        t2 = time.process_time()
        if self.enable_encrypt:
            self.training_time('addition', cnt, t2 - t1)
            self.metrics['training_time'][-1] += t2 - t1
        else:
            self.metrics['training_time'][-1] += self.training_time('addition', cnt)
        return GHs, GHnums

    def subtract_hist(self, idx):
        gh1, gh_num1, gh2, gh_num2 = self.clients[0].GHs[idx//2], self.clients[0].GHnums[idx//2], self.clients[0].GHs[idx-1], self.clients[0].GHnums[idx-1]
        GH = [[[gh1[i][j][k]-gh2[i][j][k] for k in range(self.bin_num + 1)] for j in range(len(gh1[i]))] for i in range(self.n_clients)]
        GHnum = [[[gh_num1[i][j][k]-gh_num2[i][j][k] for k in range(self.bin_num + 1)] for j in range(len(gh1[i]))] for i in range(self.n_clients)]
        return GH, GHnum

    def binning(self, features):
        eps = 1e-3
        l_k = self.bin_num
        s = np.zeros((features.shape[1], l_k + 1))
        features_bin = np.zeros_like(features, dtype=int)
        for i in range(features.shape[1]):
            attributes = features[:, i]
            for j in range(l_k + 1):
                s[i][j] = np.percentile(attributes, j / l_k * 100)
                if j == 0:
                    s[i][j] -= eps
            for j in range(attributes.shape[0]):
                for k in range(1, l_k + 1):
                    if s[i][k - 1] < attributes[j] <= s[i][k]:
                        features_bin[j][i] = k
        return s, features_bin

    def split_finding(self, ix, I, Fed=True):
        gh = sum([self.clients[0].raw_gh[i] for i in I])
        g, h = self.gh_unpack(gh, len(I))
        batch_size = self.key_size // self.M // 2
        opt_val = {}
        # print('sample count: %d' % len(I))
        # print('g=%f, h=%f' % (g, h))
        opt_val_client = []
        score = -1e9
        t1 = time.process_time()
        cnt = 0
        for i in range(self.n_clients):
            if len(self.clients[0].tree_roots) < self.non_fed_tree_num and i > 0:
                break
            if Fed == False and i > 0:
                break
            GH = copy.deepcopy(self.clients[0].GHs[ix][i])
            GH_num = copy.deepcopy(self.clients[0].GHnums[ix][i])
            if i > 0: cnt += len(GH)
            for j in range(len(GH)):
                for v in range(1, self.bin_num + 1):
                    GH[j][v] += GH[j][v-1]
                    GH_num[j][v] += GH_num[j][v-1]
                idx = 0
                while idx * batch_size < self.bin_num:
                    V_GH = 0
                    for t in range(min((idx + 1) * batch_size, self.bin_num), idx * batch_size, -1):
                        V_GH = V_GH*self.width**2 + GH[j][t]
                    if i>0:
                        # if not V_GH==0: V_GH = self.private_key.decrypt(V_GH)
                        if not V_GH==0:
                            if self.enable_encrypt:
                                V_GH = self.private_key.decrypt(V_GH)

                    for v in range(idx * batch_size + 1, min((idx + 1) * batch_size + 1, self.bin_num + 1)):
                        gh = V_GH % (self.width**2)
                        V_GH >>= 2*self.M
                        g_l, h_l = self.gh_unpack(gh, GH_num[j][v])
                        g_r = g - g_l
                        h_r = h - h_l
                        
                        new_score = g_l*g_l / (h_l + self.lamb) + \
                                        g_r*g_r / (h_r + self.lamb) - \
                                        g*g / (h + self.lamb)
                        # print('k=%d v=%d score=%f, g_l=%f, h_l=%f' % (j, v, new_score, g_l, h_l, ))
                        if new_score > score:
                            score = new_score
                            opt_val = {
                                "k_opt": j, 
                                "v_opt": v, 
                                "i_opt": i, 
                                "g_l": g_l, 
                                "g_r": g_r, 
                                "h_l": h_l,
                                "h_r": h_r, 
                                "score": score
                            }
                    idx += 1
            score = -1e9
            opt_val_client.append(opt_val)
            opt_val = {}
        t2 = time.process_time()
        if self.enable_encrypt:
            self.training_time('decrypt', cnt, t2 - t1)
            self.metrics['training_time'][-1] += t2 - t1
        else:
            self.metrics['training_time'][-1] += self.training_time('decrypt', cnt)
        # print('gain_max', score)
        # print(opt_val)
        for opt_val in sorted(opt_val_client, key=lambda x: x['score'], reverse=True):
            if Fed == False and opt_val["i_opt"] != 0: continue
            cid = opt_val["i_opt"]
            cut = self.clients[cid].s[opt_val["k_opt"]][opt_val["v_opt"]]
            # print('cut', cut)
            I_l = [i for i in I if self.clients[cid].features[i][opt_val["k_opt"]] <= cut]
            I_r = [i for i in I if self.clients[cid].features[i][opt_val["k_opt"]] > cut]
            if len(I_l) == 0 or len(I_r) == 0: break
            # calculate purity
            sample = self.clients[0].labels[I_l]
            purity = 0
            weight = len(sample) / len(I)
            p1 = sum(sample) / len(sample)
            p0 = 1 - p1
            purity += weight * max(p0, p1)
            sample = self.clients[0].labels[I_r]
            weight = len(sample) / len(I)
            p1 = sum(sample) / len(sample)
            p0 = 1 - p1
            purity += weight * max(p0, p1)
            # calculate purity end
            if purity > self.purity_threshold:
                Fed = False
                if opt_val["i_opt"] != 0: continue
            record_id = len(self.clients[cid].records)
            self.clients[cid].records[record_id] = [opt_val["k_opt"], cut]
            return cid, record_id, I_l, I_r, opt_val, Fed
        return None
    
    def gh_pack(self, raw_g, raw_h):
        raw_g += 1.0
        v_g = int(raw_g * self.er)
        v_h = int(raw_h * self.er)
        v_gh = v_g * self.width + v_h
        return v_gh
    
    def gh_unpack(self, v_gh, n=1):
        v_g, v_h = (v_gh / self.width), (v_gh % self.width)
        raw_g = v_g / self.er - 1.0*n
        raw_h = v_h / self.er
        return raw_g, raw_h
    
    def training_time(self, op, count, t=None):
        if t == None:
            return count * self.estimate_time[op][0] / self.estimate_time[op][1]
        if op not in self.estimate_time:
            self.estimate_time[op] = []
        self.estimate_time[op][0] += t
        self.estimate_time[op][1] += count

    def calculate_gradient_and_hess(self):
        t1 = time.process_time()
        raw_gh = dict()
        enc_gh = dict()
        for i in self.clients[0].instance_sample:
            y = self.clients[0].labels[i]
            y_pred = self.clients[0].y_pred[i]
            # raw_gh[i] = self.gh_pack(
            #     math.exp(p)/(1+math.exp(p))-y, 
            #     math.exp(p)/(1+math.exp(p))**2
            # )
            raw_gh[i] = self.gh_pack(
                y_pred - y, 
                y_pred * (1 - y_pred)
            )
            if self.enable_encrypt:
                enc_gh[i] = self.public_key.encrypt(raw_gh[i])
            else:
                enc_gh[i] = copy.deepcopy(raw_gh[i])
        self.clients[0].raw_gh = raw_gh
        self.clients[0].enc_gh = enc_gh
        t2 = time.process_time()
        if self.enable_encrypt:
            self.training_time('encrypt', len(self.clients[0].instance_sample), t2 - t1)
            self.metrics['training_time'][-1] += t2 - t1
        else:
            if len(self.clients[0].tree_roots) >= self.non_fed_tree_num:
                self.metrics['training_time'][-1] += self.training_time('encrypt', len(self.clients[0].instance_sample))

    def get_samples(self):
        raw_g = []
        for i in range(len(self.clients[0].labels)):
            y = self.clients[0].labels[i]
            y_pred = self.clients[0].y_pred[i]
            # raw_g.append(math.exp(p)/(1+math.exp(p))-y)
            raw_g.append(abs(y_pred - y))
        g_sorted = sorted(enumerate(raw_g), key=lambda x:x[1])
        inds = [g[0] for g in g_sorted]
        ptr = int((1-self.top_rate) * len(g_sorted))
        inds_left, inds_right = inds[:ptr], inds[ptr:]
        sample_left = self.rnd.sample(inds_left, int(self.other_rate*len(inds)))
        return sample_left+inds_right
    
    def calculate_leaf_purity(self, tree, sample_ids):
        purity = 0
        for i in range(1, 1 << (self.height + 1)):
            if tree[i] != None or sample_ids[i] == None:
                continue
            sample = self.clients[0].labels[sample_ids[i]]
            weight = len(sample) / len(sample_ids[1])
            p1 = sum(sample) / len(sample)
            p0 = 1 - p1
            purity += weight * max(p0, p1)
            # print(weight, p0, p1)
            # print("weight: %f, p0: %f, p1: %f, contrib: %f" % (weight, p0, p1, weight * max(p0, p1)))
        return purity
    
    def calculate_privacy(self, tree, tree_fed, sample_ids):
        n = 1000
        if self.all_sample == None:
            rnd = random.Random()
            rnd.seed(SEED)
            self.all_sample = []
            for i in range(self.class_num):
                self.all_sample.extend(rnd.sample(list(np.where(self.clients[0].labels == i)[0]), n // self.class_num))
            self.mat = np.zeros((n, n), dtype=int)
        for i in range(1, 1 << (self.height + 1)):
            if sample_ids[i] == None:
                continue
            if tree[i] != None and (tree_fed[i * 2] == True or tree_fed[i * 2 + 1] == True):
                continue
            if tree_fed[i] == False:
                continue
            # print('leaf: ', i)
            leaf_sample = set(sample_ids[i])
            for a in range(n):
                f1 = (self.all_sample[a] in leaf_sample)
                if not f1: continue
                for b in range(a + 1, n):
                    f2 = (self.all_sample[b] in leaf_sample)
                    if not f2: continue
                    self.mat[a][b] += 1
                    self.mat[b][a] += 1
        self.mat[0][0] += 1
        dis = self.mat.astype(float) / self.mat[0][0]
        dis[0][0] = 0
        dis += np.eye(n)
        sc = SpectralClustering(n_clusters=self.class_num, assign_labels='cluster_qr', affinity='precomputed', random_state=SEED)
        y_pred = sc.fit_predict(dis)
        y = self.clients[0].labels[self.all_sample]
        all_cls = [i for i in range(self.class_num)]
        cnt = np.zeros((self.class_num, self.class_num), dtype=int)
        for i in range(self.class_num):
            for j in range(self.class_num):
                cnt[i][j] = sum((y_pred==i) & (y==j))
        correct = 0
        for i in range(self.class_num):
            max_cls = -1
            max_cnt = -1
            for j in all_cls:
                if max_cnt < cnt[i][j]:
                    max_cls = j
                    max_cnt = cnt[i][j]
            all_cls.remove(max_cls)
            correct += max_cnt
        attack_acc = correct / n
        return attack_acc
    
    def calculate_leaf_cnt(self, tree, sample_ids):
        leaf_cnt = 0
        for i in range(1, 1 << (self.height + 1)):
            if tree[i] != None or sample_ids[i] == None:
                continue
            leaf_cnt += 1
        return leaf_cnt

    def build_tree(self):
        self.clients[0].instance_sample = self.get_samples()
        height = self.height
        sample_ids = [None for i in range(1 << (height + 1))]
        sample_ids[1] = self.clients[0].instance_sample
        tree = [None for i in range(1 << (height + 1))] # client_id, record_id 
        tree_fed = [True for i in range(1 << (height + 1))]
        val = [None for i in range(1 << (height + 1))]
        val[1] = (
            sum([self.clients[0].labels[idx] for idx in sample_ids[1]]) - 
            sum([self.clients[0].y_pred[idx] for idx in sample_ids[1]])
        )/len(sample_ids[1])
        self.calculate_gradient_and_hess()
        for i in range(1, self.n_clients):
            self.clients[i].enc_gh = copy.deepcopy(self.clients[0].enc_gh)
        # cnt = 0
        for layer in range(1, height+1):
            # print('leaf purity: %.6f' % self.calculate_leaf_purity(tree, sample_ids))
            for i in range(1<<(layer-1), 1<<layer):
                if sample_ids[i] == None: continue
                if i * 2 + 1 < (1 << (height + 1)):
                    # if not leaf node
                    if i % 2 == 0 or i == 1:
                        GH, GHnum = self.calculate_histogram(sample_ids[i])
                    else:
                        GH, GHnum = self.subtract_hist(i)
                    self.clients[0].GHs[i] = GH
                    self.clients[0].GHnums[i] = GHnum
                    sp = self.split_finding(i, sample_ids[i], tree_fed[i])
                    if sp == None: continue
                    i_opt, record_id, I_l, I_r, opt_val, Fed = sp
                    # if len(I_l)>0 and len(I_r)>0 :
                    sample_ids[i * 2] = I_l
                    sample_ids[i * 2 + 1] = I_r
                    val[i * 2] = -opt_val['g_l'] / (opt_val['h_l'] + self.lamb)
                    val[i * 2 + 1] = -opt_val['g_r'] / (opt_val['h_r'] + self.lamb)
                    if i * 2 >= (1 << height):
                        tree_fed[i * 2] = False
                        tree_fed[i * 2 + 1] = False
                    else:
                        tree_fed[i * 2] = Fed
                        tree_fed[i * 2 + 1] = Fed
                    tree[i] = (i_opt, record_id)
                    # else:
                    #     try: assert not val[i] == None
                    #     except: print((i, len(I_l), len(I_r), len(sample_ids[i])))
                    #     cnt += len(sample_ids[i])
                    #     # for s in sample_ids[i]: self.cur_preds[s] = val[i]
                # else:
                #     try: assert not val[i] == None
                #     except: print((i, len(sample_ids[i])))
                #     cnt += len(sample_ids[i])
                #     # for s in sample_ids[i]: self.cur_preds[s] = val[i]
        tree[0] = val
        # print('leaf weight', val)
        # calculate purity
        # purity = self.calculate_leaf_purity(tree, sample_ids)
        leaf_cnt = self.calculate_leaf_cnt(tree, sample_ids)
        if len(self.clients[0].tree_roots) >= self.non_fed_tree_num:
            attack_acc = self.calculate_privacy(tree, tree_fed, sample_ids)
            self.metrics['attack_acc'].append(attack_acc)
        self.metrics['leaf_purity'].append(-1)
        self.metrics['leaf_cnt'].append(leaf_cnt)
        return tree
    
    def fit(self, features, labels):
        # load data
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))
        for i, item in enumerate(features):
            self.clients[i].features = item
            self.clients[i].s, self.clients[i].features_bin = self.binning(item)
            self.clients[i].records = {}
        self.clients[0].labels = labels
        self.clients[0].tree_roots = []
        y_tmp = np.zeros((labels.shape[0]))
        self.clients[0].y_pred = np.zeros((labels.shape[0]))
        for i in trange(self.tree_num):
            self.metrics['training_time'].append(0)
            for j in range(labels.shape[0]):
                self.clients[0].y_pred[j] = sigmoid(y_tmp[j])
            self.clients[0].tree_roots.append(self.build_tree())
            y_tmp += self.predict_prob_single(features, self.clients[0].tree_roots[i])
            # calculate loss/auc/acc
            y_pred = self.clients[0].y_pred
            y = np.array([(1 if pred>0.5 else 0) for pred in y_pred])
            auc = roc_auc_score(labels, y_pred)
            loss = 0
            correct = 0
            for i in range(y.shape[0]):
                loss += -labels[i] * math.log(y_pred[i]) - (1 - labels[i]) * math.log(1 - y_pred[i])
                if y[i] == labels[i]:
                    correct += 1
            loss /= y.shape[0]
            # print(y)
            # print(labels)
            total = int(len(y))
            self.metrics['accuracy'].append(correct / total)
            self.metrics['auc'].append(auc)
            self.metrics['loss'].append(loss)
            # print('test accuracy: %.6f' % (correct / total))
            # print('test auc: %.6f' % (auc))
            # print('test loss: %.6f' % (loss))
    
    def predict_prob_single(self, features, tree):
        ypd = np.zeros((features[0].shape[0]))
        for i in range(features[0].shape[0]):
            pos = 1
            while pos * 2 + 1 < len(tree):
                if tree[pos] == None: break
                (client_id, record_id) = tree[pos]
                (attr_id, cut) = self.clients[client_id].records[record_id]
                attr = features[client_id][i][attr_id]
                if attr <= cut: pos = pos * 2
                else: pos = pos * 2 + 1
            ypd[i] += tree[0][pos] * self.lr
        return ypd

    def predict_prob(self, features):
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))
        y_pred = np.zeros((features[0].shape[0]))
        for j, tree in enumerate(self.clients[0].tree_roots):
            y_pred += self.predict_prob_single(features, tree)
        for i in range(features[0].shape[0]):
            y_pred[i] = sigmoid(y_pred[i])
        return y_pred

    def predict(self, features):
        y_pred = self.predict_prob(features)
        return np.array([(1 if pred>0.5 else 0) for pred in y_pred])
