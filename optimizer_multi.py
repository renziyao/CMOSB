import random
import pdb
# import tensorflow.compat.v1 as tf
import os
import json
import pickle
import codecs
import argparse
from Mutiobjective_NSGA2 import *
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

# from model import ParetoSetModel
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from secureboost.classifier import SecureBoostPlusClassifier
from datasets import (Synthetic, Sensorless)

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


# save_dir_mlp = os.path.join(os.getcwd(), 'result/sensorless_4')
save_dir_mlp = None


def generate_genotype_MLP(pop_size, tree_geno_size, depth_geno_size, max_lr):
    model_param = {}
    Pt_tree = []
    Pt_n_tree = []
    Pt_depth = []
    Pt_lr = []
    Pt_sample_rate = []
    Pt_max_purity = []
    for _ in range(pop_size):
        Pt_tree.append(np.random.randint(2, size=tree_geno_size).tolist())
        Pt_n_tree.append(np.random.randint(2, size=tree_geno_size).tolist())
        Pt_depth.append(np.random.randint(2, size=depth_geno_size).tolist())
        Pt_lr.append(np.random.uniform(0.01, max_lr))
        Pt_sample_rate.append(np.random.uniform(0.10, 1.0))
        Pt_max_purity.append(np.random.uniform(0.1, 1.0))
    model_param['TREE_NUM'] = Pt_tree
    model_param['N_TREE_NUM'] = Pt_n_tree
    model_param['TREE_DEPTH'] = Pt_depth
    model_param['LEARNING_RATE'] = Pt_lr
    model_param['SAMPLE_RATE'] = Pt_sample_rate
    model_param['MAX_PURITY'] = Pt_max_purity
    return model_param


def bin2int_plus1_multi(pt, geno_size):  # convert binary to int 00000->1
    translated = [np.array(pt[j: j+geno_size]).dot(2 ** np.arange(geno_size)[::-1]) + 1 for j in range(0, len(pt), geno_size)]
    # np.array([Pt[:,i:i+node_size].dot(2**np.arange(node_size)[::-1])+1 for i in range(0,Pt.shape[1],node_size)]).T
    return translated


def bin2int_plus1(pt, geno_size):  # convert binary to int 00000->1
    translated = np.array(pt).dot(2 ** np.arange(geno_size)[::-1]) + 1
    # np.array([Pt[:,i:i+node_size].dot(2**np.arange(node_size)[::-1])+1 for i in range(0,Pt.shape[1],node_size)]).T
    return translated


class Optimize_Mnist_MLP():
    def __init__(self, dataset, genotype_set, generations):
        self.model_param = generate_genotype_MLP(genotype_set['pop_size'],
                                                 genotype_set['tree_geno_size'],
                                                 genotype_set['depth_geno_size'],
                                                 genotype_set['max_lr'])
        self.pop_size = genotype_set['pop_size']
        self.max_lr = genotype_set['max_lr']
        self.tree_geno_size = genotype_set['tree_geno_size']
        self.depth_geno_size = genotype_set['depth_geno_size']
        self.generations = generations
        self.object_1 = None
        self.object_2 = None
        self.object_3 = None
        # self.object_4 = None
        self.frontier_1 = None
        self.frontier_2 = None
        self.frontier_3 = None
        # self.frontier_4 = None
        self.total_clients = 2
        self.dataset = dataset
        self.create_client_data()
        print(self.model_param)
    def create_client_data(self):
        dataset = self.dataset
        self.train_X, self.train_Y = dataset.get_train()
        self.train_Y = self.train_Y[0]
        self.test_X, self.test_Y = dataset.get_test()
        self.test_Y = self.test_Y[0]
        # IID = True
        # if Optimize_Mnist_MLP.IID == True:
        #     self.client_data = self.data.mnist_iid_shards(self.total_clients)
        # else:
        #     self.client_data = self.data.mnist_noniid(self.total_clients)

    def create_new_parents(self, F, fm, counter):
        # F - fast_nd_sort
        # fm - fitness
        # counter - generation
        distance_sort_index = []
        Pt_tree= []
        Pt_n_tree = []
        Pt_depth = []
        Pt_lr = []
        Pt_sample_rate = []
        Pt_max_purity = []
        f1 = []
        f2 = []
        f3 = []
        # f4 = []
        indexes = []  # store the indexes of new parent
        pareto_frontier = F[0]
        for front in F:
            distance_sort_index.extend(crowding_distance(front, fm))
        count = 0
        for i in distance_sort_index:
            Pt_tree.append(self.model_param['TREE_NUM'][i])
            Pt_n_tree.append(self.model_param['N_TREE_NUM'][i])
            Pt_depth.append(self.model_param['TREE_DEPTH'][i])
            Pt_lr.append(self.model_param['LEARNING_RATE'][i])
            Pt_sample_rate.append(self.model_param['SAMPLE_RATE'][i])
            Pt_max_purity.append(self.model_param['MAX_PURITY'][i])
            indexes.append(i)
            f1.append(fm[0][i])
            f2.append(fm[1][i])
            f3.append(fm[2][i])
            # f4.append(fm[3][i])

            count += 1
            print("count pop", count)
            if count >= self.pop_size:
                break
        f1_frontier = f1[0:len(pareto_frontier)]
        f2_frontier = f2[0:len(pareto_frontier)]
        f3_frontier = f3[0:len(pareto_frontier)]
        # f4_frontier = f4[0:len(pareto_frontier)]


        save_path = os.path.join(save_dir_mlp, 'Generation_' + str(counter) + '_saved_objectives')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        index_path = os.path.join(save_path, 'generation_' + str(counter) + '_MLP_Pt_index.txt')
        index_frontier = os.path.join(save_path, 'generation_' + str(counter) + '_MLP_Pt_frontier_index.txt')
        fitness_path = os.path.join(save_path, 'generation_' + str(counter) + '_MLP_Pt_fitness.txt')
        frontier_path = os.path.join(save_path, 'generation_' + str(counter) + '_MLP_Pt_frontier.txt')
        all_fitness_path = os.path.join(save_dir_mlp, 'fitness.txt')
        with open(index_path, 'w') as file:
            json.dump(indexes, file)  # save final Pt index
        with open(index_frontier, 'w') as file:
            json.dump(indexes[0:len(pareto_frontier)], file)
        with open(fitness_path, 'w') as file:
            # json.dump([f1] + [f2] + [f3] + [f4], file)
            json.dump([f1] + [f2] + [f3], file)
        with open(frontier_path, 'w') as file:
            # json.dump([f1_frontier] + [f2_frontier]+[f3_frontier] + [f4_frontier], file)
            json.dump([f1_frontier] + [f2_frontier]+[f3_frontier], file)
        with open(all_fitness_path, 'a') as file:
            # json.dump([f1] + [f2] + [f3] + [f4], file)
            json.dump([f1] + [f2] + [f3], file)
            file.write('\n')

        self.model_param['TREE_NUM'] = Pt_tree
        self.model_param['N_TREE_NUM'] = Pt_n_tree
        self.model_param['TREE_DEPTH'] = Pt_depth
        self.model_param['LEARNING_RATE'] = Pt_lr
        self.model_param['SAMPLE_RATE'] = Pt_sample_rate
        self.model_param['MAX_PURITY'] = Pt_max_purity

        self.object_1 = f1
        self.object_2 = f2
        self.object_3 = f3
        # self.object_4 = f4
        self.frontier_1 = f1_frontier
        self.frontier_2 = f2_frontier
        self.frontier_3 = f3_frontier
        # self.frontier_4 = f4_frontier

    def cross_mutation(self, cross_over_rate, mutation_rate, n_c, n_m):
        Qt_tree_num = bi_mut(cxOnePoint(self.model_param['TREE_NUM'], cross_over_rate), mutation_rate)
        self.model_param['TREE_NUM'] += Qt_tree_num
        Qt_n_tree_num = bi_mut(cxOnePoint(self.model_param['N_TREE_NUM'], cross_over_rate), mutation_rate)
        self.model_param['N_TREE_NUM'] += Qt_n_tree_num
        Qt_tree_depth = bi_mut(cxOnePoint(self.model_param['TREE_DEPTH'], cross_over_rate), mutation_rate)
        self.model_param['TREE_DEPTH'] += Qt_tree_depth
        Qt_lr = poly_mut(crossover_SBX(self.model_param['LEARNING_RATE'], n_c, cross_over_rate),
                         n_m, mutation_rate, self.max_lr, 0)
        self.model_param['LEARNING_RATE'] += Qt_lr
        Qt_sr = poly_mut(crossover_SBX(self.model_param['SAMPLE_RATE'], n_c, cross_over_rate),
                         n_m, mutation_rate, 1, 0.1)
        self.model_param['SAMPLE_RATE'] += Qt_sr
        Qt_mp = poly_mut(crossover_SBX(self.model_param['MAX_PURITY'], n_c, cross_over_rate),
                         n_m, mutation_rate, 1, 0.1)
        self.model_param['MAX_PURITY'] += Qt_mp

    def fitness(self, counter):
        model_param = {}
        model_params = {}
        f1 = []
        f2 = []
        f3 = []
        # f4 = []
        pop_len = len(self.model_param['TREE_NUM']) 
        for pop in range(pop_len):

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print('\n## generation {0}   population {1} ##'.format(counter, pop))

            model_param['TREE_NUM'] = bin2int_plus1(self.model_param['TREE_NUM'][pop], self.tree_geno_size)
            model_param['N_TREE_NUM'] = bin2int_plus1(self.model_param['N_TREE_NUM'][pop], self.tree_geno_size)
            model_param['TREE_DEPTH'] = bin2int_plus1(self.model_param['TREE_DEPTH'][pop], self.depth_geno_size)
            model_param['LEARNING_RATE'] = self.model_param['LEARNING_RATE'][pop]
            model_param['SAMPLE_RATE'] = self.model_param['SAMPLE_RATE'][pop]
            model_param['MAX_PURITY'] = self.model_param['MAX_PURITY'][pop]
            model_params[pop] = cy.deepcopy(model_param)
            print('model parameters: ', model_param, pop)
            if counter > 1 and pop < int(pop_len/2):
                f1.append(self.object_1[pop])
                f2.append(self.object_2[pop])
                f3.append(self.object_3[pop])
                # f4.append(self.object_4[pop])
                continue
            # fed = Fed_learning_SET_MLP(self.data, model_param)  # initialize parameters
            fed = SecureBoostPlusClassifier(
                n_clients=self.total_clients, 
                class_num=11, 
                tree_num=model_param['TREE_NUM'] + model_param['N_TREE_NUM'], 
                height=model_param['TREE_DEPTH'], 
                lr=model_param['LEARNING_RATE'], 
                other_rate=model_param['SAMPLE_RATE'], 
                non_fed_tree_num=model_param['N_TREE_NUM'],
                purity_threshold=model_param['MAX_PURITY'], 
            )
            fed.fit(self.train_X, self.train_Y)

            # y = fed.predict(self.test_X)
            y_pred = fed.predict(self.test_X)
            # auc = roc_auc_score(labels[0], y_pred)
            labels = self.test_Y
            auc = sum(y_pred == labels) / len(self.test_Y)
            
            training_time = sum(fed.metrics['training_time'])
            leaf_cnt = sum(fed.metrics['leaf_cnt'])
            privacy = max(fed.metrics['attack_acc'])
            print(auc, training_time, leaf_cnt, privacy)
            # fed.prepare_client_data(self.client_data)
            # test_error, No_parms, paramRate = fed.round_communication()

            if counter == self.generations and False:  # last generation save model
                model_name = 'MLP_WEIGHT_Rt_' + str(pop) + '.txt'
                param_name = 'MLP_PARAM_Rt_' + str(pop) + '.txt'
                mask_name = 'MLP_MASK_Rt_' + str(pop) + '.txt'
                if not os.path.isdir(save_dir_mlp):  # judge if the current path is directory
                    os.makedirs(save_dir_mlp)  # if not create directory
                model_path = os.path.join(save_dir_mlp, model_name)  # the path of saved model/ file name of the saved model
                param_path = os.path.join(save_dir_mlp, param_name)
                mask_path = os.path.join(save_dir_mlp, mask_name)
                with open(model_path, 'w') as file:
                    json.dump(obj_to_pickle_string(fed.Global.model.get_weights()), file)  # save weights
                with open(param_path, 'w') as file:
                    json.dump(obj_to_pickle_string(model_param), file)  # save model parameters
                with open(mask_path, 'w') as file:
                    json.dump(obj_to_pickle_string(fed.Global.mask), file)  # save model parameters

            # delete the current model and clear memory
            # del fed.Global.model
            # K.clear_session()
            # tf.reset_default_graph()

            f1.append(1 - auc)
            f2.append(training_time)
            f3.append(privacy)
            # f4.append(leaf_cnt)
            # fm = np.vstack((np.array(f1), np.array(f2), np.array(f3), np.array(f4)))
            fm = np.vstack((np.array(f1), np.array(f2), np.array(f3)))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        save_model_params = os.path.join(save_dir_mlp, 'model parameters')
        if not os.path.isdir(save_model_params):
            os.makedirs(save_model_params)
        params_path = os.path.join(save_model_params, 'generation_' + str(counter) + '_MLP_PARAM_Rt.txt')
        bin_params_path = os.path.join(save_model_params, 'generation_' + str(counter) + '_MLP_BIN_PARAM_Rt.txt')
        with open(params_path, 'w') as file:
            json.dump(obj_to_pickle_string(model_params), file)
        with open(bin_params_path, 'w') as file:
            json.dump(obj_to_pickle_string(self.model_param), file)

        # return fm, f1, f2, f3, f4
        return fm, f1, f2, f3

    def model_evolve(self):
        counter = 1
        for _ in range(self.generations):
            self.cross_mutation(cross_over_rate=0.9, mutation_rate=0.1, n_c=2, n_m=20)
            # fm, _, _,_, _ = self.fitness(counter)
            fm, _, _,_ = self.fitness(counter)
            F = fast_nd_sort(fm)
            self.create_new_parents(F, fm, counter)
            counter += 1

            
            # print("Generation_object: " + str(counter) + ",", list(self.object_1), list(self.object_2), list(self.object_3), list(self.object_4), self.model_param)
            # print("Generation_front: " + str(counter) + ",", list(self.frontier_1), list(self.frontier_2), list(self.frontier_3), list(self.frontier_4), self.model_param)
            print("Generation_object: " + str(counter) + ",", list(self.object_1), list(self.object_2), list(self.object_3), self.model_param)
            print("Generation_front: " + str(counter) + ",", list(self.frontier_1), list(self.frontier_2), list(self.frontier_3), self.model_param)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='result/')
    parser.add_argument('--dataset', type=str, default='synthetic2')
    parser.add_argument('--generations', type=int, default=40)
    args = parser.parse_args()

    # result dir
    save_dir_mlp = os.path.join(os.getcwd(), args.dir)
    # dataset
    dataset = None
    if args.dataset == 'synthetic2':
        dataset = Synthetic(n_samples=10000, n_clients=2, n_classes=10, n_clusters=1)
    elif args.dataset == 'sensorless':
        dataset = Sensorless(n_clients=2)
    else:
        ...
    genotype_set = {
         'pop_size': 20,
         'tree_geno_size': 4,
         'depth_geno_size': 3,
         'max_lr': 0.3}
    
    generations = args.generations

    optimize = Optimize_Mnist_MLP(dataset, genotype_set, generations)
    optimize.model_evolve()
