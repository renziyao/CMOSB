import codecs, json, pickle

f_dir = './result/sensorless_1/'
# f_dir = './credit_n3_4obj'
# f_dir = './synthetic_4obj'

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))

def check(gen):
    fn_fitness = f_dir + '/Generation_%d_saved_objectives/generation_%d_MLP_Pt_fitness.txt' % (gen, gen)
    fn_idx = f_dir + '/Generation_%d_saved_objectives/generation_%d_MLP_Pt_index.txt' % (gen, gen)
    fn_param = f_dir + '/model parameters/generation_%d_MLP_PARAM_Rt.txt' % (gen)
    fitness = None
    with open(fn_fitness, 'r') as f:
        fitness = eval(f.read().strip('\n'))
    idx = None
    with open(fn_idx, 'r') as f:
        idx = eval(f.read().strip('\n'))
    param = None
    with open(fn_param, 'r') as f:
        a = json.load(f)
        param = pickle_string_to_obj(a)
    sol = []
    for i in range(len(fitness[0])):
        if fitness[0][i] < 0.04 and fitness[2][i] < 0.150:
            sol.append(i)
    for i in range(len(sol)):
        # sol[i] = idx[sol[i]]
        sol[i] = (param[idx[sol[i]]], (fitness[0][sol[i]], fitness[1][sol[i]], fitness[2][sol[i]]))
    return sol

sol = []
for i in range(1, 41):
    try:
        sol.extend(check(i))
    except Exception as e:
        print(str(e))
for s in sol:
    print(s[1])
    print(s[0])
