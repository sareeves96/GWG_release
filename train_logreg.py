import os
import numpy as np
from evcouplings.couplings import MeanFieldDCA, MeanFieldCouplingsModel
from evcouplings.align import Alignment, tools, map_matrix
from evcouplings.compare import DistanceMap, sifts, distances
import pickle
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def aln_convert(f_name):
    with open(f_name, 'r') as f_org:
        with open(os.path.splitext(f_name)[0]+'.txt', 'w') as f_new:
            #f_new.write('CLUSTAL W (1.82) multiple sequence alignment\n')
            for i, line in enumerate(f_org.readlines()):
                f_new.write(f'>seq{i}|/{"1-"+str(len(line)-1)}\n{line}')

def reconstruct_c(vals):
    target = len(vals)
    current = 0
    i = 0
    while target != current:
        i += 1
        current += i
    indices = np.triu_indices(i+1, 1)
    c = np.zeros([i+1, i+1])
    c[indices] = vals
    c += c.transpose()
    return c

def generate_features(f_name):
    print("Loading alignment...")
    with open(f_name, "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")
        # aln.ids[0] = aln.ids[0] + f'/1-{len(aln[0])}'
        # print(aln.ids[0])
    dca = MeanFieldDCA(aln)
    model = dca.fit()
    cov = dca.compute_covariance_matrix()
    mi = model.mi_scores_raw
    di = model.di_scores
    fn = model.fn_scores
    cn = model.cn_scores
    print(f'Covariance matrix dims: {cov.shape}')
    print(f'Mutual information matrix dims: {mi.shape}')
    print(f'Direct information matrix dims: {di.shape}')
    print(f'Frobenius norm matrix dims: {fn.shape}')
    print(f'Corrected norm matrix dims: {cn.shape}')
    return cov, mi, di, fn, cn


def generate_contact_map_monomer(f_name, cutoff=5.):
    print('Generating distance map')
    code = os.path.split(f_name)[-1]
    pdb_code, pdb_chain = code[:4], code[4]
    print(pdb_code, pdb_chain)
    res = sifts.SIFTS('./pdb_chain_uniprot.csv').by_pdb_id(pdb_code, pdb_chain=pdb_chain)
    dist_map = distances.intra_dists(res)
    dist_map.to_file(f_name.replace(f'aln/{code}', f'map/{code}'))
    J = -dist_map.dist_matrix
    print(J.shape)
    J = J + cutoff
    J[J < 0] = 0.
    J[np.isnan(J)] = 0.  # treat unobserved values as just having no contact
    ind = np.diag_indices(J.shape[0])
    J[ind] = 0.
    C = np.copy(J)
    C[C > 0] = 1.
    C[C <= 0] = 0.
    return C


def generate_train_dataset(train_data_directory='/mnt/c/Users/gooud/GWG_release/databases/aln'):
    missed = []
    for f in os.listdir(train_data_directory):
        features = []
        labels = []
        file = os.path.join(train_data_directory, f)
        print(os.path.splitext(file))
        print(os.path.splitext(file)[0]+'.txt')
        if os.path.exists(os.path.splitext(file)[0]+'.pkl'):
            print(f'Skipping {file}')
            continue
        if os.path.splitext(file)[-1] == '.aln' and not os.path.exists(os.path.splitext(file)[0]+'.txt'):
            aln_convert(file)
            file = os.path.splitext(file)[0]+'.txt'
        #elif os.path.splitext(file)[-1] == '.aln':
        #    continue
        elif os.path.splitext(file)[-1] != '.txt':
            print(f"Skipped {file}")
            continue
        try:
            cov, mi, di, fn, cn = generate_features(file)
            cm = generate_contact_map_monomer(file)
        except ValueError:
            print(f'Couldn\'t load data for file {file}')
            missed.append(file)
            continue
        if mi.shape != cm.shape:
            print(f'incongruent shapes: {mi.shape}, {cm.shape}')
            missed.append(file)
            continue
        idx1, idx2 = np.triu_indices(cm.shape[0], 1)
        step = cm.shape[0]
        for idx in list(zip(idx1, idx2)):
            #print(cm)
            label = cm[idx]
            #print(idx)
            #print(cm[idx])
            labels.append(label)
            ax1 = slice(idx[0],None,step)
            ax2 = slice(idx[1],None,step)
            covs = cov[ax1,ax2].flatten()
            mutual_info = mi[idx]
            direct_info = di[idx]
            frob = fn[idx]
            corr = cn[idx]
            #print(covs)
            #print(mutual_info)
            #print(direct_info)
            #print(frob)
            #print(corr)
            feature = covs
            for num in [mutual_info, direct_info, frob, corr]:
                feature = np.append(feature ,num)
            features.append(feature)
        pickle.dump(zip(features, labels), open(os.path.splitext(file)[0]+'.pkl', 'wb'))

    print(missed)


def load_train_data(path, structure):
    file = os.path.join(path, structure+'.pkl')
    with open(file, 'rb') as curr:
        training_data = pickle.load(curr)
    features_positive = []
    features_negative = []
    for feature, label in training_data:
        if label == 1:
            features_positive.append(feature)
        else:
            features_negative.append(feature)
    features_negative = random.sample(features_negative, len(features_positive))
    labels_positive = list(np.ones(len(features_positive)))
    labels_negative = list(np.zeros(len(features_negative)))
    features = features_positive + features_negative
    labels = labels_positive + labels_negative
    features_shuffled, labels_shuffled = shuffle(features, labels)
    return features_shuffled, labels_shuffled


def load_test_data(path, structure):
    file = os.path.join(path, structure+'.pkl')
    with open(file, 'rb') as curr:
        features, labels = zip(*pickle.load(curr))
    c = reconstruct_c(labels)
    return features, labels, c


def train_and_test_lr_model(pkl_path, n_train, n_test, save_loc, name, mask=np.array([1]*400+[0]*4)):
    os.makedirs(save_loc, exist_ok=True)
    structures = [os.path.splitext(s)[0] for s in os.listdir(pkl_path) if '.pkl' in s]
    ind = random.sample(structures, n_train+n_test)
    structures_train = ind[:n_train]
    structures_test = ind[n_train:]
    print(f'training with structures {structures_train}')
    print(f'testing with structures {structures_test}')
    features, labels = [], []
    for s in structures_train:
        print(s)
        feat, lab = load_train_data(pkl_path, s)
        features.extend(feat)
        labels.extend(lab)
    X_tr, y_tr = shuffle(features, labels)
    X_tr_m = [list(np.array(x)*mask) for x in X_tr]
    lr_cov = LogisticRegression()
    lr_cov.fit(X_tr_m, y_tr)
    X_te, y_te = [], []


    for s in structures_test:
        feat, lab, c = load_test_data(pkl_path, s)
        num_ecs = int(sum(lab))
        X_te.extend(feat)
        y_te.extend(lab)
        X_te_m = [list(np.array(x)*mask) for x in feat]
        y_pred = lr_cov.predict_proba(X_te_m)[:, 1]
        p, r, t = precision_recall_curve(lab, y_pred)
        plt.clf()
        plt.plot(r, p)
        plt.savefig(os.path.join(save_loc, name+f'_prc_tested_{s}'))
        tup = sorted(list(zip(y_pred, lab)), key = lambda x: x[0])
        y_pred_sorted = np.array([x[0] for x in tup])
        y_te_sorted = np.array([x[1] for x in tup])

        C_cum_tp = y_te_sorted.cumsum(0)
        C_cum_fp = (np.ones_like(y_te_sorted) - y_te_sorted).cumsum(0)
        precision_at = C_cum_tp / (C_cum_tp + C_cum_fp)
        plt.clf()
        plt.plot(precision_at[:num_ecs])
        plt.savefig(os.path.join(save_loc, name+f'_rankorder_tested_{s}'))

        plt.clf()
        plt.imshow(c)
        plt.savefig(os.path.join(save_loc, name+f'_trueC_{s}'))

        plt.clf()
        plt.imshow(reconstruct_c(y_pred))
        plt.savefig(os.path.join(save_loc, name+f'_predC_{s}'))

    X_te_m = [list(np.array(x)*mask) for x in X_te]
    y_pred = lr_cov.predict_proba(X_te_m)[:, 1]
    p, r, t = precision_recall_curve(y_te, y_pred)
    plt.clf()
    plt.plot(r, p)
    plt.savefig(os.path.join(save_loc, name+f'_prc_all'))
    #with open(os.path.join(save_loc, 'w')) as f:
    #    f.write('prec')
#generate_train_dataset('/home/sreeves/aln')
#dataset = load_train_data('/home/sreeves/aln')
#pickle.dump(dataset, open('/home/sreeves/GWG_release/dataset.pkl', 'wb'))
train_and_test_lr_model('/home/sreeves/aln', n_train=2, n_test=2, save_loc='./logreg_test', name='tr_2_t_2_cov_only')