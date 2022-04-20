import os
import numpy as np
from evcouplings.couplings import MeanFieldDCA
from evcouplings.align import Alignment
from evcouplings.compare import sifts, distances
import evcouplings
import pickle
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from propy import PyPro
import pandas as pd
from additional_utils import aln_convert, reconstruct_c, plot_stats
import argparse


def generate_msa_features(f_name):
    print("Loading alignment...")
    with open(f_name, "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")
    try:
        dca = MeanFieldDCA(aln)
    except:
        aln.ids[0] = aln.ids[0] + f'/1-{len(aln[0])}'
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


def generate_protein_sequence_features(seq):
    DesObject = PyPro.GetProDes(seq)
    aac = list(DesObject.GetAAComp().values())
    ctd = list(DesObject.GetCTD().values())
    mbauto = list(DesObject.GetMoranAuto().values())
    prot_feats = aac + ctd + mbauto
    return prot_feats


def generate_contact_map_monomer(f_name, cutoff=5.):
    print('Generating distance map')
    code = os.path.split(f_name)[-1]
    pdb_code, pdb_chain = code[:4], code[4]
    print(pdb_code, pdb_chain)
    try:
        res = sifts.SIFTS('./pdb_chain_uniprot.csv').by_pdb_id(pdb_code, pdb_chain=pdb_chain)
        dist_map = distances.intra_dists(res)
    except ValueError:
        pdb_chain = 'A'
        res = evcouplings.compare.PDB.from_id(pdb_code).get_chain(pdb_chain)
        dist_map = evcouplings.compare.distances.DistanceMap.from_coords(res)

    #dist_map.to_file(f_name.replace(f'aln/{code}', f'map/{code}'))

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


def parse_alignment_into_dataset(file):
    print('This step involves involves clustering all sequences in the alignment and can take several minutes')
    cov, mi, di, fn, cn = generate_msa_features(file)
    cm = generate_contact_map_monomer(file)
    assert mi.shape == cm.shape
    idx1, idx2 = np.triu_indices(cm.shape[0], 1)
    step = cm.shape[0]
    features = []
    labels = []
    for idx in list(zip(idx1, idx2)):
        label = cm[idx]
        labels.append(label)
        ax1 = slice(idx[0], None, step)
        ax2 = slice(idx[1], None, step)
        covs = cov[ax1, ax2].flatten()
        mutual_info = mi[idx]
        direct_info = di[idx]
        frob = fn[idx]
        corr = cn[idx]
        feature = covs
        for num in [mutual_info, direct_info, frob, corr]:
            feature = np.append(feature, num)
        features.append(feature)
    return features, labels


def generate_train_dataset(train_data_directory='./aln'):
    missed = []
    for f in os.listdir(train_data_directory):
        file = os.path.join(train_data_directory, f)
        print(os.path.splitext(file))
        print(os.path.splitext(file)[0]+'.txt')
        if os.path.exists(os.path.splitext(file)[0]+'.pkl'):
            print(f'Skipping {file}')
            continue
        if os.path.splitext(file)[-1] == '.aln' and not os.path.exists(os.path.splitext(file)[0]+'.txt'):
            aln_convert(file)
            file = os.path.splitext(file)[0]+'.txt'
        elif os.path.splitext(file)[-1] != '.txt':
            print(f"Skipped {file}")
            continue
        try:
            features, labels = parse_alignment_into_dataset(file)
        except:
            print(f'Couldn\'t load data for file {file}')
            missed.append(file)
            continue
        pickle.dump(zip(features, labels), open(os.path.splitext(file)[0]+'.pkl', 'wb'))

    print(missed)


def load_and_balance_train_data(path, structure):
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
    with open(file.replace('.pkl', '.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                seq = line.strip()
                break
    prot_feats = generate_protein_sequence_features(seq)
    features_shuffled = [list(f) + prot_feats for f in features_shuffled]

    return features_shuffled, labels_shuffled


def load_test_data(path, structure):
    file = os.path.join(path, structure+'.pkl')
    with open(file, 'rb') as curr:
        features, labels = zip(*pickle.load(curr))
    c = reconstruct_c(labels)
    with open(file.replace('.pkl', '.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                seq = line.strip()
                break
    prot_feats = generate_protein_sequence_features(seq)
    features = [list(f) + prot_feats for f in features]
    return features, labels, c


def train_and_test_lr_model(pkl_path, n_train, n_test, save_loc, name,  precomputed=False):
    os.makedirs(save_loc, exist_ok=True)
    if not precomputed:
        print('Choosing structures randomly')
        structures = [os.path.splitext(s)[0] for s in os.listdir(pkl_path) if '.pkl' in s]
        ind = random.sample(structures, n_train+n_test)
        structures_train = ind[:n_train]
        structures_test = ind[n_train:]
        features, labels = [], []
        for i, s in enumerate(structures_train):
            print(i, s)
            feat, lab = load_and_balance_train_data(pkl_path, s)
            features.extend(feat)
            labels.extend(lab)
        pickle.dump(zip(structures_train, features, labels),
                    open(os.path.join(save_loc, 'precomputed_training_data.pkl'), 'wb'))
        print('Finished loading training data')
    else:
        structures_train, features, labels = zip(*pickle.load(
            open(os.path.join(save_loc, 'precomputed_training_data.pkl'), 'rb')))
        structures = [os.path.splitext(s)[0] for s in os.listdir(pkl_path) if '.pkl' in s]
        for s in structures_train:
            structures.remove(s)
        # choose remaining structures to visualize randomly
        structures_test = random.sample(structures, n_test)
        print("Loaded data from precomputed pickle object")
    print(f'training with structures {structures_train}')
    print(f'testing with structures {structures_test}')
    X_tr_m, y_tr = shuffle(features, labels)
    #X_tr_m = [list(np.array(x)*mask) for x in X_tr]
    scaler = MinMaxScaler()
    scaler.fit(X_tr_m)
    X_tr_scaled = scaler.transform(X_tr_m)
    model = LogisticRegression(penalty='l1', solver='liblinear')
    param_grid = {'C': [10**4, 10**2, 1]}
    grid = GridSearchCV(model, param_grid=param_grid, verbose=3, n_jobs=-1, refit=True, cv=3)
    grid.fit(X_tr_scaled, y_tr)
    lr_cov = grid.best_estimator_
    pickle.dump(lr_cov, open(os.path.join(save_loc, 'optimized_model.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(save_loc, 'scaler.pkl'), 'wb'))
    pd.DataFrame(grid.cv_results_).to_csv(os.path.join(save_loc, 'cv_results.csv'))

    for s in structures_test:
        print(s)
        feat, lab, c = load_test_data(pkl_path, s)
        num_ecs = int(sum(lab))
        X_te_m = feat
        #X_te_m = [list(np.array(x)*mask) for x in feat]
        X_te_scaled = scaler.transform(X_te_m)
        y_pred = lr_cov.predict_proba(X_te_scaled)[:, 1]
        p, r, t = precision_recall_curve(lab, y_pred)
        plt.plot(r, p, label=s)
        plt.legend()
        plt.savefig(os.path.join(save_loc, name+f'_prc_tested_{s}'))
        tup = sorted(list(zip(y_pred, lab)), key=lambda x: x[0], reverse=True)
        y_te_sorted = np.array([x[1] for x in tup])

        C_cum_tp = y_te_sorted.cumsum(0)
        C_cum_fp = (np.ones_like(y_te_sorted) - y_te_sorted).cumsum(0)
        precision_at = C_cum_tp / (C_cum_tp + C_cum_fp)
        plt.clf()
        plt.plot(precision_at[:num_ecs])
        plt.savefig(os.path.join(save_loc, name+f'_rankorder_tested_{s}'))

        plt.clf()
        fig, axs = plt.subplots(2)
        axs[0].imshow(reconstruct_c(y_pred))
        axs[1].imshow(c)
        plt.savefig(os.path.join(save_loc, name+f'_pred_true_C_{s}'))
        plt.close('all')

    return structures_train, structures_test


def test_on_example_prots(directories, model_opt='./lr_100_l2/optimized_model.pkl'):
    lr_cov = pickle.load(open(model_opt, 'rb'))
    scaler = pickle.load(open(model_opt.replace('optimized_model', 'scaler'), 'rb'))
    for dir in directories:
        pdb = dir[:4]
        file = os.path.join('.', dir, f'{pdb}.a2m')
        try:
            features, lab = parse_alignment_into_dataset(file)
        except:
            file = os.path.join('.', dir, f'{pdb.lower()}.a2m')
            features, lab = parse_alignment_into_dataset(file)
        seq = ''.join(evcouplings.compare.PDB.from_id(pdb).get_chain('A').residues.one_letter_code.values)
        print(seq)
        prot_feats = generate_protein_sequence_features(seq)
        features = [list(f) + prot_feats for f in features]
        num_ecs = int(sum(lab))
        X_te_scaled = scaler.transform(features)
        y_pred = lr_cov.predict_proba(X_te_scaled)[:, 1]
        plot_stats(y_pred, lab, num_ecs, pdb)


def main(args):
    train_and_test_lr_model(args.aln_folder_location, n_train=args.n_train, n_test=args.n_test, save_loc=args.save_loc,
                            name=args.name, precomputed=args.precomputed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default="./logreg_test")
    parser.add_argument('--aln_folder_location', default='./aln')
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--precomputed', action="store_true")

    args = parser.parse_args()
    main(args)

