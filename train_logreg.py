import os
import numpy as np
from evcouplings.couplings import MeanFieldDCA, MeanFieldCouplingsModel
from evcouplings.align import Alignment, tools, map_matrix
from evcouplings.compare import DistanceMap, sifts, distances
import pickle


def aln_convert(f_name):
    with open(f_name, 'r') as f_org:
        with open(os.path.splitext(f_name)[0]+'.txt', 'w') as f_new:
            #f_new.write('CLUSTAL W (1.82) multiple sequence alignment\n')
            for i, line in enumerate(f_org.readlines()):
                f_new.write(f'>seq{i}|/{"1-"+str(len(line)-1)}\n{line}')


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
            label = cm[idx]
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
                np.append(feature ,num)
            features.append(feature)
        pickle.dump(zip(features, labels), open(os.path.splitext(file)[0]+'.pkl', 'wb'))

    print(missed)

generate_train_dataset('/home/sreeves/aln')

