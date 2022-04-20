import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def generate_rankorder_precision(experiments=['1HZXOPSD_gwg', '1HZXOPSD_plm'], ident=1, lim=None, folder='.'):
    prefix = '' if ident == 1 else 'threshold '
    df = pd.DataFrame(columns=[prefix+e.split('_')[ident] for e in experiments])
    name = experiments[0].split('_')[0]
    contacts = 0
    for e in experiments:
        print(e)
        path = os.path.join(folder, e, 'precision_at_int.txt')
        contacts = int(np.load(os.path.join(folder, e, 'ground_truth_C.npy')).sum()/2)
        contacts = contacts if lim is None else lim
        with open(path, 'r') as f:
            s = f.read()
            pcs = [float(i) for i in s.split(' ')]
            df[prefix+e.split('_')[ident]] = pcs[:contacts]
    df.plot()
    plt.title(f"Rank-order precision curve for {name}")
    plt.xlabel('Prediction rank')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(folder, f'{name}_rank_order_lim_{contacts}.png'))
    return df


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


def plot_stats(y_pred, lab, num_ecs, pdb):
    p, r, t = precision_recall_curve(lab, y_pred)
    plt.clf()
    plt.plot(r, p)
    plt.savefig(f'_prc_tested_{pdb}')
    tup = sorted(list(zip(y_pred, lab)), key=lambda x: x[0], reverse=True)
    y_te_sorted = np.array([x[1] for x in tup])

    C_cum_tp = y_te_sorted.cumsum(0)
    C_cum_fp = (np.ones_like(y_te_sorted) - y_te_sorted).cumsum(0)
    precision_at = C_cum_tp / (C_cum_tp + C_cum_fp)
    plt.clf()
    plt.plot(precision_at[:num_ecs])
    plt.savefig(f'_rankorder_tested_{pdb}')

    plt.clf()
    plt.imshow(reconstruct_c(y_pred))
    plt.savefig(f'_pred_true_C_{pdb}')
    plt.close('all')
