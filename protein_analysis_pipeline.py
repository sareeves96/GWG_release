import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rbm
import samplers
import os
import evcouplings
from evcouplings.couplings import MeanFieldDCA
from evcouplings.align import Alignment, tools, map_matrix
from evcouplings.compare import DistanceMap, sifts, distances
from pcd_potts import matsave, norm_J
from eval_protein import top_k_mat
from Bio import SeqIO
import torch
import datetime
from sklearn import metrics
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def generate_alignment(pdb_code, chain, save_loc, uniref90, th):
    '''Uses hidden markov models to search the local uniref90 database of evolutionarily related sequences'''
    fasta = os.path.join(os.path.dirname(save_loc), 'temp.fa')
    prefix = os.path.splitext(save_loc)[0]
    evchain = evcouplings.compare.pdb.PDB.from_id(pdb_code).get_chain(chain)
    seq = ''.join(list(evchain.residues.dropna().one_letter_code))
    print(f'extracted sequence: {seq}')
    with open(fasta, 'w') as seq_file:
        seq_file.write(f">{pdb_code}|Chain{chain}\n{seq}")  #/1-{len(seq)}
    print('Generating alignment. This will take several minutes...')
    tools.run_jackhmmer(fasta, uniref90, prefix, use_bitscores=True, seq_threshold=th, domain_threshold=th)
    with open(prefix+'.sto', 'r') as input_handle:
        with open(save_loc, 'w') as output_handle:
            sequences = SeqIO.parse(input_handle, 'stockholm')
            count = SeqIO.write(sequences, output_handle, 'fasta')
            print(f"Sequences in converted alignment = {count}")
    print('Finished generating alignment')
    print(datetime.datetime.now())


def generate_contact_map_monomer(pdb_code, chain, save_loc):
    '''Uses the provided pdb code to extract contact map from the Protein DataBank'''
    print('Generating distance map')
    try:
        res = sifts.SIFTS('./pdb_chain_uniprot.csv').by_pdb_id(pdb_code, pdb_chain=chain)
        dist_map = distances.intra_dists(res)
    except:
        res = evcouplings.compare.PDB.from_id(pdb_code).get_chain(chain)
        dist_map = evcouplings.compare.distances.DistanceMap.from_coords(res)
    dist_map.to_file(save_loc)


def load_real_protein(a2m, intra_file, cutoff, pdb_code, chain, batch_size, th, overwrite,
                      uniref90='/home/sareeves96/databases/uniref90.fasta'):
    '''This function is an adaptation of load_real_protein from utils.py. My innovation is to make it work
    with alignments generated using JackHMMER and load experimental distance maps from the PDB'''

    if not os.path.exists(a2m) or overwrite:
        print("Generating alignment using jackhmmer...")
        generate_alignment(pdb_code, chain, a2m, uniref90, th)

    print("Loading alignment...")
    with open(a2m, "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")
        aln.ids[0] = aln.ids[0] + f'/1-{len(aln[0])}'
        print(aln.ids[0])

    if not os.path.exists(intra_file) or overwrite:
        generate_contact_map_monomer(pdb_code, chain, intra_file)

    print("Loading distmap(s)")
    distmap_intra = DistanceMap.from_file(intra_file)

    L = aln.L
    D = len(aln.alphabet)
    print("Raw Data size {}".format((L, D)))

    dca = MeanFieldDCA(aln)
    L = dca.alignment.L
    D = len(dca.alignment.alphabet)
    x_int = torch.from_numpy(map_matrix(dca.alignment.matrix, dca.alignment.alphabet_map)).float()
    x_oh = torch.nn.functional.one_hot(x_int.long(), D).float()
    print("Filtered Data size {}".format((L, D)))

    J = -distmap_intra.dist_matrix
    print(J.shape)
    J = J + cutoff
    J[J < 0] = 0.
    J[np.isnan(J)] = 0.  # treat unobserved values as just having no contact
    ind = np.diag_indices(J.shape[0])
    J[ind] = 0.
    C = np.copy(J)
    C[C > 0] = 1.
    C[C <= 0] = 0.
    print("J size = {}".format(J.shape))

    weight_file = os.path.join(os.path.dirname(a2m), "weights.pkl")
    if not os.path.exists(weight_file):
        print("Generating weights... this may take a few minutes")
        dca.alignment.set_weights()
        weights = dca.alignment.weights
        with open(weight_file, 'wb') as f:
            pickle.dump(weights, f)
    else:
        print("Loading weights")
        with open(weight_file, 'rb') as f:
            weights = pickle.load(f)

    weights = torch.tensor(weights).float()
    print("Dataset has {} examples, sum weights are {}".format(weights.size(0), weights.sum()))
    print("Scaling up by {}".format(float(weights.size(0)) / weights.sum()))
    weights *= float(weights.size(0)) / weights.sum()
    print("Distmap size {}".format(J.shape))

    train_data = TensorDataset(x_oh, weights)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    num_ecs = int(C.sum() / 2)
    print(f"contacts: {num_ecs}")

    dca_int_indices = range(len(J[0]))
    dca_int_indices = torch.tensor(dca_int_indices).long()

    return train_loader, test_loader, x_oh, num_ecs, torch.tensor(J), torch.tensor(C), dca_int_indices, L


def main(args):
    '''This function is an adaptation of the code in pcd_potts.py. It is mostly not original code'''
    os.makedirs(args.save_dir, exist_ok=True)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    my_print(datetime.datetime.now())
    my_print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dist_file = os.path.join(args.save_dir, args.pdb_code)
    align_file = dist_file + '.a2m'

    # load existing data
    train_loader, test_loader, data, num_ecs, ground_truth_J_norm, ground_truth_C, dm_indices, protein_L = \
        load_real_protein(
            align_file,
            dist_file,
            args.contact_cutoff,
            args.pdb_code,
            args.chain,
            args.batch_size,
            args.threshold,
            args.overwrite,
            args.uniref90
        )
    dim, n_out = data.size()[1:]
    ground_truth_J_norm = ground_truth_J_norm.to(device)
    matsave(ground_truth_C, "{}/ground_truth_C.png".format(args.save_dir))
    matsave(ground_truth_J_norm, "{}/ground_truth_dists.png".format(args.save_dir))
    np.save("{}/ground_truth_C".format(args.save_dir), ground_truth_C.detach().cpu().numpy())
    np.save("{}/ground_truth_dists".format(args.save_dir), ground_truth_J_norm.detach().cpu().numpy())

    model = rbm.DensePottsModel(dim, n_out, learn_J=True, learn_bias=True)
    buffer = model.init_sample(args.buffer_size)

    model.to(device)

    # make G symmetric
    def get_J():
        j = model.J
        jt = j.transpose(0, 1).transpose(2, 3)
        return (j + jt) / 2

    def get_J_sub():
        j = get_J()
        j_sub = j[dm_indices, :][:, dm_indices]
        return j_sub

    if args.sampler == "gwg":
        sampler = samplers.DiffSamplerMultiDim(dim, 1, approx=True, temp=2.)
    elif args.sampler in {'gibbs', 'plm'}:
        sampler = samplers.PerDimMetropolisSampler(dim, int(n_out), rand=False)
    else:
        raise ValueError

    my_print(device)
    my_print(model)
    my_print(buffer.size())
    my_print(sampler)
    my_print(datetime.datetime.now())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load ckpt
    #if args.ckpt_path is not None and not args.overwrite:
    #    d = torch.load(args.ckpt_path)
    #    model.load_state_dict(d['model'])
    #    optimizer.load_state_dict(d['optimizer'])
    #    sampler.load_state_dict(d['sampler'])

    # mask matrix for PLM
    L, D = model.J.size(0), model.J.size(2)
    num_node = L * D
    J_mask = torch.ones((num_node, num_node)).to(device)
    for i in range(L):
        J_mask[D * i:D * i + D, D * i:D * i + D] = 0

    itr = 0
    sq_errs = []
    rmses = []
    all_inds = list(range(args.buffer_size))
    while itr < args.n_iters:
        for x in train_loader:
            weights = x[1].to(device)
            if args.unweighted:
                weights = torch.ones_like(weights)
            x = x[0].to(device)

            if args.sampler == "plm":
                plm_J = model.J.transpose(2, 1).reshape(dim * n_out, dim * n_out)
                logits = torch.matmul(x.view(x.size(0), -1), plm_J * J_mask) + model.bias.view(-1)[None]
                x_inds = (torch.arange(x.size(-1))[None, None].to(x.device) * x).sum(-1)
                cross_entropy = nn.functional.cross_entropy(
                    input=logits.reshape((-1, D)),
                    target=x_inds.view(-1).long(),
                    reduce=False)
                cross_entropy = torch.sum(cross_entropy.reshape((-1, L)), -1)
                loss = (cross_entropy * weights).mean()

            else:
                # sample a batch of model-derived sequences from the persistent buffer
                buffer_inds = np.random.choice(all_inds, args.batch_size, replace=False)
                x_fake = buffer[buffer_inds].to(device)
                # perform k steps of GWG, such that the final x_fakes are each in Hamming distance k of the originals
                for k in range(args.sampling_steps):
                    x_fake = sampler.step(x_fake.detach(), model).detach()
                # update the buffer with new samples from the model
                buffer[buffer_inds] = x_fake.detach().cpu()
                # compute the log probability of real examples in the batch
                logp_real = (model(x).squeeze() * weights).mean()
                # compute the log probability of samples generated by the model
                logp_fake = model(x_fake).squeeze().mean()
                # the objective is to adjust the model parameters to maximize the probability of the training data
                # while minimizing the likelihood of "fake" data, obtained by sampling the model distribution
                obj = logp_real - logp_fake
                loss = -obj

            # add l1 reg
            loss += args.l1 * norm_J(get_J()).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            def save_prc(f_name):
                y_true = ground_truth_C.detach().cpu().numpy().flatten().astype(int)
                y_pred = norm_J(get_J_sub()).detach().cpu().numpy().flatten()
                print(max(y_pred))
                print(min(y_pred))
                p, r, t = metrics.precision_recall_curve(y_true, y_pred)
                plt.clf()
                plt.plot(r, p)
                plt.savefig(f_name)

            def get_rank_order_stats():

                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()
                inds = torch.triu_indices(ground_truth_C.size(0), ground_truth_C.size(1), 1)
                C_inds = ground_truth_C[inds[0], inds[1]]
                J_inds = norm_J(get_J_sub())[inds[0], inds[1]]
                J_inds_sorted = torch.sort(J_inds, descending=True).indices
                C_inds_sorted = C_inds[J_inds_sorted]

                C_cum_tp = C_inds_sorted.cumsum(0)
                C_cum_fp = (torch.ones_like(C_inds_sorted) - C_inds_sorted).cumsum(0)
                # print(C_cum_tp)
                arange = torch.arange(C_cum_tp.size(0)) + 1
                acc_at = C_cum_tp.float() / arange.float()

                precision_at = C_cum_tp.float() / (C_cum_tp.float() + C_cum_fp)

                return sq_err, rmse, acc_at, precision_at

            if itr % args.print_every == 0:
                if args.sampler == "plm":
                    my_print("({}) loss = {:.4f}".format(itr, loss.item()))
                else:
                    my_print("({}) log p(real) = {:.4f}, log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}"\
                             .format(itr,logp_real.item(),logp_fake.item(),obj.item(),sampler._hops))

                sq_err, rmse, acc_at, precision_at = get_rank_order_stats()

                my_print("\t err^2 = {:.4f}, rmse = {:.4f}, acc @ 50 = {:.4f}, acc @ 75 = {:.4f}, acc @ 100 = {:.4f}"\
                         .format(sq_err, rmse, acc_at[50], acc_at[75], acc_at[100]))
                precision_tests = [int(protein_L / k) for k in [10, 5, 2, 1]]
                precision_tests += [num_ecs]
                for r in precision_tests:
                    my_print(f"precision at {r}: {precision_at[r]}")
                logger.flush()

            if itr % args.viz_every == 0:

                sq_err, rmse, acc_at, precision_at = get_rank_order_stats()

                mode = 'a' if itr != 0 else 'w'
                with open("{}/sq_err_int.txt".format(args.save_dir), mode) as f:
                    f.write(str(sq_err.detach().cpu().numpy()))
                with open("{}/rmse_int.txt".format(args.save_dir), mode) as f:
                    f.write(str(rmse.detach().cpu().numpy()))
                with open("{}/acc_at_int.txt".format(args.save_dir), mode) as f:
                    f.write(' '.join([str(round(j, 3)) for j in acc_at.detach().cpu().numpy()]))
                with open("{}/precision_at_int.txt".format(args.save_dir), mode) as f:
                    f.write(' '.join([str(round(j, 3)) for j in precision_at.detach().cpu().numpy()]))

                sq_errs.append(sq_err.item())
                plt.clf()
                plt.plot(sq_errs, label="sq_err")
                plt.legend()
                plt.savefig("{}/sq_err.png".format(args.save_dir))

                rmses.append(rmse.item())
                plt.clf()
                plt.plot(rmses, label="rmse")
                plt.legend()
                plt.savefig("{}/rmse.png".format(args.save_dir))

                np.save("{}/model_J_norm_{}.png".format(args.save_dir, itr), norm_J(get_J()).detach().cpu().numpy())
                matsave(get_J().abs().transpose(2, 1).reshape(dim * n_out, dim * n_out),
                        "{}/model_J_{}.png".format(args.save_dir, itr))
                matsave(norm_J(get_J()), "{}/model_J_norm_{}.png".format(args.save_dir, itr))

                plt.clf()
                plt.plot(acc_at[:num_ecs].detach().cpu().numpy())
                plt.savefig("{}/acc_at_{}.png".format(args.save_dir, itr))
                plt.clf()
                plt.plot(precision_at[:num_ecs].detach().cpu().numpy())
                plt.savefig("{}/precision_at_{}.png".format(args.save_dir, itr))
                save_prc("{}/prc_at_{}.png".format(args.save_dir, itr))

            if itr % args.ckpt_every == 0:
                my_print("Saving checkpoint to {}/ckpt.pt".format(args.save_dir))
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict()
                }, "{}/ckpt.pt".format(args.save_dir))


            itr += 1

            if itr > args.n_iters:
                sq_err, rmse, acc_at, precision_at = get_rank_order_stats()

                mode = 'w'
                with open("{}/sq_err_int.txt".format(args.save_dir), mode) as f:
                    f.write(str(sq_err.detach().cpu().numpy()))
                with open("{}/rmse_int.txt".format(args.save_dir), mode) as f:
                    f.write(str(rmse.detach().cpu().numpy()))
                with open("{}/acc_at_int.txt".format(args.save_dir), mode) as f:
                    f.write(' '.join([str(round(j, 3)) for j in acc_at.detach().cpu().numpy()]))
                with open("{}/precision_at_int.txt".format(args.save_dir), mode) as f:
                    f.write(' '.join([str(round(j, 3)) for j in precision_at.detach().cpu().numpy()]))

                norms = norm_J(get_J())
                norms_top_l = top_k_mat(norms, norms.size(0))
                matsave(norms_top_l, "{}/J_norm_top_{}.png".format(args.save_dir, norms.size(0)))
                norms_top_l = top_k_mat(norms, 2 * norms.size(0))
                matsave(norms_top_l, "{}/J_norm_top_{}.png".format(args.save_dir, 2 * norms.size(0)))
                norms_top_l = top_k_mat(norms, 4 * norms.size(0))
                matsave(norms_top_l, "{}/J_norm_top_{}.png".format(args.save_dir, 4 * norms.size(0)))

                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict()
                }, "{}/ckpt.pt".format(args.save_dir))

                my_print(datetime.datetime.now())
                quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--ckpt_path', type=str, default=None)

    # mcmc
    parser.add_argument('--sampler', type=str, default='gwg')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--approx', action="store_true")
    parser.add_argument('--unweighted', action="store_true")
    parser.add_argument('--sampling_steps', type=int, default=50)
    parser.add_argument('--buffer_size', type=int, default=2560)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--ckpt_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--l1', type=float, default=.01)
    parser.add_argument('--contact_cutoff', type=float, default=5.)

    # data collection params
    parser.add_argument('--pdb_code', type=str, default="6RFH")
    parser.add_argument('--chain', type=str, default="A")
    parser.add_argument('--threshold', type=float, default=100) ##currently bitscores!!
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--uniref90', type=str, default="./databases/uniref90.fasta")

    args = parser.parse_args()
    args.device = device
    main(args)
