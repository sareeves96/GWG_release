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
from evcouplings.compare import DistanceMap, pdb, sifts, distances
from pcd_potts import l1, matsave, norm_J
from Bio import SeqIO
import torch
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def generate_alignment(pdb_code, chain, save_loc, uniref90, th):
    fasta = os.path.join(os.path.dirname(save_loc), 'temp.fa')
    prefix = os.path.splitext(save_loc)[0]
    evchain = evcouplings.compare.pdb.PDB.from_id(pdb_code).get_chain(chain)
    seq = evchain.residues
    seq = ''.join(list(seq.dropna().sort_values('seqres_id')['one_letter_code'].values))
    print(f'extracted sequence: {seq}')
    with open(fasta, 'w') as seq_file:
        seq_file.write(f">{pdb_code}|Chain{chain}/\n{seq}")
    print('Generating alignment. This will take several minutes...')
    tools.run_jackhmmer(fasta, uniref90, prefix, use_bitscores=False, seq_threshold=th, domain_threshold=th)
    with open(prefix+'.sto', 'r') as input_handle:
        with open(save_loc, 'w') as output_handle:
            sequences = SeqIO.parse(input_handle, 'stockholm')
            count = SeqIO.write(sequences, output_handle, 'a2m')
            print(f"Sequences in converted alignment = {count}")
    print('Finished generating alignment')


def generate_contact_map_monomer(pdb_code, chain, save_loc):
    print('Generating distance map')
    res = sifts.SIFTS('./pdb_chain_uniprot.csv').by_pdb_id(pdb_code, pdb_chain=chain)
    dist_map = distances.intra_dists(res)
    dist_map.to_file(save_loc)


def load_real_protein(a2m, intra_file, cutoff, pdb_code, chain, batch_size, th,
                      uniref90='/home/sareeves96/databases/uniref90.fasta'):

    if not os.path.exists(a2m):
        print("Generating alignment using jackhmmer...")
        generate_alignment(pdb_code, chain, a2m, uniref90, th)

    print("Loading alignment...")
    with open(a2m, "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")
        aln.ids[0] = aln.ids[0] + f'/1-{len(aln[0])}'
        print(aln.ids[0])

    if not os.path.exists(intra_file):
        generate_contact_map_monomer(pdb_code, chain, intra_file)

    print("Loading distmap(s)")
    distmap_intra = DistanceMap.from_file(intra_file)

    print("Done")
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
        print("Generating weights")
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
    print(dca_int_indices)
    return train_loader, test_loader, x_oh, num_ecs, torch.tensor(J), torch.tensor(C), dca_int_indices, L


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dist_file = os.path.join(args.save_dir, args.pdb_code)
    align_file = dist_file + '.a2m'

    # load existing data
    train_loader, test_loader, data, num_ecs, ground_truth_J_norm, ground_truth_C, dm_indices, protein_L = \
        load_real_protein(
            align_file, dist_file, args.contact_cutoff, args.pdb_code, args.chain, args.batch_size, args.threshold
        )
    dim, n_out = data.size()[1:]
    ground_truth_J_norm = ground_truth_J_norm.to(device)
    matsave(ground_truth_C, "{}/ground_truth_C.png".format(args.save_dir))
    matsave(ground_truth_J_norm, "{}/ground_truth_dists.png".format(args.save_dir))

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load ckpt
    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
        optimizer.load_state_dict(d['optimizer'])
        sampler.load_state_dict(d['sampler'])


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
                buffer_inds = np.random.choice(all_inds, args.batch_size, replace=False)
                x_fake = buffer[buffer_inds].to(device)
                for k in range(args.sampling_steps):
                    x_fake = sampler.step(x_fake.detach(), model).detach()

                buffer[buffer_inds] = x_fake.detach().cpu()

                logp_real = (model(x).squeeze() * weights).mean()
                logp_fake = model(x_fake).squeeze().mean()

                obj = logp_real - logp_fake
                loss = -obj

            # add l1 reg
            loss += args.l1 * norm_J(get_J()).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                if args.sampler == "plm":
                    my_print("({}) loss = {:.4f}".format(itr, loss.item()))
                else:
                    my_print("({}) log p(real) = {:.4f}, log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}"\
                             .format(itr,logp_real.item(),logp_fake.item(),obj.item(),sampler._hops))

                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()
                inds = torch.triu_indices(ground_truth_C.size(0), ground_truth_C.size(1), 1)
                C_inds = ground_truth_C[inds[0], inds[1]]
                J_inds = norm_J(get_J_sub())[inds[0], inds[1]]
                J_inds_sorted = torch.sort(J_inds, descending=True).indices
                C_inds_sorted = C_inds[J_inds_sorted]
                recall_tests = [int(protein_L / k) for k in [10, 5, 2, 1]]
                recall_tests += [num_ecs]

                C_cum_tp = C_inds_sorted.cumsum(0)
                print(C_cum_tp)
                arange = torch.arange(C_cum_tp.size(0)) + 1
                acc_at = C_cum_tp.float() / arange.float()

                C_cum_fn = (torch.ones_like(C_inds_sorted) - C_inds_sorted).cumsum(0)
                print(C_cum_fn)
                recall_at = C_cum_tp.float() / (C_cum_tp.float() + C_cum_fn)

                my_print("\t err^2 = {:.4f}, rmse = {:.4f}, acc @ 50 = {:.4f}, acc @ 75 = {:.4f}, acc @ 100 = {:.4f}"\
                         .format(sq_err, rmse, acc_at[50], acc_at[75], acc_at[100]))
                for r in recall_tests:
                    print(f"Recall at {r}: {recall_at[r]}")
                logger.flush()


            if itr % args.viz_every == 0:
                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()

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


                matsave(get_J_sub().abs().transpose(2, 1).reshape(dm_indices.size(0) * n_out,
                                                                  dm_indices.size(0) * n_out),
                        "{}/model_J_{}_sub.png".format(args.save_dir, itr))
                matsave(norm_J(get_J_sub()), "{}/model_J_norm_{}_sub.png".format(args.save_dir, itr))

                matsave(get_J().abs().transpose(2, 1).reshape(dim * n_out, dim * n_out),
                        "{}/model_J_{}.png".format(args.save_dir, itr))
                matsave(norm_J(get_J()), "{}/model_J_norm_{}.png".format(args.save_dir, itr))

                inds = torch.triu_indices(ground_truth_C.size(0), ground_truth_C.size(1), 1)
                C_inds = ground_truth_C[inds[0], inds[1]]
                J_inds = norm_J(get_J_sub())[inds[0], inds[1]]
                J_inds_sorted = torch.sort(J_inds, descending=True).indices
                C_inds_sorted = C_inds[J_inds_sorted]
                C_cumsum = C_inds_sorted.cumsum(0)
                arange = torch.arange(C_cumsum.size(0)) + 1
                acc_at = C_cumsum.float() / arange.float()

                plt.clf()
                plt.plot(acc_at[:num_ecs].detach().cpu().numpy())
                plt.savefig("{}/acc_at_{}.png".format(args.save_dir, itr))

            if itr % args.ckpt_every == 0:
                my_print("Saving checkpoint to {}/ckpt.pt".format(args.save_dir))
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict()
                }, "{}/ckpt.pt".format(args.save_dir))


            itr += 1

            if itr > args.n_iters:
                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()
                with open("{}/sq_err.txt".format(args.save_dir), 'w') as f:
                    f.write(str(sq_err))
                with open("{}/rmse.txt".format(args.save_dir), 'w') as f:
                    f.write(str(rmse))

                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict()
                }, "{}/ckpt.pt".format(args.save_dir))

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
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--l1', type=float, default=.0)
    parser.add_argument('--contact_cutoff', type=float, default=5.)

    # data collection params
    parser.add_argument('--pdb_code', type=str, default="6RFH")
    parser.add_argument('--chain', type=str, default="A")
    parser.add_argument('--threshold', type=float, default=1)

    args = parser.parse_args()
    args.device = device
    main(args)