import random
import torch
from torch import optim
import torch.nn as nn
from get_data_cross_network import load_pyg_data, target_split
from model9 import InnerProductDecoder, EncoderS, Cycler, DD, Style, EncoderP, Attention
from utils import for_gae, compute_accuracy_teacher_mask, compute_accuracy_teacher, loss_function, set_random_seeds, entropy_loss_f, DiffLoss
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
import itertools
from sklearn.metrics import f1_score
import numpy as np

set_random_seeds(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_graphs = ['acmv9', 'citationv1', 'dblpv7']
target_graphs = ['acmv9', 'citationv1', 'dblpv7']
for i in range(3):
    for j in range(3):
        if i == j:
            pass
        else:
            source_graph = source_graphs[i]
            target_graph = target_graphs[j]
            if source_graph != target_graphs:
                print("S -----> T: {}------->{}".format(source_graph, target_graph))
                data_s, a_ppmi_s = load_pyg_data('data/{}.mat'.format(source_graph), label_rate=0.5)
                data_t, a_ppmi_t = load_pyg_data('data/{}.mat'.format(target_graph), label_rate=0.5)

                data_s.adj = sp.coo_matrix(to_dense_adj(data_s.edge_index).squeeze(0))
                data_t.adj = sp.coo_matrix(to_dense_adj(data_t.edge_index).squeeze(0))

                data_s = data_s.to(device)
                data_t = data_t.to(device)
                data_t = target_split(data_t, device)
                a_ppmi_s = a_ppmi_s.to(device)
                a_ppmi_t = a_ppmi_t.to(device)

                adj_label_s, norm_s, pos_weight_s = for_gae(data_s.x, data_s.adj, device)
                adj_label_t, norm_t, pos_weight_t = for_gae(data_t.x, data_t.adj, device)

                share_model = EncoderS(data_s.x.shape[1], 512, 64, 0.5).to(device)
                private_model_s = EncoderP(data_s.x.shape[1], 512, 64, 0.5).to(device)
                private_model_t = EncoderP(data_t.x.shape[1], 512, 64, 0.5).to(device)
                attention_model = Attention(64).to(device)
                attention_model_s_l = Attention(512).to(device)
                attention_model_t_l = Attention(512).to(device)
                attention_model_s_h = Attention(64).to(device)
                attention_model_t_h = Attention(64).to(device)
                decoder_s = InnerProductDecoder(0.5, act=lambda x: x).to(device)
                decoder_t = InnerProductDecoder(0.5, act=lambda x: x).to(device)
                cycle_s = Cycler(data_s.num_nodes, 64, 256, data_t.num_nodes, 0.5).to(device)
                cycle_t = Cycler(data_t.num_nodes, 64, 256, data_s.num_nodes, 0.5).to(device)
                discriminator_d = DD(64, 16, 0.5).to(device)
                style_s_l = Style(512, 0.01, 0.4).to(device)
                style_s_h = Style(64, 0.01, 0.4).to(device)
                style_t_l = Style(512, 0.01, 0.4).to(device)
                style_t_h = Style(64, 0.01, 0.4).to(device)
                cls_model = nn.Sequential(
                    nn.Linear(64, 5),
                ).to(device)

                optimizer = optim.Adam(itertools.chain(share_model.parameters(), private_model_s.parameters(), private_model_t.parameters(),
                                                       attention_model.parameters(), attention_model_s_l.parameters(), attention_model_s_h.parameters(),
                                                       attention_model_t_l.parameters(), attention_model_t_h.parameters(),
                                                       decoder_s.parameters(), decoder_t.parameters(), style_s_l.parameters(), style_t_l.parameters(),
                                                       style_s_h.parameters(), style_t_h.parameters(), cycle_s.parameters(), cycle_t.parameters(),
                                                       discriminator_d.parameters(), cls_model.parameters()), lr=5e-3, weight_decay=5e-4)

                cls_loss = nn.CrossEntropyLoss().to(device)
                domain_loss = nn.CrossEntropyLoss()
                cycle_loss = nn.L1Loss()
                id_loss = DiffLoss()

                best_acc = 0
                best_maf = 0
                best_mif = 0
                for epoch in range(1000):
                    rate = min((epoch+1) / 1000, 0.05)

                    share_model.train()
                    private_model_s.train()
                    private_model_t.train()
                    attention_model.train()
                    attention_model_s_l.train()
                    attention_model_s_h.train()
                    attention_model_t_l.train()
                    attention_model_t_h.train()
                    decoder_s.train()
                    decoder_t.train()
                    style_s_l.train()
                    style_t_h.train()
                    style_s_l.train()
                    style_t_h.train()
                    discriminator_d.train()
                    cls_model.train()

                    optimizer.zero_grad(set_to_none=True)

                    share_z_s_l, share_z_s, mu_s, logvar_s, share_z_s_l_pm, share_z_s_pm, mu_s_pm, logvar_s_pm = share_model(data_s.x, data_s.edge_index, a_ppmi_s)
                    share_z_t_l, share_z_t, mu_t, logvar_t, share_z_t_l_pm, share_z_t_pm, mu_t_pm, logvar_t_pm = share_model(data_t.x, data_t.edge_index, a_ppmi_t)

                    private_z_s_l, private_z_s, private_mu_s, private_logvar_s, private_z_s_l_pm, private_z_s_pm, private_mu_s_pm, private_logvar_s_pm = private_model_s(data_s.x, data_s.edge_index, a_ppmi_s)
                    private_z_t_l, private_z_t, private_mu_t, private_logvar_t, private_z_t_l_pm, private_z_t_pm, private_mu_t_pm, private_logvar_t_pm = private_model_t(data_t.x, data_t.edge_index, a_ppmi_t)

                    encoded_s = attention_model([mu_s, mu_s_pm])
                    encoded_t = attention_model([mu_t, mu_t_pm])

                    s_style_l = style_s_l(attention_model_s_l([private_z_s_l, private_z_s_l_pm]))
                    s_style_h = style_s_h(attention_model_s_h([private_mu_s, private_mu_s_pm]))
                    t_style_l = style_t_l(attention_model_t_l([private_z_t_l, private_z_t_l_pm]))
                    t_style_h = style_t_h(attention_model_t_h([private_mu_t, private_mu_t_pm]))
                    generation_z_t = cycle_s(encoded_s, data_t.edge_index, t_style_l, t_style_h)
                    recover_z_s = cycle_t(generation_z_t, data_s.edge_index, s_style_l, s_style_h)
                    generation_z_s = cycle_t(encoded_t, data_s.edge_index, s_style_l, s_style_h)
                    recover_z_t = cycle_s(generation_z_s, data_t.edge_index, t_style_l, t_style_h)

                    cycle_loss_s = cycle_loss(recover_z_s, encoded_s)
                    cycle_loss_t = cycle_loss(recover_z_t, encoded_t)
                    cycle_losses = cycle_loss_s + cycle_loss_t

                    diff_loss_s = id_loss(private_mu_s, mu_s)
                    diff_loss_t = id_loss(private_mu_t, mu_t)
                    diff_loss_s_r = id_loss(private_mu_s, recover_z_s)
                    diff_loss_t_r = id_loss(private_mu_t, recover_z_t)
                    diff_loss = diff_loss_s + diff_loss_t + diff_loss_s_r + diff_loss_t_r

                    embedding_s = torch.cat((attention_model_s_h([private_z_s, private_z_s_pm]), attention_model_s_h([share_z_s, share_z_s_pm])), 1)
                    embedding_t = torch.cat((attention_model_t_h([private_z_t, private_z_t_pm]), attention_model_t_h([share_z_t, share_z_t_pm])), 1)
                    recover_s = decoder_s(embedding_s)
                    recover_t = decoder_t(embedding_t)
                    s_mu = torch.cat((private_mu_s, private_mu_s_pm, mu_s, mu_s_pm), 1)
                    s_logvar = torch.cat((private_logvar_s, private_logvar_s_pm, logvar_s, logvar_s_pm),1)
                    t_mu = torch.cat((private_mu_t, private_mu_t_pm, mu_t, mu_t_pm), 1)
                    t_logvar = torch.cat((private_logvar_t, private_logvar_t_pm, logvar_t, logvar_t_pm), 1)

                    recover_loss_s = loss_function(recover_s, adj_label_s, s_mu, s_logvar, data_s.num_nodes, norm_s, pos_weight_s)
                    recover_loss_t = loss_function(recover_t, adj_label_t, t_mu, t_logvar, data_t.num_nodes*2, norm_t, pos_weight_t)
                    recover_loss = recover_loss_s + recover_loss_t

                    recover_adj_s = decoder_s(torch.cat((attention_model_s_h([private_z_s, private_z_s_pm]), recover_z_s), 1))
                    recover_adj_t = decoder_t(torch.cat((attention_model_t_h([private_z_t, private_z_t_pm]), recover_z_t), 1))
                    cycle_recover_loss_s_r = loss_function(recover_adj_s, adj_label_s, s_mu, s_logvar, data_s.num_nodes, norm_s, pos_weight_s)
                    cycle_recover_loss_t_r = loss_function(recover_adj_t, adj_label_t, t_mu, t_logvar, data_t.num_nodes*2, norm_t, pos_weight_t)
                    cycle_recover_loss = cycle_recover_loss_s_r + cycle_recover_loss_t_r

                    domain_output_s = discriminator_d(encoded_s, data_s.edge_index, rate)
                    domain_output_s_g = discriminator_d(generation_z_s, data_s.edge_index, rate)
                    domain_output_s_r = discriminator_d(recover_z_s, data_s.edge_index, rate)
                    domain_output_t = discriminator_d(encoded_t, data_t.edge_index, rate)
                    domain_output_t_g = discriminator_d(generation_z_t, data_t.edge_index, rate)
                    domain_output_t_r = discriminator_d(recover_z_t, data_t.edge_index, rate)
                    err_s_domain = domain_loss(domain_output_s, torch.zeros(domain_output_s.size(0)).type(torch.LongTensor).to(device)) + \
                                   domain_loss(domain_output_s_r, torch.zeros(domain_output_s_r.shape[0]).type(torch.LongTensor).to(device)) + \
                                   domain_loss(domain_output_s_g, torch.zeros(domain_output_s_g.shape[0]).type(torch.LongTensor).to(device))
                    err_t_domain = domain_loss(domain_output_t, torch.ones(domain_output_t.size(0)).type(torch.LongTensor).to(device)) + \
                                   domain_loss(domain_output_t_r, torch.ones(domain_output_t_r.size(0)).type(torch.LongTensor).to(device)) + \
                                   domain_loss(domain_output_t_g, torch.ones(domain_output_t_g.size(0)).type(torch.LongTensor).to(device))

                    grl_loss = err_s_domain + err_t_domain

                    logit_s = cls_model(encoded_s)
                    cls_loss_s_o = cls_loss(logit_s, data_s.y)
                    logit_t = cls_model(encoded_t)
                    cls_loss_t_o = cls_loss(logit_t[data_t.val_mask], data_t.y[data_t.val_mask])
                    # cls_loss_s = cls_loss_s_o + cls_loss_t_o
                    cls_loss_s = cls_loss_s_o
                    entropy_loss = entropy_loss_f(logit_t)

                    acc_s = compute_accuracy_teacher(torch.argmax(logit_s.detach(), dim=1), data_s.y)
                    acc_test = compute_accuracy_teacher_mask(torch.argmax(logit_t.detach(), dim=1), data_t.y, data_t.test_mask)
                    mif1_test = f1_score(data_t.y.cpu().numpy(), torch.argmax(logit_t.detach(), dim=1).cpu().numpy(), average='micro')
                    maf1_test = f1_score(data_t.y.cpu().numpy(), torch.argmax(logit_t.detach(), dim=1).cpu().numpy(), average='macro')
                    loss = 2.5 * cls_loss_s + 0.5 * grl_loss + recover_loss + cycle_losses + cycle_recover_loss + entropy_loss * (epoch / 1000) + 0.0001 * diff_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if acc_test > best_acc:
                        best_acc = acc_test
                    if maf1_test > best_maf:
                        best_maf = maf1_test
                    if mif1_test > best_mif:
                        best_mif = mif1_test
                    print("Epoch: [{}/{}] | CLS Loss: {:.4f} | GRL Loss: {:.4f} | Re Loss: {:.4f} | Cycle Loss: {:.4f} | Cycle Recover Loss: {:.4f} | Diff Loss: {:.4f} | Entropy Loss: {:.4f} | Train acc: {:.4f} | Test acc: {:.4f} | MiF1: {:.4f} | MaF1: {:.4f}".format(
                        epoch+1, 1000, cls_loss_s.item(), grl_loss.item(), recover_loss.item(), cycle_losses.item(), cycle_recover_loss.item(), diff_loss.item(), entropy_loss.item(), acc_s, best_acc, best_mif, best_maf))
                    embeddings_s = encoded_s.detach().cpu().numpy()
                    embeddings_s_g = generation_z_s.detach().cpu().numpy()
                    embeddings_s_r = recover_z_s.detach().cpu().numpy()
                    embeddings_t = encoded_t.detach().cpu().numpy()
                    embeddings_t_g = generation_z_t.detach().cpu().numpy()
                    embeddings_t_r = recover_z_t.detach().cpu().numpy()
                    np.save('npy/{}-{}_embeddings_s.npy'.format(source_graph, target_graph), embeddings_s)
                    np.save('npy/{}-{}_embeddings_s_g.npy'.format(source_graph, target_graph), embeddings_s_g)
                    np.save('npy/{}-{}_embeddings_s_r.npy'.format(source_graph, target_graph), embeddings_s_r)
                    np.save('npy/{}-{}_embeddings_t.npy'.format(source_graph, target_graph), embeddings_s)
                    np.save('npy/{}-{}_embeddings_t_g.npy'.format(source_graph, target_graph), embeddings_t_g)
                    np.save('npy/{}-{}_embeddings_t_r.npy'.format(source_graph, target_graph), embeddings_t_r)













