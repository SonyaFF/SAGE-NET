from numpy.lib.function_base import average
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import math
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MultiheadAttention
from torch.nn.parameter import Parameter
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
'''
Our model : SAGENet   
'''    

######### SAGENet model with cross attention################################


class SAGENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(SAGENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.3)

        #self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        
        
        source_nodes, target_nodes = np.where(ehr_adj == 1)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # Load precomputed embeddings from GraphSAGE MODEL
        self.patient_embeddings = torch.tensor(np.load('/content/drive/MyDrive/Carmen-main/data/patient_embeddings_final.npy'), dtype=torch.float).to(device)
        self.drug_embeddings = torch.tensor(np.load('/content/drive/MyDrive/Carmen-main/data/drug_embeddings_final.npy'), dtype=torch.float).to(device)
        self.diagnosis_embeddings = torch.tensor(np.load('/content/drive/MyDrive/Carmen-main/data/diagnosis_embeddings.npy'), dtype=torch.float).to(device)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
        nn.ReLU(),
        nn.Linear(emb_dim * 4 , emb_dim * 2),  #############
        nn.ReLU(),
        nn.Linear(emb_dim * 2, vocab_size[2])
        )
                
        
        #####Add a cross attention layer
        self.num_heads = 2
        self.cross_attn = MultiheadAttention(emb_dim, self.num_heads)

        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)
      
        # Apply PCA to reduce feature dimension to 64
       # pca_drug = PCA(n_components=16)
       # drug_features_reduced = pca_drug.fit_transform(drug_features)
        
       # pca_patient = PCA(n_components=16)
       # patient_features_reduced = pca_patient.fit_transform(patient_features)
       # patient_features_final = patient_features_reduced[:16, :]
        
        # Convert to tensors
        #patient_features_tensor = torch.tensor(patient_features_final, dtype=torch.float).to(self.device)
        #drug_features_tensor = torch.tensor(drug_features_reduced, dtype=torch.float).to(self.device)
        
        #drug_embeddings = self.drug_sage(x=drug_features_tensor, edge_index=ehr_edge_index)
        patient_embeddings = self.patient_embeddings
        drug_embeddings = self.drug_embeddings
        
        
        #print("drug_embeddings shape:", drug_embeddings.shape)
        #print("patient_embeddings shape:", patient_embeddings.shape)
        #print("ddi_embeddings shape:", ddi_embeddings.shape)
        
        patient_embeddings_avg = patient_embeddings.mean(dim=0, keepdim=True)
        drug_embeddings_avg = drug_embeddings.mean(dim=0, keepdim=True)
        
        #print("patient_embeddings-avg shape:", patient_embeddings_avg.shape)

        ddi_embedding =  self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        ddi_embedding_avg = ddi_embedding.mean(dim=0, keepdim=True)
        
        
        # Add print statements before the concatenation to check the shape of the tensors
        #print("Shape of patient_embeddings_avg:", patient_embeddings_avg.shape)
        #print("Shape of drug_embeddings_avg:", drug_embeddings_avg.shape)
        #print("Shape of ddi_embedding:", ddi_embedding_avg .shape)
        
        patient_embeddings_avg = patient_embeddings_avg.unsqueeze(1)    
        drug_embeddings_avg  = drug_embeddings_avg .unsqueeze(1) 
        ddi_embedding_avg = ddi_embedding_avg.unsqueeze(1) 
        
        #print("Shape of patient_embeddings_avg:", patient_embeddings_avg.shape)
        #print("Shape of drug_embeddings_avg:", drug_embeddings_avg.shape)
        #print("Shape of ddi_embedding:", ddi_embedding_avg .shape)
        
        #'''Cross Attention'''#
        cross_attn_output, cross_attn_weights = self.cross_attn(patient_embeddings_avg, drug_embeddings_avg, ddi_embedding_avg) # (1, 1, dim), (1, 1, 1)
        cross_attn_output = cross_attn_output.transpose(0, 1) # (1, dim)
        #print('cross_attn_output=' , cross_attn_output.shape)

     
        output = self.output(torch.cat([ patient_embeddings_avg, drug_embeddings_avg, ddi_embedding_avg, cross_attn_output], dim=-1)) # (1, dim)###############333
        #print("output shape:", output.shape)
        
        output = output.squeeze(0)
        #print('new_output=', output.shape)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)

####################################################################
### baseline models:::::::::::#####
'''
DMNC
'''
class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, :r * w].reshape(b, r, w))  # .contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(input[:, r * w:r * w + r].reshape(b, r))  # .contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(input[:, (r * w + r):].reshape(b, r, 3), -1)  # .contiguous().view(b, r, 3)
        return read_keys, read_strengths, read_modes


class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=64, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2]+1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )
        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)


    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1) # (len, dim)

                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1) # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1) # (1, len)
                input_embedding = attn_weight.mm(input_embedding) # (1, dim)

                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0) # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))
            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)


class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.5)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)