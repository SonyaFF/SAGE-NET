from rdkit import Chem
import torch
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from torch.nn.init import xavier_uniform_

   
 
import dill
   
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from util import Metrics
from util import get_ehr_adj   
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch.nn.functional as F
from collections import defaultdict

from models import SAGENet
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params


model_name = 'SAGENet'
resume_name = 'saved/SAGENet/Epoch_102_JA_0.5073_DDI_0.0858.model'



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")
parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
parser.add_argument('--target_ddi', type=float, default=0.06, help="target ddi")    
parser.add_argument('--T', type=float, default=2.0, help='T')
parser.add_argument('--decay_weight', type=float, default=0.85, help="decay weight")
parser.add_argument('--seed', type=int, default=1029, help='random seed')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--datadir', type=str, default="../data_ordered/", help='dimension')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--early_stop', type=int, default=30, help='early stop number')

args = parser.parse_args()

if not os.path.exists(os.path.join("saved", args.model_name)):
    os.makedirs(os.path.join("saved", args.model_name))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda > -1:
    torch.cuda.manual_seed(args.seed)

# evaluate
def eval(model, data_eval, voc_size, epoch, metric_obj):
    model.eval()

    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            res = model(input[:adm_idx+1])
            # for other baseline
            target_output = res[0] if isinstance(res, tuple) else res

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)
            
            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        metric_obj.feed_data(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=os.path.join(args.datadir, 'ddi_A_final_4.pkl'))
    print("DDI Rate: {:.4}".format(ddi_rate))
    metric_obj.set_data(save=args.Test)
    ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run()
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt

  
def main():
    # Print the learning rate
    print("Learning Rate:", args.lr)
    parser.add_argument('--MIMIC', type=int, default=3, help='MIMIC version')
    data_path = os.path.join(args.datadir, '/content/drive/MyDrive/Carmen-main/data/records_final_4.pkl')
  
    voc_path = os.path.join(args.datadir, '/content/drive/MyDrive/Carmen-main/data/voc_final_4.pkl')
    # Define the data path and load the data
    #data_path = os.path.join(args.datadir, '/content/drive/MyDrive/Carmen-main/data/data_final.pkl')
    
    #ehr_adj2_new.pkl
    #ehr_adj_path = os.path.join(args.datadir,'/content/drive/MyDrive/Carmen-main/data/ehr_adj_final_4.pkl')
    ehr_adj_path = os.path.join(args.datadir,'/content/drive/MyDrive/Carmen-main/data/ehr_adj2_new.pkl')
    
    ddi_adj_path = os.path.join(args.datadir,'/content/drive/MyDrive/Carmen-main/data/ddi_A_final_4.pkl')
    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    
    data2_path = os.path.join(args.datadir,'/content/drive/MyDrive/Carmen-main/data/data_with_labels.pkl')
    data2 = dill.load(open(data2_path, 'rb'))
    #print(data.columns)
    # One-hot encode patient features
    patient_features = pd.get_dummies(data2['SUBJECT_ID'].astype(str))
    unique_patient_data = data2.drop_duplicates(subset='SUBJECT_ID')
    unique_patient_data.set_index('SUBJECT_ID', inplace=True)
    patient_features = unique_patient_data.join(patient_features)
    patient_features = patient_features.apply(pd.to_numeric, errors='coerce')
    
    # One-hot encode drug features
    exploded_ndc = data2[['SUBJECT_ID', 'NDC']].explode('NDC')
    drug_features = pd.get_dummies(exploded_ndc['NDC'].astype(str), prefix='NDC')
    drug_features = drug_features.groupby(exploded_ndc['SUBJECT_ID']).sum()
    
    # One-hot encode diagnosis features
    exploded_icd9 = data2[['SUBJECT_ID', 'ICD9_CODE']].explode('ICD9_CODE')
    diagnosis_features = pd.get_dummies(exploded_icd9['ICD9_CODE'].astype(str), prefix='ICD9')
    diagnosis_features = diagnosis_features.groupby(exploded_icd9['SUBJECT_ID']).sum()
    
    # Normalize features
    scaler = StandardScaler()
    patient_features = scaler.fit_transform(patient_features.fillna(0))
    drug_features = scaler.fit_transform(drug_features.fillna(0))
    diagnosis_features = scaler.fit_transform(diagnosis_features.fillna(0))
    
    patient_features_tensor = torch.tensor(patient_features, dtype=torch.float).to(device)
    drug_features_tensor = torch.tensor(drug_features, dtype=torch.float).to(device)
    diagnosis_features_tensor = torch.tensor(diagnosis_features, dtype=torch.float).to(device)
        
            
            
    # Convert adjacency matrices to edge indices
    ehr_source_nodes, ehr_target_nodes = np.where(ehr_adj == 1)
    ddi_source_nodes, ddi_target_nodes = np.where(ddi_adj == 1)
    
    ehr_edge_index = torch.tensor([ehr_source_nodes, ehr_target_nodes], dtype=torch.long).to(device)
    ddi_edge_index = torch.tensor([ddi_source_nodes, ddi_target_nodes], dtype=torch.long).to(device)
    

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    metric_obj = Metrics(data, med_voc, args)

    # np.random.seed(2048)
    # np.random.shuffle(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = SAGENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=args.ddi)
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
    
    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        result = []
        for _ in range(1):
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test, voc_size, 0, metric_obj)
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)
        print ('test time: {}'.format(time.time() - tic))
        return 
   
    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 100   #################################################
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        prediction_loss_cnt, neg_loss_cnt = 0, 0
        model.train()
        for step, input in enumerate(data_train):
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item
 
                target_output1, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss_multi_target).to(device))
                if args.ddi:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]], path= os.path.join(args.datadir,'ddi_A_final_4.pkl'))
                    if current_ddi_rate <= args.target_ddi:
                        loss = 0.9 * loss_bce + 0.1 * loss_multi
                        prediction_loss_cnt += 1
                    else:
                        rnd = np.exp((args.target_ddi - current_ddi_rate) / args.T)
                        if np.random.rand(1) < rnd:
                            loss = loss_ddi
                            neg_loss_cnt += 1
                        else:
                            loss = 0.9 * loss_bce + 0.1 * loss_multi
                            prediction_loss_cnt += 1
                else:
                    loss = 0.9 * loss_bce + 0.1 * loss_multi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        args.T *= args.decay_weight

        print ()
        tic2 = time.time() 
        
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, metric_obj)
        #return ddi_rate, ja, prauc, ehr_adj, avg_p, avg_r, avg_f1, avg_med
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, "best.model"), 'wb'))

        print ('best_epoch: {}'.format(best_epoch))

        if epoch - best_epoch > args.early_stop:
            print("Early Stop...")
            break

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main()
