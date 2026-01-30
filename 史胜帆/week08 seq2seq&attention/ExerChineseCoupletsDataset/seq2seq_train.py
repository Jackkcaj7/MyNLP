#seq2seq模型训练
from EncoderDecoderAttenModel import Seq2Seq
import torch
import torch.nn as nn
import json
import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_process import Vocabulary,get_proc
from tqdm import tqdm

if __name__ == "__main__":
    writer = SummaryWriter()
    train_loss_cnt = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open("ExerChineseCoupletsDataset/encoder.json",'r',encoding='utf-8') as f:
        enc_data = json.load(f)
    with open("ExerChineseCoupletsDataset/decoder.json",'r',encoding='utf-8') as f:
        dec_data = json.load(f)
    with open('ExerChineseCoupletsDataset/vocab_idx.bin','rb') as f:
        vocab_idx = pickle.load(f)
    
    dataset = list(zip(enc_data,dec_data))
    datald = DataLoader(dataset,batch_size = 128,shuffle=True,
                        collate_fn=get_proc(vocab_idx))
    
    #模型搭建
    model = Seq2Seq(enc_input_dim=len(vocab_idx),
                    dec_input_dim=len(vocab_idx),
                    ebd_dim = 256,
                    hidden_dim=360,
                    dropout=0.2,num_layers = 2)
    
    model.to(device)
    #损失函数 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01,
                                 weight_decay=1e-5)
    
    for epoch in range(20):
        model.train()
        tpbar = tqdm(datald)
        for enc_input,dec_input,target in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            target = target.to(device)

            y_hat,_ = model(enc_input,dec_input)

            loss = criterion(y_hat.view(-1,y_hat.size(-1)),
                             target.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f"epoch:{epoch + 1},loss:{loss.item()}")
            writer.add_scalar('loss/train',loss.item(),train_loss_cnt)
            train_loss_cnt += 1

    torch.save(model.state_dict(),'seq2seq_state.bin')