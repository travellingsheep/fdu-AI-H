import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import time
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

sorted_labels_eng= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC" , "I-MISC"]

sorted_labels_chn = [
'O',
'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
, 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
, 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
, 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
, 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
, 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
, 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
, 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]

def check(language, gold_path, my_path):
    if language == "English":
        sort_labels = sorted_labels_eng
    else:
        sort_labels = sorted_labels_chn
    y_true = []
    y_pred = [] #, encoding="utf-8"
    with open(gold_path, "r", encoding="utf-8") as g_f, open(my_path, "r", encoding="utf-8") as m_f:
        g_lines = g_f.readlines()
        m_lines = m_f.readlines()
        # assert len(g_lines) == len(m_lines), "Length is Not Equal."
        for i in range(len(g_lines)):
            if g_lines[i].strip() == "":  # 跳过空行
                continue
            if i >= len(m_lines) or m_lines[i].strip() == "":  # 检查预测文件对应行
                continue
            
            g_parts = g_lines[i].strip().split(" ")
            m_parts = m_lines[i].strip().split(" ")
            
            if len(g_parts) >= 2 and len(m_parts) >= 2:
                g_word, g_tag = g_parts[0], g_parts[1]
                m_word, m_tag = m_parts[0], m_parts[1]
                y_true.append(g_tag)
                y_pred.append(m_tag)
    
    report = metrics.classification_report(
        y_true = y_true, y_pred=y_pred, labels=sort_labels[1:], digits=4, output_dict=True
    )
    return report["micro avg"]["f1-score"]

def read_data(file_path):
    sentences=[]
    sentence_s=[]
    sentence_o=[]
    
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            if not line:
                if sentence_s:
                    sentences.append((sentence_o,sentence_s))
                    sentence_s=[]
                    sentence_o=[]
            else:
                parts=line.split()
                sentence_o.append(parts[0])
                sentence_s.append(parts[1])
        if sentence_s:
            sentences.append((sentence_o,sentence_s))

    return sentences

class NERDataset(Dataset):
    def __init__(self, data: List[Tuple[List[str], List[str]]], char_to_idx: Dict[str, int], 
                 label_to_idx: Dict[str, int], max_len: int = 256):
        super().__init__()
        self.data = data
        self.char_to_idx = char_to_idx
        self.label_to_idx = label_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chars, labels = self.data[idx]
        
        char_ids = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in chars]
        label_ids = [self.label_to_idx[label] for label in labels]
        
        if len(char_ids) > self.max_len:
            char_ids = char_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
        
        seq_len = len(char_ids)
        char_ids += [self.char_to_idx['<PAD>']] * (self.max_len - len(char_ids))
        label_ids += [self.label_to_idx['O']] * (self.max_len - len(label_ids))
        
        return {
            'input_ids': torch.tensor(char_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'seq_len': torch.tensor(seq_len, dtype=torch.long)
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        #position max_len*1,div_term d_model//2
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #[0,1,2]->[[0],[1],[2]] 即3->3,1;若squeeze(0)则是3->1,3即[[0,1,2]]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        #position * div_term max_len,d_model//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #pe max_len,d_model->1,max_len,d_model
        
        self.register_buffer('pe', pe)

    def forward(self, x): #x:[batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
    def _compute_partition_function(self, emissions, mask): #所有可能路径
        _, seq_len, _ = emissions.size()
        mask = mask.bool()
        score = self.start_transitions + emissions[:, 0]  # [batch_size, num_tags] emissions:[batch_size, seq_len, num_tags],:是切片，代表第一维的全部内容，第二个位置上的0代表取第零咧，后面第三维的操作没写，默认取全部
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # [batch_size, num_tags, 1]
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)  # [batch_size, num_tags] #dim=1是因为第二维是from_label，第三维是to_label
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score) 
        
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # [batch_size]
    
    def _compute_score(self, emissions, tags, mask): #真实标签的分数
        _, seq_len = tags.size()
        mask = mask.bool()

        score = self.start_transitions[tags[:, 0]]  # [batch_size]
        score = score + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1) #gather要求index和input的形状相同，因此先unsqueeze，然后广播至num_tags
        
        for i in range(1, seq_len):
            transition_score = self.transitions[tags[:, i-1], tags[:, i]]
            emission_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score = score + (transition_score + emission_score) * mask[:, i].float()
        
        last_tag_indices = mask.sum(1) - 1  # 最后一个有效标签的索引 dim=1：跨“列”求和，结果是每一行的和
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        
        return score
    
    def forward(self, emissions, tags, mask):
        log_partition = self._compute_partition_function(emissions, mask)
        gold_score = self._compute_score(emissions, tags, mask)
        return torch.mean(log_partition - gold_score) #在batch内取平均
    
    def viterbi_decode(self, emissions, mask):
        batch_size, seq_len, _ = emissions.size()
        mask = mask.bool()
        
        score = self.start_transitions + emissions[:, 0]  # [batch_size, num_tags]
        history = []
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # [batch_size, num_tags, 1]
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)  # [batch_size, num_tags]，这里选的是from_label中分数最大的，仍然是每一个to_label都有一个分数
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)
        
        score = score + self.end_transitions
        
        best_paths = []
        for b in range(batch_size):
            seq_len_b = mask[b].sum().item()
            _, best_last_tag = score[b].max(dim=0)
            best_path = [best_last_tag.item()]
            
            for i in range(len(history) - 1, -1, -1):
                if i < seq_len_b - 1:
                    best_last_tag = history[i][b][best_last_tag]
                    best_path.append(best_last_tag.item())
            
            best_path.reverse()
            best_paths.append(best_path[:seq_len_b])
        
        return best_paths

class TransformerCRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, d_model: int = 64, 
                 nhead: int = 8, num_layers: int = 4, max_len: int = 256, dim_feedforward: int = None):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_tags)
        self.crf = CRF(num_tags)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_ids, labels=None, seq_lens=None):
        mask = (input_ids != 0).float()
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)

        attention_mask = (input_ids == 0)
        
        hidden_states = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        emissions = self.classifier(hidden_states)
        
        if labels is not None:
            loss = self.crf(emissions, labels, mask)
            return loss
        else:
            predictions = self.crf.viterbi_decode(emissions, mask)
            return predictions

def build_vocab(data: List[Tuple[List[str], List[str]]]):
    chars = set(['<PAD>', '<UNK>'])
    labels = set()
    
    for char_seq, label_seq in data:
        chars.update(char_seq)
        labels.update(label_seq)
    
    char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    return char_to_idx, label_to_idx, idx_to_label

def evaluate_model(model, dataloader, idx_to_label, device, data_path, output_file):
    model.eval()
    sentences = read_data(data_path)
    
    pred_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            seq_lens = batch['seq_len'].to(device)
            
            predictions = model(input_ids)
            
            for i, pred_path in enumerate(predictions):
                seq_len = seq_lens[i].item()
                pred_labels = pred_path[:seq_len]
                
                sentence_idx = batch_idx * dataloader.batch_size + i
                if sentence_idx < len(sentences):
                    words, _ = sentences[sentence_idx]
                    min_len = min(len(words), len(pred_labels))
                    
                    for j in range(min_len):
                        pred_tag = idx_to_label[pred_labels[j]]
                        pred_results.append(f"{words[j]} {pred_tag}")
                    
                    if sentence_idx < len(sentences) - 1:
                        pred_results.append("")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in pred_results:
            f.write(line + '\n')
    
    language = "Chinese" if "Chinese" in data_path else "English"
    f1_score = check(language, data_path, output_file)
    return f1_score

def train_model(sample_data,language,device,epochs):
    experiment_no = 5
    
    char_to_idx, label_to_idx, idx_to_label = build_vocab(sample_data)
    
    print(f"标签数量: {len(label_to_idx)}")
    print(f"标签: {list(label_to_idx.keys())}")
    print(f"词汇表大小: {len(char_to_idx)}")
    
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:]
    
    train_dataset = NERDataset(train_data, char_to_idx, label_to_idx)
    val_dataset = NERDataset(val_data, char_to_idx, label_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    vocab_size = len(char_to_idx)
    num_tags = len(label_to_idx)
    
    model = TransformerCRF(vocab_size, num_tags, d_model=64, nhead=8, num_layers=4, dim_feedforward=2048)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    if language == "Chinese":
        train_data_path = r"train.txt"
    else:
        train_data_path = r"train.txt"
    
    num_epochs = epochs
    best_f1 = 0
    f1_scores = []
    train_losses = []
    batch_losses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            forward_start = time.time()
            loss = model(input_ids, labels)
            forward_time = time.time() - forward_start
            
            backward_start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            backward_time = time.time() - backward_start
            
            batch_time = time.time() - batch_start_time
            
            total_loss += loss.detach()
            
            if batch_idx % 25 == 0:
                current_loss = loss.item()
                batch_losses.append(current_loss)
                epoch_losses.append(current_loss)
                
                print(f"Epoch {epoch+1}, Batch {batch_idx}")
                print(f"  前向传播: {forward_time:.3f}s")
                print(f"  反向传播: {backward_time:.3f}s")
                print(f"  总计: {batch_time:.3f}s")
                print(f"  损失值: {current_loss:.4f}")
                print("-" * 30)

        avg_loss = (total_loss / len(train_loader)).item()
        train_losses.append(avg_loss)
        scheduler.step()
        
        if epoch >= 4:  
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'vocab_size': len(char_to_idx),
                'num_tags': len(label_to_idx)
            }, f'No_{experiment_no}_transformer_crf_epoch{epoch+1}_language={language}.pth')
            
            train_output = f'train_eval_exp{experiment_no}_epoch{epoch+1}_{language}.txt'
            val_f1_micro = evaluate_model(model, val_loader, idx_to_label, device, 
                                        train_data_path, train_output)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"训练损失: {avg_loss:.4f}")
            print(f"训练集上F1: {val_f1_micro:.4f}")
            print(f"预测文件已保存: {train_output}")
            f1_scores.append(val_f1_micro)
            print("-" * 50)

            if val_f1_micro > best_f1:
                best_f1 = val_f1_micro
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'char_to_idx': char_to_idx,
                    'label_to_idx': label_to_idx,
                    'idx_to_label': idx_to_label,
                    'vocab_size': len(char_to_idx),
                    'num_tags': len(label_to_idx)
                }, f'No_{experiment_no}_best_transformer_crf_model_language={language}.pth')
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'vocab_size': len(char_to_idx),
                'num_tags': len(label_to_idx)
            }, f'No_{experiment_no}_transformer_crf_epoch{epoch+1}_language={language}.pth')
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"训练损失: {avg_loss:.4f}")
            print("-" * 50)
    
    print(f"训练完成！最佳F1分数: {best_f1:.4f}")
    
    return model, char_to_idx, label_to_idx, idx_to_label, f1_scores, train_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="language")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device: cuda or cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    
    if args.language == "Chinese":
        train_path = r"train.txt"
        validation_path = r"validation.txt"
    else:
        train_path = r"train.txt"
        validation_path = r"validation.txt"
    data = read_data(train_path)
    print(f"训练数据样本数: {len(data)}")
    num_epochs = 5
    model, char_to_idx, label_to_idx, idx_to_label, f1_scores, train_losses = train_model(data,args.language,device,num_epochs)
    validation_data = read_data(validation_path)
    print(f"验证数据样本数: {len(validation_data)}")
    
    validation_dataset = NERDataset(validation_data, char_to_idx, label_to_idx)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    
    print("=" * 60)
    print("在验证集上评估模型...")
    model.eval()
    
    experiment_no = 12

    print("=" * 60)
    print("在验证集上评估所有保存的模型...")

    best_val_f1 = 0
    best_epoch = 0
    best_model_path = ""
    validation_f1_scores = [] 
    validation_epochs = [] 

    for epoch in range(1, num_epochs + 1):  
        model_path = f'No_{experiment_no}_transformer_crf_epoch{epoch}_language={args.language}.pth'
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        output_file = f'predictions_exp{experiment_no}_epoch{epoch}_{args.language}.txt'
        print(f"\nEpoch {epoch} 验证结果:")
        val_f1_score = evaluate_model(model, validation_loader, idx_to_label, device, 
                                    validation_path, output_file)
        print(f"Epoch {epoch}: 验证集 F1-Score: {val_f1_score:.4f}")

        validation_epochs.append(epoch)
        validation_f1_scores.append(val_f1_score)

        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_epoch = epoch
            best_model_path = model_path

    print(f"\n验证集最佳模型: {best_model_path} (Epoch {best_epoch})")
    print(f"最佳验证集 F1-Score (customized_check): {best_val_f1:.4f}")
    print("=" * 60)
    