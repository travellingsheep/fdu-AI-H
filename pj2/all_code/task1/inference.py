import numpy as np
from collections import defaultdict
import pickle
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

class HMM:
    def __init__(self,smoothing):
        self.states = []
        self.state_num = 0
        self.initial_prob = {}
        self.transition_prob = {}
        self.emission_prob = {}
        self.vocab = set()
        self.smoothing = smoothing
        
    def train(self,training_data):
        initial_state_count = defaultdict(int)
        transition_count = defaultdict(lambda: defaultdict(int))
        emission_count = defaultdict(lambda: defaultdict(int))

        for sentence, tags in training_data:
            prev_tag = None
            for word, tag in zip(sentence, tags):
                self.states.append(tag)
                self.vocab.add(word)
                
                if prev_tag == None:
                    initial_state_count[tag] += 1
                else:
                    transition_count[prev_tag][tag] += 1
                emission_count[tag][word] += 1

                prev_tag=tag

        self.states = list(set(self.states))
        vocab_length = len(self.vocab)
        self.state_num=len(self.states)
        sentence_num = len(training_data)
        for state in self.states:
            self.initial_prob[state]=(initial_state_count[state]+self.smoothing)/(sentence_num+self.smoothing*self.state_num)
        for from_state in self.states:
            total_transitions=sum(transition_count[from_state].values())+self.smoothing*self.state_num
            for to_state in self.states:
                self.transition_prob[f"{from_state}->{to_state}"]=(transition_count[from_state][to_state]+self.smoothing)/total_transitions
        for state in self.states:
            self.emission_prob[state]={}
            total_emission=sum(emission_count[state].values())+self.smoothing*(vocab_length+1)
            for word in self.vocab:
                self.emission_prob[state][word]=(emission_count[state].get(word,0)+self.smoothing)/total_emission
            self.emission_prob[state]['<UNK>']=self.smoothing/total_emission
    
    def viterbi_decode(self, sentence):
        length = len(sentence)
        DPTable = {state: [0] * length for state in self.states}
        backpointer = {state: [0] * length for state in self.states}

        for state in self.states:
            word = sentence[0]
            emission_prob = self.emission_prob[state].get(word, self.emission_prob[state]["<UNK>"])
            DPTable[state][0] = self.initial_prob[state] * emission_prob #不加_不会导致与emission_prob这个dict混淆
        
        for t in range(1, length):
            word = sentence[t]
            for to_state in self.states:
                max_prob = 0
                best_state = self.states[0]
                for from_state in self.states:
                    transition_prob = self.transition_prob.get(f"{from_state}->{to_state}", self.smoothing / len(self.states))
                    prob = DPTable[from_state][t-1] * transition_prob
                    if prob > max_prob:
                        max_prob = prob
                        best_state = from_state
                emission_prob = self.emission_prob[to_state].get(word, self.emission_prob[to_state]["<UNK>"])
                DPTable[to_state][t] = max_prob * emission_prob
                backpointer[to_state][t] = self.states.index(best_state)
        
        best_path = [0] * length
        best_path[length-1] = np.argmax([DPTable[state][length-1] for state in self.states])
        for t in range(length-2, -1, -1):
            best_path[t] = backpointer[self.states[best_path[t+1]]][t+1]
        
        return [self.states[i] for i in best_path]
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'states': self.states,
                'initial_prob': self.initial_prob,
                'transition_prob': self.transition_prob,
                'emission_prob': self.emission_prob,
                'vocab': self.vocab
            }, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    model = HMM(smoothing=0)
    model.states = data['states']
    model.state_num = len(model.states)
    model.initial_prob = data['initial_prob']
    model.transition_prob = data['transition_prob']
    model.emission_prob = data['emission_prob']
    model.vocab = data['vocab']
    return model

def predict_and_save(input_file, model, output_file):
    dataset = read_data(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, _ in dataset:
            predicted_tags = model.viterbi_decode(sentence)
            for char, tag in zip(sentence, predicted_tags):
                f.write(f"{char} {tag}\n")
            f.write("\n")
if __name__ == "__main__":
    model_path = r"Chinese_model.pt"
    model = load_model(model_path)

    input_file = r"chinese_test.txt"
    output_file = r"Chinese_output.txt"
    predict_and_save(input_file, model, output_file)