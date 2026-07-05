import numpy as np
from collections import defaultdict
import argparse
import pickle
import os
import numpy as np
from collections import defaultdict
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

class CRF_Perceptron_Dynamic:
    def __init__(self, label_to_index, word_to_index, templates=None):
        self.label_to_index = label_to_index
        self.num_labels = len(label_to_index)
        self.word_to_index = word_to_index 
        self.index_to_label = {i: label for label, i in label_to_index.items()}
        self.start_label_index = label_to_index.get("<START>")
        self.stop_label_index = label_to_index.get("<STOP>")
        self.bigram_params = defaultdict(float)

        self.u_feature_params = defaultdict(float)
        self.b_feature_params = defaultdict(float)
        self.templates = templates

    def _get_bigram_score(self, prev_label_index, current_label_index):
        return self.bigram_params.get((prev_label_index, current_label_index), 0.0)

    def _get_label_word_score(self, label_index, word_index, word=None, prev_word_index=None, next_word_index=None, language="English", features=None, prev_label_index=None):
        score = 0.0
        if features:
            u_features, b_features = features
            for feat in u_features:
                score += self.u_feature_params.get((label_index, feat), 0.0)

            if prev_label_index is not None:
                for feat in b_features:
                    score += self.b_feature_params.get((prev_label_index, label_index, feat), 0.0)
        return score

    def viterbi_decode(self, word_indices, words=None, language="English"):
        seq_len = len(word_indices)
        viterbi_table = np.zeros((seq_len, self.num_labels))
        backpointers = np.zeros((seq_len, self.num_labels), dtype=int)

        for label_index in range(self.num_labels):
            if label_index != self.start_label_index and label_index != self.stop_label_index:
                features = extract_features(words, 0, self.templates) if (words and self.templates) else ([], [])
                viterbi_table[0, label_index] = (
                    self._get_label_word_score(
                        label_index, word_indices[0], words[0] if words else None,
                        None, word_indices[1] if len(word_indices) > 1 else None, language,
                        features=features, prev_label_index=self.start_label_index
                    ) + self._get_bigram_score(self.start_label_index, label_index)
                )
                backpointers[0, label_index] = self.start_label_index

        for t in range(1, seq_len):
            for current_label_index in range(self.num_labels):
                if current_label_index != self.start_label_index and current_label_index != self.stop_label_index:
                    scores = []
                    indices = []
                    for prev_label_index in range(self.num_labels):
                        if prev_label_index != self.start_label_index and prev_label_index != self.stop_label_index:
                            features = extract_features(words, t, self.templates) if (words and self.templates) else ([], [])
                            score = (
                                viterbi_table[t - 1, prev_label_index] +
                                self._get_label_word_score(
                                    current_label_index, word_indices[t], words[t] if words else None,
                                    word_indices[t-1] if t > 0 else None,
                                    word_indices[t+1] if t < seq_len-1 else None, language,
                                    features=features, prev_label_index=prev_label_index
                                ) + self._get_bigram_score(prev_label_index, current_label_index))
                            scores.append(score)
                            indices.append(prev_label_index)
                    max_score_index = np.argmax(scores)
                    viterbi_table[t, current_label_index] = scores[max_score_index]
                    backpointers[t, current_label_index] = indices[max_score_index]

        final_scores = []
        last_indices = []
        for label_index in range(self.num_labels):
            if label_index != self.start_label_index and label_index != self.stop_label_index:
                score = viterbi_table[-1, label_index] + self._get_bigram_score(label_index, self.stop_label_index)
                final_scores.append(score)
                last_indices.append(label_index)
        
        max_final_index = np.argmax(final_scores)
        last_label_index = last_indices[max_final_index]
        best_path = [last_label_index]
        for t in range(seq_len - 1, 0, -1):
            prev_label_index = backpointers[t, best_path[-1]]
            best_path.append(prev_label_index)
        best_path.reverse()
        return best_path

    def forward(self, words, language="English"):
        word_indices = [self.word_to_index.get(word, self.word_to_index.get("<UNK>", 0)) for word in words]
        return self.viterbi_decode(word_indices, words, language)

def train_online(model, training_data, evaluate_dataset, num_iterations, language, gold_path, model_dir):
    bigram_summed = defaultdict(float)
    u_feature_summed = defaultdict(float)
    b_feature_summed = defaultdict(float)
    total_sentences = 0
    best_f1 = 0
    best_params = None

    for iteration in range(num_iterations):
        print(f"Iteration: {iteration + 1}")
        for words, true_labels in training_data:
            word_indices = [model.word_to_index.get(word, model.word_to_index.get("<UNK>", 0)) for word in words]
            true_label_indices = [model.label_to_index[label] for label in true_labels]
            predicted_indices = model.viterbi_decode(word_indices, words, language)

            for i in range(1, len(true_label_indices)):
                true_bigram = (true_label_indices[i-1], true_label_indices[i])
                predicted_bigram = (predicted_indices[i-1], predicted_indices[i])
                if true_bigram != predicted_bigram:
                    model.bigram_params[true_bigram] += 1
                    model.bigram_params[predicted_bigram] -= 1

            if true_label_indices:
                true_stop = (true_label_indices[-1], model.stop_label_index)
                predicted_stop = (predicted_indices[-1], model.stop_label_index)
                if true_stop != predicted_stop:
                    model.bigram_params[true_stop] += 1
                    model.bigram_params[predicted_stop] -= 1

            for i in range(len(word_indices)):
                features = extract_features(words, i, model.templates) if model.templates else ([], [])
                u_features, b_features = features
                
                for feat in u_features:
                    true_u_feat = (true_label_indices[i], feat)
                    pred_u_feat = (predicted_indices[i], feat)
                    if true_u_feat != pred_u_feat:
                        model.u_feature_params[true_u_feat] += 1
                        model.u_feature_params[pred_u_feat] -= 1
                
                if i > 0:
                    for feat in b_features:
                        true_b_feat = (true_label_indices[i-1], true_label_indices[i], feat)
                        pred_b_feat = (predicted_indices[i-1], predicted_indices[i], feat)
                        if true_b_feat != pred_b_feat:
                            model.b_feature_params[true_b_feat] += 1
                            model.b_feature_params[pred_b_feat] -= 1

            for key, value in model.bigram_params.items():
                bigram_summed[key] += value
            for key, value in model.u_feature_params.items():
                u_feature_summed[key] += value
            for key, value in model.b_feature_params.items():
                b_feature_summed[key] += value
            total_sentences += 1

        temp_path = os.path.join(model_dir, f"temp_validation_iter{iteration+1}.txt")
        save_result(evaluate_dataset, model, temp_path)
        result = check(language, gold_path, temp_path)
        f1 = result
        print(f"Iteration {iteration+1}, F1: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            best_params = {
                'bigram_params': model.bigram_params.copy(),
                'u_feature_params': model.u_feature_params.copy(),
                'b_feature_params': model.b_feature_params.copy()
            }

    avg_bigram_params = defaultdict(float)
    avg_u_feature_params = defaultdict(float)
    avg_b_feature_params = defaultdict(float)
    for key in bigram_summed:
        avg_bigram_params[key] = bigram_summed[key] / total_sentences if total_sentences > 0 else 0.0
    for key in u_feature_summed:
        avg_u_feature_params[key] = u_feature_summed[key] / total_sentences if total_sentences > 0 else 0.0
    for key in b_feature_summed:
        avg_b_feature_params[key] = b_feature_summed[key] / total_sentences if total_sentences > 0 else 0.0

    model.bigram_params = avg_bigram_params
    model.u_feature_params = avg_u_feature_params
    model.b_feature_params = avg_b_feature_params
    temp_path = os.path.join(model_dir, f"temp_validation_avg_iter{num_iterations}.txt")
    save_result(evaluate_dataset, model, temp_path)
    result = check(language, gold_path, temp_path)
    avg_f1 = result
    print(f"Averaged Perceptron F1: {avg_f1}")

    if best_f1 > avg_f1 and best_params:
        model.bigram_params = best_params['bigram_params']
        model.u_feature_params = best_params['u_feature_params']
        model.b_feature_params = best_params['b_feature_params']
        final_f1 = best_f1
        print(f"Using best non-averaged parameters with F1: {best_f1}")
    else:
        final_f1 = avg_f1
        print(f"Using averaged parameters with F1: {avg_f1}")

    log_path = os.path.join(model_dir, "param_stats_log.csv")
    header = "num_iterations,total_sentences,final_score\n"
    log_line = f"{num_iterations},{total_sentences},{final_f1}\n"
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)

    my_path = os.path.join(model_dir, f"validation0{num_iterations}_epoch={num_iterations}.txt")
    save_result(evaluate_dataset, model, my_path)
    result = check(language, gold_path, my_path)
    print(f"Final result: {result}")
    return final_f1

def save_result(dataset, model, output_file, language = "English"):
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence, _ in dataset:
                predicted_labels = [model.index_to_label[idx] for idx in model.forward(sentence, language = language)]
                for word, label in zip(sentence, predicted_labels):
                    f.write(f"{word} {label}\n")
                f.write("\n")

def parse_crf_template(template_path):
    templates = []
    with open(template_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            name, expr = line.split(':', 1)
            offsets = []
            for part in expr.split('/'):
                idx = part.find('[')
                if idx != -1:
                    offset = part[idx+1:part.find(']')]
                    row, col = map(int, offset.split(','))
                    offsets.append((row, col))
            templates.append((name, offsets))
    return templates

def extract_features(sentence, i, templates):
    u_features = []
    b_features = []
    
    for name, offsets in templates:
        vals = []
        for row, col in offsets:
            pos = i + row
            if 0 <= pos < len(sentence):
                vals.append(sentence[pos])
            else:
                vals.append('BOS' if pos < 0 else 'EOS')
        feature = name + '=' + '/'.join(vals)
        
        if name.startswith('U'):
            u_features.append(feature)
        elif name.startswith('B'):
            b_features.append(feature)
    
    return u_features, b_features
def load_model(model_path, templates):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = CRF_Perceptron_Dynamic(
        data['label_to_index'],
        data['word_to_index'],
        templates=templates
    )
    model.bigram_params = data['bigram_params']
    model.u_feature_params = data['u_feature_params']
    model.b_feature_params = data['b_feature_params']
    return model

def predict_on_new_data(model, input_file, output_file, language="English"):
    dataset = read_data(input_file)
    save_result(dataset, model, output_file, language=language)
    print(f"预测结果已保存到: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM Model Training and Evaluation")
    parser.add_argument("--language", type=str, required=True, help="To train which language")

    args = parser.parse_args()
    language = args.language
    model_path = r".pkl"
    input_file = r".txt"
    output_file = r".txt"

    template_path = r".utf8"
    templates = parse_crf_template(template_path)

    model = load_model(model_path, templates)
    predict_on_new_data(model, input_file, output_file, language=language)