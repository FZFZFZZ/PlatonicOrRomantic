import numpy as np
from IPython.display import display, HTML
import re
from lime.lime_text import LimeTextExplainer
from .predictor import LstmPredictor
import random
import pandas as pd
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class LimeExplainer:
    CLASS_NAMES = ['-1', '0', '1']
    INVISIBLE_A = "\u2060"
    INVISIBLE_B = "\u2061"
    NUM_SAMPLES = 200 # empirically good in terms of convergence and speed
    NUM_KEYWORDS = 10 # how many keywords to highlight in one dialogue
    NUM_KEYSENTENCE = 2 # how many sentences to highlight in one dialogue
    
    def __init__(self, predictor: LstmPredictor):
        self.predictor = predictor

    def convert_text_to_dialogue(self, text):
        dialogue = []
        segments = re.split(f"({self.INVISIBLE_A}|{self.INVISIBLE_B})", text)

        role_map = {self.INVISIBLE_A: 'A', self.INVISIBLE_B: 'B'}
        current_role = None
        for seg in segments:
            if seg in role_map:
                current_role = role_map[seg]
            elif current_role and seg.strip():
                dialogue.append({'role': current_role, 'response': seg.strip()})
        if not dialogue:
            dialogue = [{'role': 'A', 'response': text.strip()}]
        return dialogue
    
    def predict_proba_wrapper(self, text_samples):
        results = []
        for text in text_samples:
            dialogue = self.convert_text_to_dialogue(text)
            # Here, predictor.predict expects a list of dialogues, so wrap dialogue in a list
            prob = predictor.predict([dialogue], explainer_mode=True)
            # Assuming predictor.predict returns a list with one element for each input dialogue
            results.append(prob[0])
        probs = np.array(results)
        return probs

    def explain(self, dialogues, num_features=NUM_KEYWORDS, mode=0):
        if mode == 0:
            text = ""
            for sentence in dialogues[0]:
                marker = self.INVISIBLE_A if sentence['role'] == 'A' else self.INVISIBLE_B
                text += marker + sentence['response'] + " "
            explainer = LimeTextExplainer(class_names=self.CLASS_NAMES)
            explanation = explainer.explain_instance(
                text,
                self.predict_proba_wrapper,
                num_features=num_features,
                num_samples=self.NUM_SAMPLES
            )
            return explanation.as_list()
        elif mode == 1:
            dialogue_xs = dialogues[0]
            length = len(dialogue_xs)
            if length == 0:
                return []
            i = 0
            curr_dialogue = [dialogue_xs[i]]
            log = ''
            while i < length:
                curr_dialogue.append(dialogue_xs[i])
                label = self.predictor.predict([curr_dialogue])
                log += f"Until sentence {i}, the label is {label}\n"
                i += 1
            return log
        elif mode == 2:
            true_prob = self.predictor.predict(dialogues, explainer_mode=True)
            true_index = np.argmax(true_prob, axis=1)
            length = len(dialogues[0])
            if length <= self.NUM_KEYSENTENCE:
                return "Dialogue is too short! All sentences are important to the label generated."
            diff_list = []
            res_list = []
            for i in range(length):
                copy_d = dialogues[0].copy()
                del copy_d[i]
                new_prob = self.predictor.predict([copy_d], explainer_mode=True)
                diff = np.abs(true_prob[0][true_index] - new_prob[0][true_index]) 
                if diff_list == []:
                    flag = True
                else:
                    if len(diff_list) < self.NUM_KEYSENTENCE:
                        flag = True
                    else:
                        flag = False
                    for dif in diff_list:
                        if diff > dif:
                            flag = True
                            break
                if flag:
                    diff_list.append(diff)
                    res_list.append(i)
                    if len(diff_list) > self.NUM_KEYSENTENCE:
                        min_val = min(diff_list)
                        res_list.remove(res_list[diff_list.index(min_val)])
                        diff_list.remove(min_val)
            res_sentences = list(map(lambda x: dialogues[0][x], res_list))
            res = []
            for i in range(len(res_sentences)):
                res.append((res_sentences[i], -float(diff_list[i][0])))
            return res


if __name__ == '__main__':
    def extract_dialogues(cell):
        try:
            parsed = ast.literal_eval(cell)  # safely parse the string into a dict
            return [parsed['text']]            # extract the 'text' field (the list of turns)
        except Exception as e:
            print(f"Failed to parse: {cell}\nError: {e}")
            return None
    predictor = LstmPredictor()
    Lime = LimeExplainer(predictor)
    # dialogues = [
#        [
#  {"role": "A", "response": "Oh, hi."},
#  {"role": "B", "response": "Good afternoon. So...hi...uh... I was wondering if you had plans for dinner."},
#  {"role": "A", "response": "Uh, you mean dinner tonight?"},
#  {"role": "B", "response": "There is an inherent ambiguity in theword 'dinner', technically it refers to the largest meal of the day wheneverit's consumed. So to clarify here, by dinner I mean supper."},
#  {"role": "A", "response": "Supper?"},
#  {"role": "B", "response": "Or dinner. I was thinking 6:30 if you can go.Or a different time."},
#  {"role": "A", "response": "Uh, 6:30 is great."},
#  {"role": "B", "response": "Really? Great."},
#  {"role": "A", "response": "Yeah, I like hanging out with you guys."},
#  {"role": "B", "response": "Us guys?"},
#  {"role": "A", "response": "Yeah, you know, your friends."},
#  {"role": "B", "response": "They might... be there."},
#  {"role": "A", "response": "Okay whatever. It sounds like fun."},
#  {"role": "B", "response": "Great. Did we say a time?"},
#  {"role": "A", "response": "6:30."},
#  {"role": "B", "response": "And that's still good for you?"},
#  {"role": "A", "response": "It's fine."},
#  {"role": "B", "response": "Cause it's not carved in stone."},
#  {"role": "A", "response": "No, 6:30 is great."},
#  {"role": "B", "response": "I'll get my chisel."},
#  {"role": "A", "response": "Why?"},
#  {"role": "B", "response": "To...carve the...I'll see you at 6:30."}
#]
#]
    # print(Lime.explain(dialogues, mode=0)) # word_level: mode=0; sentence_level_sequential: mode=1; sentence_level_limelike: mode=2
    # print(Lime.explain(dialogues, mode=1))
    # print(Lime.explain(dialogues, mode=2))
    # print(predictor.predict(dialogues, explainer_mode=True))
    df = pd.read_csv("Data/train.csv")
    dialogues = df["Dialogue"].apply(extract_dialogues).to_list()
    labels = df["Label"].tolist()
    words_1 = []
    words_0 = []
    words_n1 = []

    # 1
    #for i in range(0, 100):
    #    print(i)
    #    random_int = random.randint(0, 1000)
    #    try:
    #        words = [item[0] for item in Lime.explain(dialogues[random_int], mode=0) if item[1] <= 0]
    #        if predictor.predict(dialogues[i])[0] == 1:   
    #            words_1 += words
    #        elif predictor.predict(dialogues[i])[0] == 0:
    #            words_0 += words
    #        else:
    #            words_n1 += words
    #    except:
    #        print("Error in prediction")
    #        continue
    #text_1 = ' '.join(words_1)
    #if len(words_1) != 0:
    #    wordcloud_1 = WordCloud(width=800, height=400, background_color='white', max_words=300, min_font_size=5).generate(text_1)
    #    output_path_1 = 'wordcloud_1_test.png'
    #    wordcloud_1.to_file(output_path_1)


    # 0
    for j in range(0, 300):
        print(j)
        random_int = random.randint(1000, 2000)
        try:
            words = [item[0] for item in Lime.explain(dialogues[random_int], mode=0) if item[1] <= 0]
            if predictor.predict(dialogues[j])[0] == 1:   
                words_1 += words
            elif predictor.predict(dialogues[j])[0] == 0:
                words_0 += words
            else:
                words_n1 += words
        except:
            print("Error in prediction")
            continue
    text_0 = ' '.join(words_0)
    if len(words_0) != 0:
        wordcloud_0 = WordCloud(width=800, height=400, background_color='white', max_words=300, min_font_size=5, colormap='PuBu').generate(text_0)
        output_path_0 = 'wordcloud_0_test.png'
        wordcloud_0.to_file(output_path_0)

    # -1
    for k in range(0, 300):
        print(k)
        random_int = random.randint(2000, 3000)
        try:
            words = [item[0] for item in Lime.explain(dialogues[random_int], mode=0) if item[1] <= 0]
            if predictor.predict(dialogues[k])[0] == 1:   
                words_1 += words
            elif predictor.predict(dialogues[k])[0] == 0:
                words_0 += words
            else:
                words_n1 += words
        except:
            print("Error in prediction")
            continue
    text_n1 = ' '.join(words_n1)
    if len(words_n1) != 0:
        wordcloud_n1 = WordCloud(width=800, height=400, background_color='white', max_words=300, min_font_size=5, colormap='hot').generate(text_n1)
        output_path_n1 = 'wordcloud_n1_test.png'
        wordcloud_n1.to_file(output_path_n1)
    