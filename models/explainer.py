import numpy as np
import re
from lime.lime_text import LimeTextExplainer
from .predictor import LstmPredictor

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
            prob = self.predictor.predict([dialogue], explainer_mode=True)
            # Assuming predictor.predict returns a list with one element for each input dialogue
            results.append(prob[0])
        probs = np.array(results)
        return probs

    def explain(self, dialogue: list, num_features=NUM_KEYWORDS, mode=0):
        if mode == 0:
            text = ""
            for sentence in dialogue:
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
            dialogue_xs = dialogue
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
            true_prob = self.predictor.predict([dialogue], explainer_mode=True)
            true_index = np.argmax(true_prob, axis=1)
            length = len(dialogue)
            if length <= self.NUM_KEYSENTENCE:
                return "Dialogue is too short! All sentences are important to the label generated."
            diff_list = []
            res_list = []
            for i in range(length):
                copy_d = dialogue.copy()
                del copy_d[i]
                new_prob = self.predictor.predict([copy_d], explainer_mode=True)
                diff = np.abs(true_prob[0][true_index] - new_prob[0][true_index]) 
                if diff_list == []:
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
            res_sentences = list(map(lambda x: dialogue[x], res_list))
            res = []
            for i in range(len(res_sentences)):
                res.append((res_sentences[i], -float(diff_list[i][0])))
            return res


if __name__ == '__main__':
    predictor = LstmPredictor()
    Lime = LimeExplainer(predictor)
    dialogue = \
        [
            {
                "role": "A",
                "response": "Hey Sofia! 🎨😄 Just tried some cool surrealism-inspired vector art today. What's up with you?"
            },
            {
                "role": "B",
                "response": "Hey Kevin! Just uploaded my digital art from a VR concert. Week's been a whirlwind lol."
            },
            {
                "role": "A",
                "response": "A VR concert sounds wild! I bet it was visually awesome. What music were they playing?"
            },
            {
                "role": "B",
                "response": "Oh, loads of sick pop, EDM vibes! You should join next time to experience it too."
            },
            {
                "role": "A",
                "response": "Pop and EDM! Sounds like a vibrant mix. Which artwork did you create for it?"
            },
            {
                "role": "B",
                "response": "It’s a vivid virtual world inspired by my concert vibes and playlists. Hope you like it!"
            },
            {
                "role": "A",
                "response": "That sounds incredible! I'd love to see more of your VR artwork sometime."
            },
            {
                "role": "B",
                "response": "You should totally swing by a VR event or join my online community!"
            },
            {
                "role": "A",
                "response": "I've never been to a VR event. Got tips for a newcomer?"
            },
            {
                "role": "B",
                "response": "Start with an EDM themed one—gets you vibing instantly! 😊"
            },
            {
                "role": "A",
                "response": "EDM isn't usually my thing, any relaxing events to start with maybe?"
            },
            {
                "role": "B",
                "response": "Try a chillout session, maybe a serene studio virtual tour!"
            },
            {
                "role": "A",
                "response": "A serene studio tour sounds perfect! Count me in. 😊"
            },
            {
                "role": "B",
                "response": "I'll keep you updated whenever there's one you might like!"
            },
            {
                "role": "A",
                "response": "Thanks! Can't wait to discover this new realm within our art world."
            },
            {
                "role": "B",
                "response": "Our virtual art glide awaits, hype's inevitable! 😊🤗"
            },
            {
                "role": "A",
                "response": "Let's dive in soon, curious about how it merges with digital art!"
            },
            {
                "role": "B",
                "response": "$S$"
            },
            {
                "role": "A",
                "response": "Hypothetical question, ever tried mixing other musical perspectives digitally?"
            },
            {
                "role": "B",
                "response": "Mostly on pop influences, any element you'd suggest?"
            }
        ]
    print(Lime.explain(dialogue, mode=2)) # word_level: mode=0; sentence_level_sequential: mode=1; sentence_level_limelike: mode=2
