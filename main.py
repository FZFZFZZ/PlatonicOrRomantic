from openai import OpenAI
import numpy as np
import logging
import Data.platonic as platonic
import Data.romantic as romantic
import json
from datetime import datetime, timedelta
import random
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHATTER_A = "Platonic"
CHATTER_B = "Platonic"

GENDER_A = "boy"
GENDER_B = "girl"

# SEED = 7659098
MAX_GENERATION = 46
MAX_CHAT_ROUND = 20

DATA_PATH = "Data/train.csv"

MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-4o-mini"

def generate_random_time():
    start_date = datetime(2025, 1, 1, 0, 0)
    end_date = datetime(2025, 12, 31, 23, 59)
    delta_seconds = int((end_date - start_date).total_seconds())
    random_seconds = random.randint(0, delta_seconds)
    random_datetime = start_date + timedelta(seconds=random_seconds)
    return random_datetime.strftime("%m-%d %H:%M")

def clean(output):
    if output.startswith("```json"):
        lines = output.splitlines()
        json_str = "\n".join(lines[1:-1])
    else:
        json_str = output
    return json_str

def main():
    # np.random.seed(SEED)
    # random.seed(SEED)
    client = OpenAI()
    client_deepseek = OpenAI(api_key="sk-85a6104c32ca417382a57671b3e26fc7", base_url="https://api.deepseek.com")

    # get label
    if CHATTER_A == "Platonic" and CHATTER_B == "Platonic":
        label = "-1"
    elif CHATTER_A == "Romantic" and CHATTER_B == "Romantic":
        label = "1"
    else:
        label = "0"

    # Format prompt and task prompt
    task_p = open("task_prompt.txt", "r").read()
    format_p = open("format_prompt.txt", "r").read()
    
    i = 0
    while i < MAX_GENERATION:
        try:
            logging.info(f"Generation {i}")

            # initialise empty dialogue dict and empty other's info
            logging.info("Initialising dialogue and other's info")
            dialogue = {
                "text" : []
            }
            if CHATTER_A == "Platonic":
                B_info_for_A_to_infer = {
                    "name": "",
                    "gender": "",
                    "occupation": "",
                    "hobby": "",
                    "intention": ""
                }
            else:
                B_info_for_A_to_infer = {
                    "name": "",
                    "gender": "",
                    "occupation": "",
                    "major": "",
                    "mbti": "",
                    "zodiac": "",
                    "age": "",
                    "mood": "",
                    "hobby": "",
                    "intention": "",
                    "birthday": ""
                }
            if CHATTER_B == "Platonic":
                A_info_for_B_to_infer = {
                    "name": "",
                    "gender": "",
                    "occupation": "",
                    "hobby": "",
                    "intention": ""
                }
            else:
                A_info_for_B_to_infer = {
                    "name": "",
                    "gender": "",
                    "occupation": "",
                    "major": "",
                    "mbti": "",
                    "zodiac": "",
                    "age": "",
                    "mood": "",
                    "hobby": "",
                    "intention": "",
                    "birthday": ""
                }

            # Generate a time when the conversation happens
            time = generate_random_time()

            # who is starting the conversation
            curr = "A" if np.random.rand() < 0.5 else "B"
            logging.info(f"{curr} is starting the conversation!")

            # Initialise A & B
            logging.info(f"Initialising character A")
            if CHATTER_A == "Platonic":
                name_A, Role_A = platonic.initialise(GENDER_A)
            else:
                name_A, Role_A = romantic.initialise(GENDER_A)
            print(name_A, Role_A)
            logging.info(f"Initialising character B")
            if CHATTER_B == "Platonic":
                name_B, Role_B = platonic.initialise(GENDER_B)
            else:
                name_B, Role_B = romantic.initialise(GENDER_B)
            print(name_B, Role_B)

            # Prefill Other's info
            logging.info("Prefilling other's info")
            completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"""I have the following dictionary template for storing inferred information about Person B:\n {B_info_for_A_to_infer} \n
                                       Below is a piece of information about Person B: {Role_B} \n
                                       Please fill in only the "name" and "occupation" fields based on the provided information. Do not modify other fields; leave them as empty strings. Your output must be a JSON. You must follow the template exactly without any changes in structure or additional keys.""",
                        "temperature": 0.5,
                    }]
                )
            output = completion.choices[0].message.content
            output = clean(output)
            result_dict = json.loads(output)
            B_info_for_A_to_infer.update(result_dict)
            completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"""I have the following dictionary template for storing inferred information about Person A:\n {A_info_for_B_to_infer} \n
                                       Below is a piece of information about Person B: {Role_A} \n
                                       Please fill in only the "name" and "occupation" fields based on the provided information. Do not modify other fields; leave them as empty strings. Your output must be a JSON. You must follow the template exactly without any changes in structure or additional keys.""",
                        "temperature": 0.5,
                    }]
                )
            output = completion.choices[0].message.content
            output = clean(output)
            result_dict = json.loads(output)
            A_info_for_B_to_infer.update(result_dict)        

            # Initialise event chain (why the conversation happens)
            logging.info("Initialising event chain")
            if CHATTER_A == "Platonic":
                if curr == "A":
                    event_chain_A = platonic.get_story_start_message(name_A, name_B, Role_A, time)
                else:
                    event_chain_A = platonic.get_story_receive_message(Role_A, name_A, time)
            else:
                if curr == "A":
                    event_chain_A = romantic.get_story_start_message(name_A, name_B, Role_A, time)
                else:
                    event_chain_A = romantic.get_story_receive_message(Role_A, name_A, time)

            if CHATTER_B == "Platonic":
                if curr == "B":
                    event_chain_B = platonic.get_story_start_message(name_B, name_A, Role_B, time)
                else:
                    event_chain_B = platonic.get_story_receive_message(Role_B, name_B, time)
            else:
                if curr == "B":
                    event_chain_B = romantic.get_story_start_message(name_B, name_A, Role_B, time)
                else:
                    event_chain_B = romantic.get_story_receive_message(Role_B, name_B, time)

            chat_round = 0
            while True:
                logging.info(f"Chat round {chat_round}")
                if chat_round >= MAX_CHAT_ROUND:
                    logging.info("Max chat round reached!")
                    break

                # synthesise prompt for this round
                curr_role = Role_A if curr == "A" else Role_B
                curr_event_chain = event_chain_A if curr == "A" else event_chain_B
                curr_info_to_infer = A_info_for_B_to_infer if curr == "B" else B_info_for_A_to_infer
                dialogue["text"].append({"role": curr, "response": ""})
                p = ("<Role>" + curr_role + "/<Role>\n" + "<event>" + curr_event_chain + "/<event>\n" + 
                     "<dialogue>" + json.dumps(dialogue) + "</dialogue>\n"
                     + "<other's info>" + json.dumps(curr_info_to_infer) + "</other's info>")
                
                i=0
                while i < 10:
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "developer", "content": task_p + "\n" + format_p},
                                {"role": "user", "content": p}
                            ],
                            temperature=1.3
                        )
                        response = completion.choices[0].message.content
                        response = clean(response)
                        prev_dialogue = dialogue
                        dialogue = json.loads(response)

                        # If the conversation ends, break out of the loop.
                        if dialogue["text"][-1]["response"] == "$EXIT$":
                            logging.info("Conversation ended!")

                        break
                    except Exception as e:
                        logging.error("Error encountered: %s. Regenerating that response...", e)
                        dialogue = prev_dialogue
                        i+=1
                        continue
                        
                if dialogue["text"][-1]["response"] == "$EXIT$":
                    logging.info("Conversation ended!")
                    break
                
                # update other's info
                try:
                    if curr == "A":
                        p_update_other = "<dialogue>" + json.dumps(dialogue) + "</dialogue>\n" + "<other's info>" + json.dumps(B_info_for_A_to_infer) + "</other's info>"
                        completion = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                            {"role": "developer", "content": "Given a dialogue <dialogue> containing B's message, update the dictionary template  <other's info> </other's info> for storing inferred information about Person B based on the dialogue. You can leave one empty or unchanged. But try your best to infer the 'intention', if there is such key. Your output must follow the template exactly without any changes in structure or additional keys. Output a JSON object."},
                            {
                                "role": "user",
                                "content": p_update_other
                            }],
                            temperature=0.7
                        )
                        response = completion.choices[0].message.content
                        response = clean(response)
                        B_info_for_A_to_infer = json.loads(response)
                    else:
                        p_update_other = "<dialogue>" + json.dumps(dialogue) + "</dialogue>\n" + "<other's info>" + json.dumps(A_info_for_B_to_infer) + "</other's info>"
                        completion = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                            {"role": "developer", "content": "Given a dialogue <dialogue> containing A's message, update the dictionary template <other's info> </other's info> for storing inferred information about Person A based on the dialogue. You can leave one empty or unchanged. But try your best to infer the 'intention', if there is such key. Your output must follow the template exactly without any changes in structure or additional keys. Output a JSON object."},
                            {
                                "role": "user",
                                "content": p_update_other
                            }],
                            temperature=0.7
                        )
                        response = completion.choices[0].message.content
                        response = clean(response)
                        A_info_for_B_to_infer = json.loads(response)
                except Exception as e:
                    logging.info("Fail to update. Continue")
                    continue

                # switch role
                curr = "A" if curr == "B" else "B"
                chat_round += 1

            # retrieve the previous ID
            df = pd.read_csv(DATA_PATH)
            if not df.empty:
                new_id = df.iloc[-1]['ID'] + 1
            else:
                new_id = 0

            # add new instance to the dataset
            new_instance = pd.DataFrame({
                'ID': [new_id],
                'Dialogue': [dialogue],
                'Label': [label]
            })
            new_instance.to_csv(DATA_PATH, mode='a', header=False, index=False)

            # occasionally print out the dialogue
            if i % 1 == 0:
                logging.info(f"Dialogue: \n{dialogue}")
            i += 1
        except Exception as e:
            logging.error(f"Error: {e}. Discard current instance. Regenerate the current one.")
            continue
    
    logging.info("Finished generating all data!")
        

if __name__ == "__main__":
    main()