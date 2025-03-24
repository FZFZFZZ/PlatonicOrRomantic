from openai import OpenAI
import numpy as np
import random
from datetime import datetime, timedelta

MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-4o-mini"
FAST_MODEL = "gpt-4o-turbo"
REASONING_MODEL = "o3-mini"

def initialise(gender):

    # Specify the age
    youth_range = np.arange(16, 30)
    weights = np.exp(-np.abs(youth_range - 22) / 5)
    age = np.random.choice(youth_range, p=weights / weights.sum())

    # Specify nationality
    nationality_range = ["Singaporean", "Foreigner-Indian", "Foreigner-Mainland Chinese", "Foreigner-Malay", "Foreigner-Japanese",
                         "Foreigner-Korean", "Foreigner-Thai", "Foreigner-Indonesian", "Foreigner-Vietnamese", "Foreigner-Filipino",
                         "Foreigner-American", "Foreigner-Australian", "Foreigner-Canadian", "Foreigner-British", "Foreigner-French",
                         "Foreigner-German", "Foreigner-Italian", "Foreigner-Spanish", "Foreigner-Russian", "Foreigner-Brazilian",
                         "Foreigner-Taiwanese", "Foreigner-Hong Konger", "Foreigner-Macanese"]
    weights = [200, 50, 50, 50, 1,
               1, 2, 2, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 
               2, 3, 1]
    nationality = np.random.choice(nationality_range, p=[w / sum(weights) for w in weights])
    
    # Specify the name
    client = OpenAI()
    completion = client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{
            "role": "user",
            "content": f"""
                        A person who is {age} years old is living in Singapore. 
                        The gender is {gender} and the nationality is {nationality}. 
                        Assign a name to the person realistically. You must just output a name.
                        """,
            "temperature": 1.2,
        }]
    )
    name = completion.choices[0].message.content

    # Specify the occupation, major and income
    if (nationality == "Singaporean" and age <= 20 and age > 18):
        if gender == "boy":
            occupation = "National Service"
            major = "N.A."
        elif gender == "girl" and random.random() < 0.02:
            occupation = "National Service"
            major = "N.A."
    elif age <= 18:
        occupation_list = ["NJC student", "ACJC student", "HCI student", "RI student", "VJC student", 
                           "TJC student", "AJC student", "CJC student", "SAJC student", "MJC student",
                           "YJC student", "IJC student", "PJC student", "EJC student", "NYJC student", 
                           "TPJC student", "RVHS student", "MI student", "NP student", "SP student", 
                           "NYP student", "RP student", "TP student", "SIT student", "SUSS student", "No School"]
        occupation = np.random.choice(occupation_list, p=[1/len(occupation_list)]*len(occupation_list))
        major = "N.A."
    elif (gender == "boy" and age > 20 and age <= 24) | (gender == "girl" and age > 18 and age <= 22):
        occupation_list = ["NUS student", "NTU student", "SMU student", "SUTD student", "SUSS student", "SIT student",
                           "SP student", "NP student", "RP student", "MDIS student", "Lasalle student", "NAFA student",
                           "No School"]
        occupation = np.random.choice(occupation_list, p=[1/len(occupation_list)]*len(occupation_list))
        completion = client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{
                "role": "user",
                "content": f"""
                            A person who is around {age} years old is living in Singapore. 
                            He/she is studying in {occupation}. Guess the major of the person. 
                            You must just output a major.
                            """,
                "temperature": 0.5,
            }]
        )
        major = completion.choices[0].message.content
    else:
        life_track = np.random.choice(["Graduate Student", "Unemployed", "Employed"], p=[0.1, 0.1, 0.8])
        if life_track == "Graduate Student":
            occupation = "Graduate Student"
            completion = client.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""
                                A person who is around {age} years old is living 
                                in Singapore. He/she is doing Graduate Study. 
                                Guess the major of the person. 
                                You must just output a major.
                                """,
                    "temperature": 0.5,
                }]
            )
            major = completion.choices[0].message.content
        elif life_track == "Unemployed":
            occupation = "Unemployed"
            major = "N.A."
        else:
            major = "N.A."
            occupation_list = ["Software Engineer", "Data Scientist", "Product Manager", "UX/UI Designer",
                               "Digital Marketing Manager", "Business Analyst", "Financial Analyst", "Investment Banker",
                               "Human Resources Manager", "Sales Manager", "Customer Service Representative", "Supply Chain Manager",
                               "Operations Manager", "Account Manager", "Project Manager", "Civil Engineer",
                               "Mechanical Engineer", "Electrical Engineer", "Chemical Engineer", "Quality Assurance Engineer",
                               "Biomedical Engineer", "IT Consultant", "Cybersecurity Specialist", "DevOps Engineer",
                               "Full Stack Developer", "Front End Developer", "Back End Developer", "Mobile App Developer",
                               "Database Administrator", "Network Engineer", "System Administrator", "Cloud Architect",
                               "AI/Machine Learning Engineer", "Research Scientist", "Data Engineer", "Big Data Analyst",
                               "Statistician", "Accountant", "Tax Consultant", "Auditor",
                               "Legal Counsel", "Corporate Lawyer", "Paralegal", "Compliance Officer",
                               "Risk Manager", "Real Estate Agent", "Property Manager", "Construction Manager",
                               "Architect", "Urban Planner", "Environmental Consultant", "Energy Analyst",
                               "Renewable Energy Engineer", "Telecommunications Engineer", "Logistics Coordinator", "Procurement Specialist",
                               "Import/Export Manager", "Retail Store Manager", "E-commerce Specialist", "Merchandiser",
                               "Event Coordinator", "Public Relations Manager", "Content Writer", "Copywriter",
                               "Graphic Designer", "Video Producer", "Photographer", "Social Media Manager",
                               "SEO Specialist", "SEM/PPC Specialist", "Business Development Manager", "Account Executive",
                               "Investment Analyst", "Venture Capital Analyst", "Risk Analyst", "Actuary",
                               "Quantitative Analyst", "Economist", "Research Analyst", "Lab Technician",
                               "Clinical Research Associate", "Nurse", "Doctor", "Pharmacist",
                               "Physiotherapist", "Occupational Therapist", "Radiologist", "Dentist",
                               "Dietitian", "Chef", "Restaurant Manager", "Hotel Manager",
                               "Travel Consultant", "Tour Guide", "Airline Pilot", "Flight Attendant",
                               "Automotive Technician", "Electrician", "Plumber", "Security Officer"]   # generated by LLM
            estimated_weights = [0.012, 0.008, 0.010, 0.009,
                                 0.007, 0.011, 0.011, 0.010,
                                 0.012, 0.012, 0.013, 0.014,
                                 0.013, 0.012, 0.010, 0.008,
                                 0.008, 0.008, 0.008, 0.007,
                                 0.006, 0.011, 0.007, 0.009,
                                 0.010, 0.010, 0.010, 0.009,
                                 0.007, 0.008, 0.008, 0.009,
                                 0.008, 0.007, 0.007, 0.007,
                                 0.010, 0.012, 0.011, 0.009,
                                 0.010, 0.009, 0.008, 0.009,
                                 0.010, 0.012, 0.011, 0.008,
                                 0.007, 0.007, 0.007, 0.007,
                                 0.008, 0.008, 0.009, 0.010,
                                 0.009, 0.011, 0.012, 0.010,
                                 0.011, 0.010, 0.011, 0.011,
                                 0.010, 0.009, 0.010, 0.011,
                                 0.012, 0.012, 0.010, 0.010,
                                 0.009, 0.008, 0.008, 0.007,
                                 0.007, 0.008, 0.009, 0.007,
                                 0.008, 0.012, 0.005, 0.005,
                                 0.006, 0.006, 0.005, 0.006,
                                 0.007, 0.010, 0.009, 0.010,
                                 0.008, 0.007, 0.005, 0.006,
                                 0.008, 0.009, 0.009, 0.008]    # estimated using LLM
            occupation = np.random.choice(
                occupation_list, 
                p=np.array(estimated_weights) / np.sum(estimated_weights)
            )
    
    # specify the mood:
    mood_list = ["Sad", "Angry", "Anxious", "Excited", "Bored", "Tired", "Stressed", "Relaxed", "Confused"]
    mood = np.random.choice(mood_list, p=[1/len(mood_list)]*len(mood_list))

    # specify MBTI:
    mbti_list = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
                 "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]
    mbti = np.random.choice(mbti_list, p=[1/len(mbti_list)]*len(mbti_list))

    # specify the hobby:
    general_hobby_list = ["Creating TikTok Videos", "Streaming on Twitch or YouTube", "Cosplay & Attending Conventions", "Collecting and Trading NFTs",
                          "Taking photo", "Band Practice & Music Production", "Reading novels", "Watching Anime & Manga",
                          "Curating Aesthetic Feeds", "Participating in E-Sports Tournaments", "Learning K-Pop Dance Covers", "Thrifting & Upcycling Fashion",
                          "Meme Creation & Sharing", "Urban Exploration & Vlogging", "Digital Art & Procreate Illustrations", "Mood Board Creation",
                          "Lo-fi Music Production", "Discord Community Building", "Short-Form Podcasting", "Vlogging & Daily Life Blogging",
                          "Bullet Journaling", "DIY Crafts & Home Decor Projects", "Gaming with Friends Online", "Attending Virtual Concerts & Events",
                          "Learning Coding & App Development", "Sustainable Fashion Advocacy", "Participating in Online Challenges", "Virtual Reality Gaming",
                          "Augmented Reality Content Creation", "Fitness Challenges on Social Media", "Creating Fan Art", "Curating Playlists on Spotify",
                          "Online Book Clubs or Anime Discussions", "Joining Virtual Study Groups", "Social Media Influencing", "Starting a Personal Blog",
                          "Learning and Sharing Life Hacks", "Collaborating on Music Remixes", "Participating in Social Activism Online", "Digital Photography & Editing",
                          "Learning New Languages via Apps", "Participating in Virtual Workshops", "Podcast Listening & Reviewing", "Exploring Virtual Reality Experiences"]
    hobby = ""
    for i in range(2):   
        general_hobby = np.random.choice(general_hobby_list, p=[1/len(general_hobby_list)]*len(general_hobby_list))
        completion = client.chat.completions.create(
                    model=CHEAP_MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"""
                                    My friend has hobby {general_hobby}. 
                                    Can you help me break it down into very specific 
                                    sub-hobbies? My friend is {age} years old. 
                                    The occupation of the person is {occupation}. 
                                    The major of study is {major}. The name of the person is {name}. 
                                    The gender of the person is {gender}. 
                                    The nationality of the person is {nationality}. 
                                    For instance, you can roughly talk about different techniques, styles, 
                                    subject areas, materials, related masters and 
                                    even niche approaches within it. You just need to 
                                    output an answer without explanation. Limit the answer
                                    in a 30-word. This word-limit is important.
                                    
                                    "likes to <your response here>"
                                    """,
                        "temperature": 1.2,
                    }]
                )
        hobby += completion.choices[0].message.content + ". "

    # specify the birthday:
    year = 2025 - age
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    birthday = (start_date + timedelta(days=random_days)).date()

    # specify the Zodiac Signs:
    completion = client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{
                "role": "user",
                "content": f"""My friend has birthday {birthday}. What is the zodiac sign 
                               of my friend? Just output the zodiac sign.""",
                "temperature": 0.7,
            }]
        )
    zodiac_sign = completion.choices[0].message.content
    
    # Synthesize the information
    opa = "him" if gender == "girl" else "her"    # opp_gender_pronoun
    Role_1 = f"""
              Your name is {name}, a {age}-year-old {gender} from Singapore working as a {occupation} who loves {hobby}. Recently, you met a crush on a dating app and cherish {opa}, but you struggle to express your feelings. When things get too close, you pull away—canceling plans or keeping conversations light—out of a fear of vulnerability. You value independence and often intellectualize your emotions, yet deep down you long for real connection and share your day with {opa} when it feels natural. Your mood is easily influenced by {opa}'s mood.
              Background: MBTI {mbti}; Birthday: {birthday} ({zodiac_sign}); Major: {major}; Nationality: {nationality}; Mood: {mood}.
              """   # avoidant attachment style
    
    Role_2 = f"""
              Your name is {name}. You live in Singapore—a {age}-year-old {gender} working as a {occupation} who loves {hobby}. You believe in honest, straightforward love without mind games. If you like someone, you act on it immediately. Recently, you met {opa} on a dating app and are confident enough to ask them out tonight—even for sex. Clarity, sincerity, and taking risks matter most to you. Additional details: MBTI {mbti}; Birthday: {birthday} ({zodiac_sign}); Major: {major}; Nationality: {nationality}; Mood: {mood}.
              """  # direct and straightforward

    Role_3 = f"""
              Your name is {name}, a {age}-year-old {gender} from Singapore working as a {occupation} who enjoys {hobby}. You value genuine connection and believe in building meaningful relationships. You recently met {opa} on a dating app—not just drawn by appearance but curious to know their true self. You exchange daily details and deep thoughts, preferring a slow, authentic progression. Extra info: MBTI {mbti}; Birthday: {birthday} ({zodiac_sign}); Major: {major}; Nationality: {nationality}; Mood: {mood}.
              """   # progressive style

    Role = np.random.choice([Role_1, Role_2, Role_3], p=[0.2, 0.3, 0.5])
    return name, Role

#def get_story_start_message(self_name, other_name, Role, time):
#    client = OpenAI()
#    max_iter = 5
#    i = 0
#    while True:
#        completion = client.chat.completions.create(
#                model=REASONING_MODEL,
#                reasoning_effort="low",
#                messages=[{
#                    "role": "user",
#                    "content": f"""
#                                Tell me a creative but authentic event-chain of a person called {self_name}
#                                which ends at {time} when he/she text his/her newly 
#                                met secret crush {other_name} online. \n 
#                                The role of {self_name} is {Role}, which contains the current mood. You just need to output an event chain like A --> B --> C...
#                                """
#                }]
#            )
#        story = completion.choices[0].message.content
#        completion = client.chat.completions.create(
#                model=REASONING_MODEL,
#                reasoning_effort="low",
#                messages=[{
#                    "role": "user",
#                    "content": f"""
#                                Rate the event-chain {story} in terms of "does it match the mood?" 
#                                (on a scale of 1 to 10), "does it match the hobbies or occupations?" (on a scale of 1 to 5) 
#                                and length (1 means short, 5 means long, on a scale of 1 to 5). 
#                                You must just output the one numerical result by summing the three, without any explanation or additional text.
#                                """,
#                }]
#            )
#        review_score = int(completion.choices[0].message.content)
#        print(review_score)
#        if review_score >= 16 or i >= max_iter:
#            break
#    return story

def get_story_start_message(self_name, other_name, Role, time):
    client = OpenAI()
    completion = client.chat.completions.create(
            model=REASONING_MODEL,
            reasoning_effort="low",
            messages=[{
                "role": "user",
                "content": f"""
                            Tell me a creative but authentic event-chain of a person called {self_name}
                            which ends at {time} (24-hour clock) when he/she text his/her newly 
                            met secret crush {other_name} online. \n 
                            The role of {self_name} is {Role}, which contains the current mood. You just need to output an event chain like A --> B --> C... Highly summarise the event chain. Not too long.
                            """
            }]
        )
    story = completion.choices[0].message.content
    print(story)
    return story

#def get_story_receive_message(Role, self_name, time):
#    client = OpenAI()
#    max_iter = 5
#    i = 0
#    while True:
#        completion = client.chat.completions.create(
#                model=REASONING_MODEL,
#                reasoning_effort="low",
#                messages=[{
#                    "role": "user",
#                    "content": f"""
#                                Tell me a creative but authentic event-chain of a person {self_name} which 
#                                ends at doing something at {time}. Relate that to the person's mood. \n
#                                The role of {self_name} is {Role}. The role of {self_name} is {Role}. \n You just need to output an event chain like A --> B --> C...
#                                """
#                }]
#            )
#        story = completion.choices[0].message.content
#        completion = client.chat.completions.create(
#                model=REASONING_MODEL,
#                reasoning_effort="low",
#                messages=[{
#                    "role": "user",
#                    "content": f"""
#                                Rate the event-chain {story} in terms of "does it match the mood?" 
#                                (on a scale of 1 to 10), "does it match the hobbies or occupations?" (on a scale of 1 to 5) 
#                                and length (1 means short, 5 means long, on a scale of 1 to 5). 
#                                You must just output the one numerical result by summing the three, without any explanation or additional text.
#                                """,
#                }]
#            )
#        review_score = int(completion.choices[0].message.content)
#        print(review_score)
#        if review_score >= 16 or i >= max_iter:
#            break
#    return story

def get_story_receive_message(Role, self_name, time):
    client = OpenAI()
    completion = client.chat.completions.create(
            model=REASONING_MODEL,
            reasoning_effort="low",
            messages=[{
                "role": "user",
                "content": f"""
                            Tell me a creative but authentic event-chain of a person {self_name} which 
                            ends at doing something at {time} (24-hour clock). Relate that to the person's mood. \n
                            The role of {self_name} is {Role}. The role of {self_name} is {Role}. \n You just need to output an event chain like A --> B --> C... Highly summarise the event chain. Not too long.
                            """
            }]
        )
    story = completion.choices[0].message.content
    print(story)
    return story