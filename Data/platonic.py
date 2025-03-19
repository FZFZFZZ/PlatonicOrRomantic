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
    weights = [40, 10, 10, 10, 1,
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

    # Specify the occupation & major
    if (nationality == "boy" and age <= 20 and age > 18):
        if gender == "male":
            occupation = "National Service"
            major = "N.A."
        elif gender == "girl" & random.random() < 0.02:
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
                "temperature": 1.0,
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
                    "temperature": 1.0,
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
                                    Can you help me break it down into very specific, 
                                    detailed sub-hobbies? My friend is {age} years old. 
                                    The occupation of the person is {occupation}. 
                                    The major of study is {major}. The name of the person is {name}. 
                                    The gender of the person is {gender}. 
                                    The nationality of the person is {nationality}. 
                                    For instance, you can roughly talk about different techniques, styles, 
                                    subject areas, materials, related masters and 
                                    even niche approaches within it. You just need to 
                                    output an answer without explanation. Limit the answer
                                    in a 60-word paragraph. 
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
    Role_1 = f"""
              Your name is {name}. You live in Singapore. You are a {age} years old {gender}. You are a {occupation}. You like to {hobby}. 
              Recently, you are using a socialising app and meet a friend. You do not really want to pursue a romantic relationship
              with the friend. Thus, you try your best to avoid being too intimate with the friend (e.g. two people alone).
              But you two still share similar hobbies, which make that your friend indeed. \n
              
              Some more background:
              Your MBTI: {mbti}; Your birthday: {birthday} ({zodiac_sign}); Your major of study: {major};
              Your nationality: {nationality}; Your current mood: {mood}. 
              """
    
    Role_2 = f"""
              Your name is {name}. You live in Singapore. You are a {age} years old {gender}. You are a {occupation}. You like to {hobby}.
              Wow! You have just met a new friend on a socialising app. You treat that as your best brother. 
              You share every shit and jokes with that big homie.
              Which is to say, the relationship is too platonic that 
              you have not even thought about any romantic relationship. \n
              
              Some more background:
              Your MBTI: {mbti}; Your birthday: {birthday} ({zodiac_sign}); Your major of study: {major};
              Your nationality: {nationality}; Your current mood: {mood}. 
              """
    
    Role_3 = f"""
              Your name is {name}. You live in Singapore. You are a {age} years old {gender}. You are a {occupation}. You like to {hobby}.
              You have always felt that romantic and sexual relationships are not really your thing. 
              You enjoy deep friendships and meaningful connections, but the idea of being in a romantic relationship
              or engaging in intimacy has never been appealing to you. 
      
              You recently met a new friend on a socialising app. You enjoy their company and like spending time together, 
              but you view relationships in a way that is purely based on companionship rather than romance or attraction. 
              You are clear about your identity and boundaries, and you feel comfortable expressing them. \n
              
              Some more background:
              Your MBTI: {mbti}; Your birthday: {birthday} ({zodiac_sign}); Your major of study: {major};
              Your nationality: {nationality}; Your current mood: {mood}. 
              """
    
    Role = np.random.choice([Role_1, Role_2, Role_3], p=[0.4, 0.4, 0.2])
    return name, Role

def get_story_start_message(self_name, other_name, Role, time):
    client = OpenAI()
    max_iter = 3
    i = 0
    while True:
        completion = client.chat.completions.create(
                model=REASONING_MODEL,
                reasoning_effort="low",
                messages=[{
                    "role": "user",
                    "content": f"""
                                Tell me a creative but authentic event-chain of a person called {self_name}
                                which ends at {time} when he/she text his/her new normal friend
                                {other_name} online. \n 
                                The role of {self_name} is {Role}, which contains the current mood. \n You just need to output an event chain like A --> B --> C...
                                """
                }]
            )
        story = completion.choices[0].message.content
        completion = client.chat.completions.create(
                model=REASONING_MODEL,
                reasoning_effort="low",
                messages=[{
                    "role": "user",
                    "content": f"""
                                Rate the event-chain {story} in terms of "does it match the mood?" 
                                (on a scale of 1 to 10), "does it match the hobbies or occupations?" (on a scale of 1 to 5) 
                                and length (1 means short, 5 means long, on a scale of 1 to 5). 
                                You must just output the one numerical result by summing the three, without any explanation or additional text.
                                """,
                }]
            )
        review_score = int(completion.choices[0].message.content)
        print(review_score)
        if review_score >= 16 or i >= max_iter:
            break
    return story

def get_story_receive_message(Role, self_name, time):
    client = OpenAI()
    max_iter = 5
    i = 0
    while True:
        completion = client.chat.completions.create(
                model=REASONING_MODEL,
                reasoning_effort="medium",
                messages=[{
                    "role": "user",
                    "content": f"""
                                Tell me a creative but authentic event-chain of a person {self_name} which 
                                ends at doing something at {time}. Relate that to the person's mood. \n
                                The role of {self_name} is {Role}. 
                                """
                }]
            )
        story = completion.choices[0].message.content
        completion = client.chat.completions.create(
                model=REASONING_MODEL,
                reasoning_effort="medium",
                messages=[{
                    "role": "user",
                    "content": f"""
                                Rate the event-chain {story} in terms of "does it match the mood?" 
                                (on a scale of 1 to 10), "does it match the hobbies or occupations?" (on a scale of 1 to 5) 
                                and length (1 means short, 5 means long, on a scale of 1 to 5). 
                                You must just output the one numerical result by summing the three, without any explanation or additional text.
                                """,
                }]
            )
        review_score = int(completion.choices[0].message.content)
        print(review_score)
        if review_score >= 16 or i >= max_iter:
            break
    return story
