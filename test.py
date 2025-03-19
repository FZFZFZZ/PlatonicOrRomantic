from datetime import datetime, timedelta
import random

def generate_random_time():
    start_date = datetime(2025, 1, 1, 0, 0)
    end_date = datetime(2025, 12, 31, 23, 59)
    delta_seconds = int((end_date - start_date).total_seconds())
    random_seconds = random.randint(0, delta_seconds)
    random_datetime = start_date + timedelta(seconds=random_seconds)
    return random_datetime.strftime("%m-%d %H:%M")

print(generate_random_time())