import csv
import random
import re
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
TOTAL_ROWS = 30000
OUTPUT_FILE = "haptic_dataset_v3.csv"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Fixed small output spaces for the three single haptics
INTENTS = ["statement", "question", "request", "alert"]
SUBJECTS = ["me", "you", "other", "none"]
ACTIONS = ["need", "move", "danger", "feel", "communicate", "do", "none"]

# ============================================================
# CONCEPT BANKS
# Concept can be large because it is displayed through Braille
# ============================================================
NEEDS = {
    "water": ["water", "a drink", "something to drink"],
    "food": ["food", "something to eat", "a meal", "a snack"],
    "bathroom": ["bathroom", "restroom", "toilet", "washroom"],
    "medicine": ["medicine", "medication", "meds"],
    "help": ["help", "assistance", "some help"],
    "phone": ["phone", "mobile", "cellphone"],
    "charger": ["charger", "charging cable", "phone charger"],
    "blanket": ["blanket", "cover"],
    "jacket": ["jacket", "coat"],
    "shoes": ["shoes", "footwear"],
    "glasses": ["glasses", "spectacles"],
    "cane": ["cane", "walking cane"],
    "wallet": ["wallet", "purse"],
    "keys": ["keys", "house keys"],
    "ticket": ["ticket", "pass"],
    "chair": ["chair", "seat"],
    "umbrella": ["umbrella", "rain umbrella"],
    "towel": ["towel"],
    "map": ["map"],
    "directions": ["directions", "the way"]
}

HAZARDS = {
    "car": ["car", "vehicle"],
    "bus": ["bus"],
    "truck": ["truck"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle", "motorbike"],
    "pole": ["pole", "post"],
    "stairs": ["stairs", "steps"],
    "curb": ["curb", "edge of the pavement"],
    "hole": ["hole", "pit"],
    "ditch": ["ditch"],
    "fire": ["fire", "flames"],
    "traffic": ["traffic", "moving traffic"],
    "dog": ["dog"],
    "crowd": ["crowd", "group of people"],
    "puddle": ["puddle", "water on the ground"],
    "wet_floor": ["wet floor", "slippery floor"],
    "broken_glass": ["broken glass", "glass on the ground"],
    "open_manhole": ["open manhole", "manhole"],
    "platform_edge": ["platform edge", "edge of the platform"],
    "construction": ["construction area", "construction"]
}

PLACES = {
    "home": ["home", "the house"],
    "school": ["school", "campus"],
    "work": ["work", "the office"],
    "hospital": ["hospital", "clinic"],
    "restaurant": ["restaurant", "cafe"],
    "store": ["store", "shop"],
    "park": ["park"],
    "bus_stop": ["bus stop"],
    "train_station": ["train station", "station"],
    "bathroom": ["bathroom", "restroom"]
}

FEELINGS = {
    "hungry": ["hungry", "starving"],
    "thirsty": ["thirsty", "parched"],
    "tired": ["tired", "exhausted", "sleepy"],
    "cold": ["cold", "freezing"],
    "hot": ["hot", "too warm"],
    "sick": ["sick", "unwell"],
    "scared": ["scared", "afraid"],
    "happy": ["happy"],
    "sad": ["sad", "upset"],
    "dizzy": ["dizzy", "lightheaded"],
    "pain": ["in pain", "hurting"]
}

# ============================================================
# TEMPLATE BANKS
# ============================================================
QUESTION_NEED = [
    "do you need {c}",
    "do you want {c}",
    "would you like {c}",
    "could you use {c}",
    "should i get you {c}",
    "can i get you {c}",
    "shall i bring you {c}",
    "would you like me to get you {c}",
    "are you looking for {c}",
    "were you asking for {c}",
    "is {c} what you need",
    "would {c} help",
    "do you need me to bring you {c}",
    "should i bring you {c}"
]

REQUEST_NEED = [
    "i need {c}",
    "i want {c}",
    "please get me {c}",
    "can you get me {c}",
    "could you bring me {c}",
    "please bring me {c}",
    "i could use {c}",
    "i really need {c}",
    "can i have {c}",
    "get me {c} please",
    "i am looking for {c}",
    "help me find {c}",
    "could i get {c}",
    "please help me get {c}"
]

ALERTS = [
    "watch out for the {c}",
    "look out for the {c}",
    "be careful there is a {c}",
    "careful there is a {c}",
    "there is a {c} ahead",
    "danger there is a {c}",
    "stop there is a {c}",
    "watch your step near the {c}",
    "avoid the {c}",
    "move away from the {c}",
    "stay away from the {c}",
    "there is a {c} coming towards you",
    "a {c} is coming",
    "be careful of the {c}",
    "you are approaching a {c}",
    "there is a {c} in front of you"
]

LOCATION_QUESTIONS = [
    "where is the {c}",
    "where can i find the {c}",
    "do you know where the {c} is",
    "how do i get to the {c}",
    "which way is the {c}",
    "are we near the {c}",
    "is the {c} nearby",
    "how far is the {c}",
    "can you show me where the {c} is"
]

MOVEMENT = [
    "i am going to {c}",
    "i am heading to {c}",
    "take me to {c}",
    "lets go to {c}",
    "i want to go to {c}",
    "we are going to {c}",
    "i need to get to {c}",
    "i am on my way to {c}",
    "lets head to {c}"
]

FEELING_STATEMENTS = [
    "i am {c}",
    "i feel {c}",
    "i am feeling {c}",
    "im really {c}",
    "i feel very {c}",
    "right now i feel {c}",
    "ive been feeling {c}"
]

FEELING_QUESTIONS = [
    "are you {c}",
    "do you feel {c}",
    "are you feeling {c}",
    "do you still feel {c}",
    "have you been feeling {c}",
    "are you feeling very {c}"
]

# These are real conversational families.
# They all map to the limited hardware vocabulary.
CONVERSATION_FAMILIES = [
    # question + you + do
    ([
        "what are you doing",
        "what are you doing right now",
        "what are you up to",
        "what are you busy with",
        "what are you working on",
        "what have you been doing"
    ], "question", "you", "do", "none"),

    # question + none + do
    ([
        "what happened",
        "what is happening",
        "whats happening",
        "whats going on",
        "what just happened"
    ], "question", "none", "do", "none"),

    # question + you + move
    ([
        "where are you going",
        "where are you headed",
        "where are you heading",
        "where are you off to"
    ], "question", "you", "move", "destination"),

    # communication question
    ([
        "what did you say",
        "what are you saying",
        "what do you mean",
        "what were you saying",
        "did you say something"
    ], "question", "you", "communicate", "none"),

    # communication request
    ([
        "can you repeat that",
        "please repeat that",
        "say that again",
        "can you say that again",
        "please say it again",
        "repeat what you said"
    ], "request", "you", "communicate", "none"),

    # understanding
    ([
        "i dont understand",
        "i did not understand",
        "i dont get it",
        "i didnt understand that"
    ], "statement", "me", "communicate", "none"),

    # wait
    ([
        "please wait",
        "wait for me",
        "hold on",
        "give me a moment",
        "wait a second"
    ], "request", "you", "do", "none"),

    # simple movement commands
    ([
        "come here",
        "come over here",
        "follow me",
        "come with me"
    ], "request", "you", "move", "here"),

    # left
    ([
        "turn left",
        "go left",
        "move left",
        "take a left"
    ], "request", "you", "move", "left"),

    # right
    ([
        "turn right",
        "go right",
        "move right",
        "take a right"
    ], "request", "you", "move", "right"),

    # stop as movement, not an alert
    ([
        "please stop",
        "stop here",
        "stop moving",
        "dont move"
    ], "request", "you", "move", "stop"),

    # yes
    ([
        "yes",
        "yeah",
        "yes thats right",
        "correct"
    ], "statement", "me", "communicate", "yes"),

    # no
    ([
        "no",
        "nope",
        "no thats wrong",
        "thats not right"
    ], "statement", "me", "communicate", "no"),

    # acknowledgement
    ([
        "okay",
        "alright",
        "i understand",
        "got it",
        "okay i understand"
    ], "statement", "me", "communicate", "okay"),

    # thanks
    ([
        "thank you",
        "thanks",
        "thanks a lot",
        "thank you very much"
    ], "statement", "me", "communicate", "thanks"),

    # greeting
    ([
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ], "statement", "none", "communicate", "hello")
]

# ============================================================
# NATURAL SPEECH VARIATIONS
# ============================================================
PREFIXES = ["", "", "", "", "hey ", "um ", "uh ", "so ", "well "]

ASR_REPLACEMENTS = {
    "going to": ["gonna"],
    "want to": ["wanna"],
    "give me": ["gimme"],
    "do you": ["d you"],
    "i am": ["im"],
    "did not": ["didnt"],
    "do not": ["dont"]
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def add_asr_variation(text):
    if random.random() > 0.12:
        return text
    possible = [x for x in ASR_REPLACEMENTS if x in text]
    if possible:
        original = random.choice(possible)
        text = text.replace(original, random.choice(ASR_REPLACEMENTS[original]), 1)
    return text

def make_row(text, intent, subject, action, concept):
    # Only use harmless discourse fillers here.
    if random.random() < 0.10:
        prefix = random.choice(PREFIXES)
        if prefix and not text.startswith(("watch out", "danger", "stop")):
            text = prefix + text
    text = normalize_text(add_asr_variation(text))

    assert intent in INTENTS, f"Invalid intent: {intent}"
    assert subject in SUBJECTS, f"Invalid subject: {subject}"
    assert action in ACTIONS, f"Invalid action: {action}"

    return text, intent, subject, action, concept

def random_concept(bank):
    concept = random.choice(list(bank))
    return concept, random.choice(bank[concept])

# ============================================================
# GENERATORS
# ============================================================
def generate_question_need():
    concept, surface = random_concept(NEEDS)
    text = random.choice(QUESTION_NEED).format(c=surface)
    return make_row(text, "question", "you", "need", concept)

def generate_request_need():
    concept, surface = random_concept(NEEDS)
    text = random.choice(REQUEST_NEED).format(c=surface)
    return make_row(text, "request", "me", "need", concept)

def generate_alert():
    concept, surface = random_concept(HAZARDS)
    text = random.choice(ALERTS).format(c=surface)
    return make_row(text, "alert", "you", "danger", concept)

def generate_location_question():
    concept, surface = random_concept(PLACES)
    text = random.choice(LOCATION_QUESTIONS).format(c=surface)
    return make_row(text, "question", "none", "move", concept)

def generate_movement():
    concept, surface = random_concept(PLACES)
    text = random.choice(MOVEMENT).format(c=surface)
    return make_row(text, "statement", "me", "move", concept)

def generate_feeling():
    concept, surface = random_concept(FEELINGS)
    text = random.choice(FEELING_STATEMENTS).format(c=surface)
    return make_row(text, "statement", "me", "feel", concept)

def generate_feeling_question():
    concept, surface = random_concept(FEELINGS)
    text = random.choice(FEELING_QUESTIONS).format(c=surface)
    return make_row(text, "question", "you", "feel", concept)

def generate_conversation():
    phrases, intent, subject, action, concept = random.choice(CONVERSATION_FAMILIES)
    return make_row(random.choice(phrases), intent, subject, action, concept)

GENERATORS = [
    generate_question_need,
    generate_request_need,
    generate_alert,
    generate_location_question,
    generate_movement,
    generate_feeling,
    generate_feeling_question,
    generate_conversation
]

GENERATOR_WEIGHTS = [15, 15, 15, 10, 10, 10, 10, 15]

# ============================================================
# DATASET GENERATION
# ============================================================
def generate_dataset(total_rows):
    rows = []
    seen = set()
    attempts = 0
    max_attempts = total_rows * 200

    while len(rows) < total_rows and attempts < max_attempts:
        generator = random.choices(GENERATORS, weights=GENERATOR_WEIGHTS, k=1)[0]
        row = generator()
        if row[0] not in seen:
            seen.add(row[0])
            rows.append(row)
        attempts += 1

    random.shuffle(rows)

    if len(rows) < total_rows:
        print(f"WARNING: Requested {total_rows} unique rows but generated only {len(rows)}.")
        print("Add more real paraphrases instead of forcing artificial duplicates.")

    return rows

def write_csv(rows):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_text", "intent", "subject", "action", "concept"])
        writer.writerows(rows)

def print_report(rows):
    print(f"\nGenerated {len(rows)} rows -> {OUTPUT_FILE}")
    for index, name in [(1, "Intent"), (2, "Subject"), (3, "Action")]:
        counts = Counter(row[index] for row in rows)
        print(f"\n{name} distribution:")
        for key, value in sorted(counts.items()):
            print(f"  {key:15s}: {value}")

    concepts = Counter(row[4] for row in rows)
    print(f"\nDistinct concepts: {len(concepts)}")
    print("\nSample rows:")
    for row in random.sample(rows, min(20, len(rows))):
        print(row)

if __name__ == "__main__":
    dataset = generate_dataset(TOTAL_ROWS)
    write_csv(dataset)
    print_report(dataset)