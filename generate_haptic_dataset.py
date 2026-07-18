"""
generate_haptic_dataset.py
---------------------------------------------------
Synthetic dataset generator for a DeafBlind haptic
translation glove — Intent & Slot Filling training data.

Columns: input_text, sentence_type, subject, core_concept
sentence_type in {question, alert, statement, need}

Design notes:
- itertools.product() builds the full (template x concept) combinatorial
  space per sentence_type, then itertools.cycle() walks it in a shuffled,
  round-robin order. This guarantees even coverage across concepts and
  templates (no clumping / overfitting to a few popular pairs), while
  random.* handles all the surface-level noise: synonym swap, subject
  choice, filler words, colloquialisms, and simulated typos.
- Run this script locally: `python generate_haptic_dataset.py`
  It writes haptic_dataset_7000.csv in the current directory.
"""

import csv
import random
import itertools

# ============================================================
# CONFIG
# ============================================================
TOTAL_ROWS = 7000
OUTPUT_FILE = "haptic_dataset_7000.csv"
# Leave unseeded for a fresh random dataset every run.
# Uncomment the next line for reproducible output:
# random.seed(42)

# ============================================================
# 1. CORE CONCEPTS (120 total — hazards, needs, feelings, places)
# ============================================================
HAZARD_CONCEPTS = [
    "car", "bicycle", "motorcycle", "truck", "train", "bus", "pole", "curb",
    "ditch", "hole", "stairs", "step", "ice", "wet_floor", "broken_glass",
    "fire", "hot_stove", "knife", "wire", "fence", "barbed_wire",
    "electric_fence", "construction", "scaffolding", "ladder", "forklift",
    "crowd", "stranger", "dog", "bee", "wasp", "cliff_edge", "ledge",
    "puddle", "tree_root", "loose_rug", "escalator", "elevator",
    "platform_edge", "low_ceiling", "open_manhole", "spinning_fan",
    "falling_object", "traffic", "gate", "rock", "spill", "sharp_corner",
    "moving_cart",
]

NEED_CONCEPTS = [
    "water", "food", "bathroom", "medicine", "blanket", "help", "rest",
    "sleep", "jacket", "glasses", "phone", "cane", "guide_dog",
    "wheelchair", "walker", "hearing_aid", "interpreter", "flashlight",
    "charger", "umbrella", "gloves", "hat", "towel", "soap", "coat",
    "scarf", "socks", "shoes", "pillow", "tissue", "snack", "painkillers",
    "bandage", "wifi", "map", "directions", "money", "wallet", "keys",
    "id_card", "ticket", "chair",
]

FEELING_CONCEPTS = [
    "hungry", "thirsty", "tired", "cold", "hot", "happy", "sad", "sick",
    "dizzy", "anxious", "excited", "bored", "lonely", "scared", "pain",
]

PLACE_CONCEPTS = [
    "home", "work", "school", "park", "store", "hospital", "kitchen",
    "restroom", "bus_stop", "train_station", "restaurant", "office",
    "garden", "bedroom", "hotel",
]

CONCEPTS = HAZARD_CONCEPTS + NEED_CONCEPTS + FEELING_CONCEPTS + PLACE_CONCEPTS
assert len(set(CONCEPTS)) >= 100, "Need at least 100 distinct core concepts"

# ============================================================
# 2. SYNONYM BANK (heavy lexical variety; core_concept column
#    ALWAYS stays the canonical value, only input_text varies)
# ============================================================
SYNONYMS = {
    "water": ["water", "drink", "hydration", "liquid", "fluids"],
    "food": ["food", "meal", "something to eat", "snack", "grub"],
    "bathroom": ["bathroom", "restroom", "toilet", "washroom", "loo"],
    "restroom": ["restroom", "bathroom", "toilet", "washroom"],
    "medicine": ["medicine", "meds", "pills", "medication"],
    "help": ["help", "assistance", "aid", "support"],
    "money": ["money", "cash", "funds", "dough"],
    "phone": ["phone", "cellphone", "mobile", "cell"],
    "stairs": ["stairs", "steps", "staircase"],
    "fire": ["fire", "flames", "blaze", "smoke"],
    "car": ["car", "vehicle", "ride", "automobile"],
    "train": ["train", "subway", "metro", "railway"],
    "bus": ["bus", "shuttle", "coach"],
    "dog": ["dog", "pooch", "canine"],
    "guide_dog": ["guide dog", "service dog", "seeing eye dog"],
    "pain": ["pain", "hurting", "aching", "soreness"],
    "tired": ["tired", "exhausted", "sleepy", "worn out"],
    "hungry": ["hungry", "starving", "famished"],
    "thirsty": ["thirsty", "parched", "dehydrated"],
    "cold": ["cold", "chilly", "freezing"],
    "hot": ["hot", "warm", "overheated"],
    "sick": ["sick", "ill", "unwell"],
    "scared": ["scared", "afraid", "frightened"],
    "anxious": ["anxious", "nervous", "uneasy"],
    "wallet": ["wallet", "purse", "billfold"],
    "keys": ["keys", "house keys", "car keys"],
    "charger": ["charger", "cable", "power cord"],
    "umbrella": ["umbrella", "rain cover"],
    "blanket": ["blanket", "cover", "quilt"],
    "jacket": ["jacket", "coat", "outerwear"],
    "glasses": ["glasses", "spectacles", "eyewear"],
    "wheelchair": ["wheelchair", "chair"],
    "flashlight": ["flashlight", "torch", "light"],
    "wifi": ["wifi", "internet", "connection"],
    "map": ["map", "directions", "route"],
    "ticket": ["ticket", "pass", "fare"],
    "hospital": ["hospital", "clinic", "er", "emergency room"],
    "restaurant": ["restaurant", "diner", "cafe"],
    "home": ["home", "house", "place"],
    "work": ["work", "job", "office"],
    "school": ["school", "class", "campus"],
    "store": ["store", "shop", "market"],
    "traffic": ["traffic", "cars", "vehicles"],
    "wire": ["wire", "cable", "cord"],
    "hole": ["hole", "gap", "pit"],
    "ice": ["ice", "black ice", "frozen patch"],
}

def get_synonym(concept):
    """Return a randomly chosen synonym for use in input_text."""
    options = SYNONYMS.get(concept, [concept.replace("_", " ")])
    return random.choice(options)

# ============================================================
# 3. SUBJECTS
# ============================================================
THIRD_PARTY_PEOPLE = [
    "mom", "dad", "sister", "brother", "teacher", "doctor", "friend",
    "grandma", "grandpa", "neighbor", "son", "daughter", "uncle", "aunt",
    "coworker", "nurse",
]
STATEMENT_SUBJECTS = ["i", "he", "she", "we", "they"] + THIRD_PARTY_PEOPLE

ATTRIBUTE_WORDS = [
    "name", "color", "size", "price", "location", "time", "date", "owner",
    "weight", "brand", "flavor", "title", "address", "number", "schedule",
    "status",
]
ATTRIBUTE_OBJECTS = [
    "book", "shirt", "car", "meeting", "house", "phone", "bag", "movie",
    "song", "flight", "package", "recipe", "restaurant", "hotel", "train",
    "appointment", "medicine", "document", "photo", "gift",
]

# ============================================================
# 4. TEMPLATES  (>=15 each, {c} = concept synonym slot)
# ============================================================
QUESTION_TEMPLATES = [
    "where is the {c}",
    "do you have the {c}",
    "is there a {c} nearby",
    "can you find my {c}",
    "do you need the {c}",
    "is the {c} close by",
    "how far is the {c}",
    "can i get some {c}",
    "do we have any {c}",
    "is my {c} here",
    "where can i find the {c}",
    "do you see the {c}",
    "is there {c} around here",
    "how much is the {c}",
    "can you check the {c}",
    "do you know where the {c} is",
]

ALERT_TEMPLATES = [
    "watch out for the {c}",
    "danger {c} ahead",
    "stop there is a {c}",
    "careful the {c} is close",
    "caution {c} nearby",
    "be careful of the {c}",
    "watch your step near the {c}",
    "warning {c} up ahead",
    "heads up a {c} is coming",
    "look out for the {c}",
    "mind the {c}",
    "slow down there is a {c}",
    "attention the {c} is right there",
    "stay back from the {c}",
    "avoid the {c}",
    "careful theres a {c} on your left",
]

STATEMENT_TEMPLATES = [
    "{subj} going to the {c}",
    "{subj} at the {c} right now",
    "{subj} near the {c}",
    "{subj} found the {c}",
    "{subj} just left the {c}",
    "{subj} heading to the {c}",
    "{subj} had some {c}",
    "{subj} over by the {c}",
    "{subj} saw the {c}",
    "{subj} passed the {c}",
    "{subj} finished with the {c}",
    "{subj} fixing the {c}",
    "{subj} feeling {c}",
    "{subj} talking about the {c}",
    "{subj} waiting for the {c}",
]

NEED_TEMPLATES = [
    "i need my {c}",
    "lets go to the {c}",
    "i need some {c}",
    "i want {c} now",
    "can someone get me {c}",
    "i really need {c}",
    "please bring my {c}",
    "i could use some {c}",
    "i need help with the {c}",
    "give me the {c} please",
    "i am asking for {c}",
    "i need to find the {c}",
    "i need a {c}",
    "take me to the {c}",
    "i need {c} right now",
]

ATTRIBUTE_TEMPLATES = [
    "what is the {attr} of the {obj}",
    "what {attr} is the {obj}",
    "where is the {obj}",
    "what is the {obj} {attr}",
]

THIRD_PARTY_NEED_TEMPLATES = [
    "does {p} need {c}",
    "does {p} want {c}",
    "is {p} looking for {c}",
    "did {p} bring {c}",
    "does {p} have {c}",
]

# ============================================================
# 5. NOISE INJECTION (simulate real spoken / Whisper transcripts)
# ============================================================
FILLERS = ["um", "uh", "hey", "so", "like", "well", "actually", "you know"]

COLLOQUIALISMS = {
    " want to ": " wanna ",
    " going to ": " gonna ",
    " do you ": " d'you ",
    " kind of ": " kinda ",
    " let us ": " lets ",
    " give me ": " gimme ",
    " got to ": " gotta ",
}

def add_filler(text):
    """~35% chance: insert a filler word at the start or mid-sentence."""
    if random.random() < 0.35:
        filler = random.choice(FILLERS)
        words = text.split()
        if random.random() < 0.5 or len(words) < 2:
            words = [filler] + words
        else:
            idx = random.randint(1, len(words) - 1)
            words.insert(idx, filler)
        text = " ".join(words)
    return text

def colloquialize(text):
    """~25% chance: swap a phrase for its casual contraction."""
    if random.random() < 0.25:
        for phrase, replacement in COLLOQUIALISMS.items():
            if phrase in text:
                text = text.replace(phrase, replacement)
                break
    return text

def add_typo(text):
    """~15% chance: simulate an ASR/typing slip (swap, drop, or double a letter)."""
    if random.random() < 0.15:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w) > 3]
        if candidates:
            idx = random.choice(candidates)
            w = words[idx]
            i = random.randrange(1, len(w) - 1)
            typo_type = random.choice(["swap", "drop", "double"])
            if typo_type == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif typo_type == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
            words[idx] = w
            text = " ".join(words)
    return text

def finalize(text):
    text = colloquialize(text)
    text = add_filler(text)
    text = add_typo(text)
    return text.strip()

# ============================================================
# 6. COMBINATORIAL COVERAGE (itertools) + RANDOM SURFACE NOISE
# ============================================================
def make_cycle(templates, concepts):
    """
    Build the full itertools.product() space of template x concept,
    shuffle it, and return an itertools.cycle() iterator so we can
    pull from it repeatedly without ever clumping on a few popular
    pairs (this is what prevents overfitting at 7000 rows).
    """
    combos = list(itertools.product(templates, concepts))
    random.shuffle(combos)
    return itertools.cycle(combos)

question_cycle = make_cycle(QUESTION_TEMPLATES, CONCEPTS)
alert_cycle = make_cycle(ALERT_TEMPLATES, CONCEPTS)
statement_cycle = make_cycle(STATEMENT_TEMPLATES, CONCEPTS)
need_cycle = make_cycle(NEED_TEMPLATES, CONCEPTS)
attribute_cycle = make_cycle(ATTRIBUTE_TEMPLATES, list(itertools.product(ATTRIBUTE_WORDS, ATTRIBUTE_OBJECTS)))
thirdparty_cycle = make_cycle(THIRD_PARTY_NEED_TEMPLATES, list(itertools.product(THIRD_PARTY_PEOPLE, CONCEPTS)))

def gen_question_row():
    """Mix of plain questions, attribute questions, and third-party need questions."""
    r = random.random()
    if r < 0.55:
        template, concept = next(question_cycle)
        text = template.format(c=get_synonym(concept))
        return finalize(text), "question", "you", concept
    elif r < 0.78:
        template, (attr, obj) = next(attribute_cycle)
        text = template.format(attr=attr, obj=obj)
        return finalize(text), "question", "you", f"{attr}_{obj}"
    else:
        template, (person, concept) = next(thirdparty_cycle)
        text = template.format(p=person, c=get_synonym(concept))
        return finalize(text), "question", person, concept

def gen_alert_row():
    template, concept = next(alert_cycle)
    text = template.format(c=get_synonym(concept))
    return finalize(text), "alert", "you", concept

def gen_statement_row():
    template, concept = next(statement_cycle)
    subj = random.choice(STATEMENT_SUBJECTS)
    text = template.format(subj=subj, c=get_synonym(concept))
    label = "me" if subj == "i" else subj
    return finalize(text), "statement", label, concept

def gen_need_row():
    template, concept = next(need_cycle)
    text = template.format(c=get_synonym(concept))
    return finalize(text), "need", "me", concept

GENERATORS = {
    "question": gen_question_row,
    "alert": gen_alert_row,
    "statement": gen_statement_row,
    "need": gen_need_row,
}

# ============================================================
# 7. DATASET ASSEMBLY
# ============================================================
def generate_dataset(total=TOTAL_ROWS):
    per_type = total // 4
    remainder = total - per_type * 4
    types = list(GENERATORS.keys())
    targets = {t: per_type for t in types}
    for i in range(remainder):
        targets[types[i]] += 1

    seen_texts = set()
    rows = []
    for stype, target in targets.items():
        gen = GENERATORS[stype]
        collected = []
        attempts = 0
        max_attempts = target * 60  # generous ceiling given random noise variety
        while len(collected) < target and attempts < max_attempts:
            row = gen()
            text = row[0]
            if text not in seen_texts:
                seen_texts.add(text)
                collected.append(row)
            attempts += 1
        rows.extend(collected)

    random.shuffle(rows)
    return rows[:total]

def write_csv(rows, filename=OUTPUT_FILE):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_text", "sentence_type", "subject", "core_concept"])
        writer.writerows(rows)

if __name__ == "__main__":
    dataset = generate_dataset(TOTAL_ROWS)
    write_csv(dataset)
    print(f"Generated {len(dataset)} rows -> {OUTPUT_FILE}")

    # Quick sanity summary
    from collections import Counter
    counts = Counter(r[1] for r in dataset)
    print("Sentence type distribution:", dict(counts))
    print("Distinct core_concepts used:", len(set(r[3] for r in dataset)))
