import pandas as pd
import random

OUTPUT_FILE = "haptic_expansion.csv"
ROWS = 5000

subjects = {
    "me": ["i", "i am", "im"],
    "you": ["you", "you are", "youre"],
    "other": [
        "he", "she", "they", "my mom", "my dad",
        "my friend", "the doctor", "the teacher",
        "my brother", "my sister"
    ]
}

data = []

def add(text, intent, subject, action, concept):
    data.append({
        "input_text": text,
        "intent": intent,
        "subject": subject,
        "action": action,
        "concept": concept
    })

# ==================================================
# CONVERSATIONAL QUESTIONS
# ==================================================

questions = [
    ("how are you", "you", "feel", "none"),
    ("how are you doing", "you", "feel", "none"),
    ("are you okay", "you", "feel", "none"),
    ("what are you doing", "you", "do", "none"),
    ("what are you up to", "you", "do", "none"),
    ("where are you going", "you", "move", "destination"),
    ("where are we going", "other", "move", "destination"),
    ("what is your name", "you", "communicate", "none"),
    ("who are you", "you", "communicate", "none"),
    ("what did you say", "you", "communicate", "none"),
    ("can you repeat that", "you", "communicate", "none"),
    ("do you understand", "you", "communicate", "none"),
]

for text, subject, action, concept in questions:
    add(text, "question", subject, action, concept)

# ==================================================
# THIRD-PERSON NEEDS
# ==================================================

people = [
    "he", "she", "they", "my mom", "my dad",
    "my friend", "my brother", "my sister",
    "the doctor", "the teacher"
]

needs = {
    "water": ["water", "a drink"],
    "food": ["food", "something to eat"],
    "medicine": ["medicine", "medication", "painkillers"],
    "help": ["help", "assistance"],
    "rest": ["rest", "a break"],
    "bathroom": ["the bathroom", "the restroom"]
}

for person in people:
    for concept, words in needs.items():
        for word in words:
            add(
                f"{person} needs {word}",
                "statement", "other", "need", concept
            )
            add(
                f"{person} wants {word}",
                "statement", "other", "need", concept
            )
            add(
                f"does {person} need {word}",
                "question", "other", "need", concept
            )

# ==================================================
# MOVEMENT WITHOUT KNOWN DESTINATION
# ==================================================

movement_none = [
    "i am going",
    "i am going to go",
    "i have to go",
    "i am leaving",
    "we are going",
    "we should go",
]

for text in movement_none:
    add(text, "statement", "me", "move", "none")

# ==================================================
# KNOWN DESTINATIONS
# ==================================================

places = [
    "home", "school", "hospital", "bathroom",
    "restaurant", "office", "park", "store"
]

for place in places:
    add(f"i am going to the {place}", "statement", "me", "move", place)
    add(f"take me to the {place}", "request", "you", "move", place)
    add(f"where is the {place}", "question", "you", "move", place)

# ==================================================
# FEELINGS
# ==================================================

feelings = [
    "happy", "sad", "tired", "sick",
    "scared", "cold", "hot", "hungry", "thirsty"
]

for feeling in feelings:
    add(f"i feel {feeling}", "statement", "me", "feel", feeling)
    add(f"i am feeling {feeling}", "statement", "me", "feel", feeling)
    add(f"i am {feeling}", "statement", "me", "feel", feeling)
    add(f"are you {feeling}", "question", "you", "feel", feeling)

# ==================================================
# GENERAL CONVERSATION WITH NO PHYSICAL CONCEPT
# ==================================================

general = [
    ("i am not sure", "statement", "me", "communicate", "none"),
    ("i dont know", "statement", "me", "communicate", "none"),
    ("i understand", "statement", "me", "communicate", "none"),
    ("i dont understand", "statement", "me", "communicate", "none"),
    ("thank you", "statement", "me", "communicate", "none"),
    ("please repeat that", "request", "you", "communicate", "none"),
    ("say that again", "request", "you", "communicate", "none"),
    ("speak again", "request", "you", "communicate", "none"),
]

for row in general:
    add(*row)

# ==================================================
# DIRECTIONS
# ==================================================

directions = ["left", "right", "forward", "back"]

for direction in directions:
    add(f"turn {direction}", "request", "you", "move", direction)
    add(f"go {direction}", "request", "you", "move", direction)
    add(f"move {direction}", "request", "you", "move", direction)

# ==================================================
# CREATE VARIATIONS
# ==================================================

base_data = data.copy()

fillers = ["hey", "please", "well", "so", "um"]

while len(data) < ROWS:
    row = random.choice(base_data).copy()
    text = row["input_text"]

    variation = random.randint(0, 3)

    if variation == 0:
        text = random.choice(fillers) + " " + text
    elif variation == 1:
        text = text + " please"
    elif variation == 2:
        text = text.replace("i am", "im")
    elif variation == 3:
        text = text.replace("you are", "youre")

    row["input_text"] = text
    data.append(row)

df = pd.DataFrame(data[:ROWS])

df = df.drop_duplicates()

df.to_csv(OUTPUT_FILE, index=False)

print(f"Generated {len(df)} expansion rows")
print(df["intent"].value_counts())
print(df["subject"].value_counts())
print(df["action"].value_counts())
print(f"Saved to {OUTPUT_FILE}")