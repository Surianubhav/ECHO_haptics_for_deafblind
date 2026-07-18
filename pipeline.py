import sys
import json
import re
import requests # Used for the Sarvam Cloud API fallback

# --- 1. LOCAL GRADE 2 BRAILLE CONTRACTION DICTIONARY ---
# Map common day-to-day objects/words directly to their Grade 2 single-cell abbreviations.
# Standard character fallback maps to standard 6-dot binary arrays.
BRAILLE_GRADE_2 = {
    # Custom Shortcuts & Grade 2 Contractions
    "water": "w", "car": "c", "ditch": "d", "knowledge": "k", "people": "p", 
    "go": "g", "have": "h", "more": "m", "not": "n", "rather": "r", "so": "s",
    "that": "t", "us": "u", "very": "v", "you": "y", "as": "z"
}

BRAILLE_CELLS = {
    'a': 0b000001, 'b': 0b000011, 'c': 0b001001, 'd': 0b011001, 'e': 0b010001,
    'f': 0b001011, 'g': 0b011011, 'h': 0b010011, 'i': 0b001010, 'j': 0b011010,
    'k': 0b000101, 'l': 0b000111, 'm': 0b001101, 'n': 0b011101, 'o': 0b010101,
    'p': 0b001111, 'q': 0b011111, 'r': 0b010111, 's': 0b001110, 't': 0b011110,
    'u': 0b100101, 'v': 0b100111, 'w': 0b011110, 'x': 0b101101, 'y': 0b111101,
    'z': 0b110101, ' ': 0b000000, '#': 0b111100  # Number sign indicator
}

# --- 2. LOCAL NLP / TEXT EXTRACTION ENGINE ---
def analyze_sentence_locally(text: str):
    """
    Parses sentences deterministically. Returns (success_bool, result_dict)
    If a sentence is outside our day-to-day regex/keyword rules, it flags success=False
    """
    clean_text = text.lower().strip().replace("?", "").replace(".", "").replace("!", "")
    words = clean_text.split()
    
    # Initialize the target output schema
    result = {
        "sentence_type": 0b000, # Finger 1
        "subject": 0b000,       # Finger 2
        "interrogative": 0b000, # Finger 3
        "braille_words": []     # List of words to display on palm
    }
    
    # 1. Detect Sentence Type & Urgency (Finger 1)
    if any(w in words for w in ["car", "ditch", "danger", "look out", "stop"]):
        result["sentence_type"] = 0b100 # Rapid Buzz / Alert
    elif text.strip().endswith("?") or any(w in words for w in ["how", "what", "where", "why", "who", "can", "do"]):
        result["sentence_type"] = 0b001 # Question
    else:
        result["sentence_type"] = 0b010 # Normal Statement
        
    # 2. Detect Subject / Person (Finger 2)
    if "you" in words or "your" in words or "u" in words:
        result["subject"] = 0b010 # 2nd Person
    elif "i" in words or "me" in words or "we" in words or "my" in words:
        result["subject"] = 0b001 # 1st Person
    elif any(w in words for w in ["he", "she", "they", "it", "them"]):
        result["subject"] = 0b100 # 3rd Person
        
    # 3. Detect Interrogative Modifier (Finger 3)
    if "how" in words: result["interrogative"] = 0b001
    elif "what" in words: result["interrogative"] = 0b010
    elif "where" in words: result["interrogative"] = 0b011
    elif "why" in words: result["interrogative"] = 0b100
    elif "when" in words: result["interrogative"] = 0b101
    elif "who" in words: result["interrogative"] = 0b110
    
    # 4. Extract Object / Core Concept
    target_object = None
    known_objects = ["water", "car", "ditch", "ok", "order", "right", "left", "total"]
    for w in words:
        if w in known_objects or w.isdigit():
            target_object = w
            break
            
    # Handle Numeric strings ("100")
    if target_object and target_object.isdigit():
        result["braille_words"].append("#") # Prepend Number Sign
        for digit in target_object:
            # In Braille numbers 1-9,0 are represented by letters a-j
            digit_map = {'1':'a','2':'b','3':'c','4':'d','5':'e','6':'f','7':'g','8':'h','9':'i','0':'j'}
            result["braille_words"].append(digit_map[digit])
    elif target_object:
        # Check if the whole object fits inside a single Grade 2 contraction
        if target_object in BRAILLE_GRADE_2:
            result["braille_words"].append(BRAILLE_GRADE_2[target_object])
        else:
            result["braille_words"].append(target_object)
    else:
        # If we cannot cleanly find a known day-to-day object, local parsing confidence is low.
        return False, None

    return True, result

# --- 3. SARVAM CLOUD API FALLBACK TIER ---
def analyze_sentence_via_cloud(text: str, api_key: str):
    """
    Sends complex, unhandled, or ambiguous multi-lingual sentences to the Sarvam AI Cloud
    Instructs the LLM to return data matching our exact 3-finger / palm byte schema.
    """
    url = "https://api.sarvam.ai/v1/chat/completions" # Generic standard endpoint placeholder for context
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Analyze the following sentence for a DeafBlind haptic translation glove: "{text}"
    Output a raw JSON object with these EXACT integer/list mappings:
    - sentence_type: (1 for question, 2 for statement, 4 for critical alert)
    - subject: (1 for 1st person, 2 for 2nd person, 4 for 3rd person)
    - interrogative: (1 for how, 2 for what, 3 for where, 4 for why, 5 for when, 6 for who, 0 for none)
    - braille_words: A list containing either a single Grade 2 contracted letter shortcut (e.g. 'w' for water) or individual character letters spelled out.
    Return ONLY raw JSON. No markdown wrappers.
    """
    
    payload = {
        "model": "sarvam-2b-ft", # Explicitly pointing to your hackathon sandbox models
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=4)
        if response.status_code == 200:
            ai_output = response.json()['choices'][0]['message']['content']
            return json.loads(ai_output)
    except Exception as e:
        print(f"Cloud Fallback Failed: {e}", file=sys.stderr)
    
    # Static micro-fallback system if the network completely cuts out during presentation
    return {"sentence_type": 2, "subject": 0, "interrogative": 0, "braille_words": ["error"]}

# --- 4. DATA PACKAGER & ARDUINO BYTE TRANSMITTER ---
def generate_haptic_packets(parsed_data):
    """
    Converts the structured parameters into individual 3-byte packets 
    ready to stream straight over the Arduino Serial interface.
    """
    packets = []
    finger_1 = parsed_data["sentence_type"]
    finger_2 = parsed_data["subject"]
    finger_3 = parsed_data["interrogative"]
    
    # Merge the 3 unique finger states into a single raw byte
    # Bits: [0 0 0 0 0 F3 F2 F1]
    finger_packet_byte = (finger_3 << 6) | (finger_2 << 3) | finger_1
    
    for word_token in parsed_data["braille_words"]:
        for char in word_token:
            palm_byte = BRAILLE_CELLS.get(char, 0b000000)
            duration_byte = 150 # 150ms per letter flash frame
            
            # Formulate our complete 3-byte command
            packets.append(bytes([finger_packet_byte, palm_byte, duration_byte]))
            
    return packets

# --- 5. EXECUTION PIPELINE RUNNER ---
def process_whisper_input(raw_speech_text: str, sarvam_api_key: str = "MOCK_KEY"):
    print(f"\n[Whisper Input]: '{raw_speech_text}'")
    
    # Step A: Attempt ultra-fast local keyword execution
    success, output = analyze_sentence_locally(raw_speech_text)
    
    if success:
        print("--> Processed via Local Engine (Confidence: 100%)")
    else:
        print("--> Local Parsing Confidence Low. Routing to Sarvam AI Cloud...")
        output = analyze_sentence_via_cloud(raw_speech_text, sarvam_api_key)
        
    print(f"[Parsed Context Result]: {output}")
    
    # Step B: Generate raw hardware-executable bytes
    binary_packets = generate_haptic_packets(output)
    
    print(f"[Generated Hardware Serial Stream]:")
    for idx, packet in enumerate(binary_packets):
        print(f"  Frame {idx+1} -> Fingers: {bin(packet[0])} | Palm Matrix: {bin(packet[1])} | Duration: {packet[2]}ms")
        
    return binary_packets

# --- TEST CASES RUNNER ---
if __name__ == "__main__":
    # Test 1: Standard Day-to-Day Question Handled Locally
    process_whisper_input("Do you need water?")
    
    # Test 2: High Priority Safety Alert Handled Locally
    process_whisper_input("There is a car coming towards you!")
    
    # Test 3: Complex out-of-scope sentence triggering the cloud routing path
    process_whisper_input("The airline representative says your flight leaves at midnight.")