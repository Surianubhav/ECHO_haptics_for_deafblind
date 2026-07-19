#ifndef BRAILLE_VOCABULARY_H
#define BRAILLE_VOCABULARY_H

#include <Arduino.h>

// Grade 1 Braille Alphabet Bitmasks (Letters a-z)
const uint8_t BRAILLE_ALPHABET[26] = {
  0b000001, // a (dot 1)
  0b000011, // b (dot 1,2)
  0b001001, // c (dot 1,4)
  0b011001, // d (dot 1,4,5)
  0b010001, // e (dot 1,5)
  0b001011, // f (dot 1,2,4)
  0b011011, // g (dot 1,2,4,5)
  0b010011, // h (dot 1,2,5)
  0b001010, // i (dot 2,4)
  0b011010, // j (dot 2,4,5)
  0b000101, // k (dot 1,3)
  0b000111, // l (dot 1,2,3)
  0b001101, // m (dot 1,3,4)
  0b011101, // n (dot 1,3,4,5)
  0b010101, // o (dot 1,3,5)
  0b001111, // p (dot 1,2,3,4)
  0b011111, // q (dot 1,2,3,4,5)
  0b010111, // r (dot 1,2,3,5)
  0b001110, // s (dot 2,3,4)
  0b011110, // t (dot 2,3,4,5)
  0b100101, // u (dot 1,3,6)
  0b100111, // v (dot 1,2,3,6)
  0b111010, // w (dot 2,4,5,6)
  0b101101, // x (dot 1,3,4,6)
  0b111101, // y (dot 1,3,4,5,6)
  0b110101  // z (dot 1,3,5,6)
};

struct Contraction {
  const char* fullWord;
  const char* shortcut;
};

// Grade 2 Contraction Lookup Table
const Contraction GRADE2_DICTIONARY[] = {
  {"ch",         "ch"},  
  {"sh",         "sh"},
  {"th",         "th"},
  {"about",      "ab"},
  {"above",      "abv"},
  {"according",  "ac"},
  {"across",     "acr"},
  {"after",      "af"},
  {"afternoon",  "afn"},
  {"afterward",  "afw"},
  {"again",      "ag"},
  {"against",    "agst"},
  {"almost",     "alm"},
  {"already",    "alr"},
  {"also",       "al"},
  {"although",   "alth"},
  {"altogether", "alt"},
  {"always",     "alw"},
  {"apple",      "ap"},
  {"blind",      "bl"},
  {"braille",    "brl"},
  {"children",   "chn"},
  {"father",     "fa"},
  {"friend",     "fr"},
  {"mother",     "mo"},
  {"sister",     "sis"},
  {"tomorrow",   "tm"}
};

// Syllable boundary exception rules
const char* EXTRA_EXCLUSIONS[] = {
  "outhouse", "shanghai", "mishap", "hogshead"
};

const int DICTIONARY_SIZE = sizeof(GRADE2_DICTIONARY) / sizeof(Contraction);
const int EXCLUSIONS_SIZE = sizeof(EXTRA_EXCLUSIONS) / sizeof(char*);

bool isContractionExcluded(const String& inputWord) {
  for (int i = 0; i < EXCLUSIONS_SIZE; i++) {
    if (inputWord.equalsIgnoreCase(EXTRA_EXCLUSIONS[i])) {
      return true;
    }
  }
  return false;
}

bool getGrade2Shortcut(const String& inputWord, char* outputBuffer, size_t maxLen) {
  if (isContractionExcluded(inputWord)) {
    return false; 
  }
  for (int i = 0; i < DICTIONARY_SIZE; i++) {
    if (inputWord.equalsIgnoreCase(GRADE2_DICTIONARY[i].fullWord)) {
      strncpy(outputBuffer, GRADE2_DICTIONARY[i].shortcut, maxLen - 1);
      outputBuffer[maxLen - 1] = '\0';
      return true;
    }
  }
  return false;
}

#endif