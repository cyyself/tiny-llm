#!/usr/bin/env python3
"""Generate a Q&A training corpus for the tiny LLM."""
import random
random.seed(42)

qa_pairs = []

# --- Animal facts (6 animals, 4 questions each = 24 pairs) ---
animals = {
    'cat': {'sound': 'meow', 'legs': 'four', 'food': 'fish', 'home': 'house'},
    'dog': {'sound': 'bark', 'legs': 'four', 'food': 'meat', 'home': 'house'},
    'bird': {'sound': 'chirp', 'legs': 'two', 'food': 'seeds', 'home': 'nest'},
    'fish': {'sound': 'splash', 'legs': 'zero', 'food': 'worms', 'home': 'water'},
    'cow': {'sound': 'moo', 'legs': 'four', 'food': 'grass', 'home': 'farm'},
    'duck': {'sound': 'quack', 'legs': 'two', 'food': 'bread', 'home': 'pond'},
}
for name, info in animals.items():
    qa_pairs.append(f"Q: What sound does a {name} make? A: A {name} makes a {info['sound']} sound.")
    qa_pairs.append(f"Q: How many legs does a {name} have? A: A {name} has {info['legs']} legs.")
    qa_pairs.append(f"Q: What does a {name} eat? A: A {name} eats {info['food']}.")
    qa_pairs.append(f"Q: Where does a {name} live? A: A {name} lives in a {info['home']}.")

# --- Held-out animals: include sound+legs+home only (food is held out for testing) ---
held_out_animals = {
    'horse': {'sound': 'neigh', 'legs': 'four', 'home': 'stable'},
    'rabbit': {'sound': 'squeak', 'legs': 'four', 'home': 'burrow'},
}
for name, info in held_out_animals.items():
    qa_pairs.append(f"Q: What sound does a {name} make? A: A {name} makes a {info['sound']} sound.")
    qa_pairs.append(f"Q: How many legs does a {name} have? A: A {name} has {info['legs']} legs.")
    qa_pairs.append(f"Q: Where does a {name} live? A: A {name} lives in a {info['home']}.")

# --- Colors (5 pairs) ---
color_facts = [
    ('sky', 'blue'), ('grass', 'green'), ('sun', 'yellow'),
    ('snow', 'white'), ('night', 'dark'),
]
for thing, color in color_facts:
    qa_pairs.append(f"Q: What color is the {thing}? A: The {thing} is {color}.")

# --- Days (7 pairs) ---
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, day in enumerate(days):
    qa_pairs.append(f"Q: What day comes after {day}? A: The day after {day} is {days[(i + 1) % 7]}.")

# --- Simple math (9 pairs: 1+1 through 3+3) ---
for a in range(1, 4):
    for b in range(1, 4):
        qa_pairs.append(f"Q: What is {a} plus {b}? A: {a} plus {b} is {a + b}.")

# --- Opposites (10 pairs) ---
opposites = [
    ('hot', 'cold'), ('big', 'small'), ('fast', 'slow'), ('tall', 'short'),
    ('happy', 'sad'), ('light', 'dark'), ('old', 'young'), ('hard', 'soft'),
    ('wet', 'dry'), ('loud', 'quiet'),
]
for a, b in opposites:
    qa_pairs.append(f"Q: What is the opposite of {a}? A: The opposite of {a} is {b}.")

# Repeat 15x to help the tiny model memorize and generalize patterns
all_pairs = qa_pairs * 15
random.shuffle(all_pairs)

corpus = '\n'.join(all_pairs) + '\n'
with open('data/corpus.txt', 'w') as f:
    f.write(corpus)

print(f"Q&A pairs (unique): {len(qa_pairs)}")
print(f"Total lines:        {len(all_pairs)}")
print(f"Corpus size:        {len(corpus):,} chars")
print(f"\nSample Q&A pairs:")
for p in qa_pairs[:8]:
    print(f"  {p}")
