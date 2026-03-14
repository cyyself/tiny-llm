#!/usr/bin/env python3
"""Generate a large, repetitive corpus for training the tiny LLM."""
import random
random.seed(42)

subjects = ['The cat', 'The dog', 'The bird', 'The fish', 'The horse', 'The rabbit',
            'The old man', 'The young girl', 'The teacher', 'The student',
            'The farmer', 'The doctor', 'The painter', 'The writer', 'The cook',
            'The boy', 'The girl', 'The child', 'The woman', 'The man',
            'The king', 'The queen', 'The prince', 'The princess', 'The knight',
            'A small bird', 'A tall tree', 'A bright star', 'A dark cloud', 'A red flower']

verbs_past = ['walked', 'sat', 'stood', 'ran', 'slept', 'played', 'worked', 'sang',
              'danced', 'smiled', 'laughed', 'cried', 'looked', 'watched', 'listened']

locations = ['in the garden', 'on the hill', 'by the river', 'near the forest',
             'in the house', 'on the road', 'by the lake', 'near the mountain',
             'in the field', 'on the bridge', 'by the sea', 'near the castle',
             'in the park', 'on the street', 'by the fire', 'near the window',
             'in the kitchen', 'on the roof', 'by the door', 'near the tree']

times = ['in the morning', 'in the evening', 'at night', 'at dawn',
         'in the afternoon', 'at sunset', 'at noon', 'before dark',
         'after the rain', 'during the storm', 'on a warm day', 'on a cold night']

adjectives = ['beautiful', 'small', 'large', 'old', 'young', 'bright', 'dark',
              'quiet', 'loud', 'gentle', 'strong', 'wise', 'kind', 'brave', 'fast',
              'slow', 'tall', 'short', 'happy', 'sad', 'warm', 'cold', 'soft', 'hard']

nouns = ['the sun', 'the moon', 'the stars', 'the wind', 'the rain', 'the snow',
         'the river', 'the mountain', 'the forest', 'the sea', 'the sky', 'the earth',
         'the fire', 'the water', 'the light', 'the shadow', 'the flower', 'the tree',
         'the stone', 'the bird', 'the fish', 'the book', 'the song', 'the story']

actions = ['to sing a song', 'to tell a story', 'to build a house', 'to plant a tree',
           'to cook a meal', 'to read a book', 'to write a letter', 'to paint a picture',
           'to catch a fish', 'to climb a hill', 'to cross the river', 'to find the path']

colors = ['red', 'blue', 'green', 'gold', 'silver', 'white', 'black', 'brown', 'yellow', 'purple']
animals = ['cat', 'dog', 'bird', 'fish', 'horse', 'rabbit', 'fox', 'deer', 'bear', 'wolf']
foods = ['bread', 'cheese', 'fruit', 'soup', 'rice', 'meat', 'fish', 'cake', 'pie', 'tea']

lines = []

# Pattern 1: Subject verb location time
for _ in range(200):
    lines.append(f'{random.choice(subjects)} {random.choice(verbs_past)} {random.choice(locations)} {random.choice(times)}.')

# Pattern 2: The ADJ NOUN VERB location
for _ in range(200):
    a = random.choice(adjectives)
    n = random.choice(nouns).replace('the ', '')
    lines.append(f'The {a} {n} {random.choice(verbs_past)} {random.choice(locations)}.')

# Pattern 3: Subject wanted ACTION location
for _ in range(150):
    lines.append(f'{random.choice(subjects)} wanted {random.choice(actions)} {random.choice(locations)}.')

# Pattern 4: NOUN is ADJ and ADJ
for _ in range(150):
    n = random.choice(nouns)
    a1 = random.choice(adjectives)
    a2 = random.choice(adjectives)
    while a2 == a1:
        a2 = random.choice(adjectives)
    lines.append(f'{n[0].upper()}{n[1:]} is {a1} and {a2}.')

# Pattern 5: There was a ADJ ANIMAL location
for _ in range(150):
    lines.append(f'There was a {random.choice(adjectives)} {random.choice(animals)} {random.choice(locations)}.')

# Pattern 6: Subject ate FOOD and drank tea
for _ in range(100):
    lines.append(f'{random.choice(subjects)} ate {random.choice(foods)} and drank tea.')

# Pattern 7: The COLOR NOUN was location
for _ in range(150):
    c = random.choice(colors)
    n = random.choice(nouns).replace('the ', '')
    lines.append(f'The {c} {n} was {random.choice(locations)}.')

# Pattern 8: Longer narrative sentences
templates = [
    '{s} {v} {l} and then went home.',
    '{s} {v} {l} because {n} was {a}.',
    'Every day {s} {v} {l} to see {n}.',
    'Once upon a time {s} {v} {l}.',
    '{s} could see {n} from {l}.',
    'When {n} appeared, {s} {v} {l}.',
    '{s} loved {l} because it was {a}.',
    'It was a {a} day when {s} {v} {l}.',
]
for _ in range(300):
    t = random.choice(templates)
    line = t.format(
        s=random.choice(subjects).lower(),
        v=random.choice(verbs_past),
        l=random.choice(locations),
        n=random.choice(nouns),
        a=random.choice(adjectives)
    )
    line = line[0].upper() + line[1:]
    lines.append(line)

random.shuffle(lines)

# Group into paragraphs
paragraphs = []
i = 0
while i < len(lines):
    n = random.randint(5, 7)
    paragraphs.append(' '.join(lines[i:i + n]))
    i += n

corpus = '\n\n'.join(paragraphs) + '\n'
with open('data/corpus.txt', 'w') as f:
    f.write(corpus)
print(f'Corpus size: {len(corpus):,} chars')
print(f'Lines: {len(lines)}')
print(f'Paragraphs: {len(paragraphs)}')
print(f'Preview:\n{corpus[:300]}')
