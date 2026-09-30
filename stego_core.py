"""
StegaGen - core algorithm (no user interface in this file).

Idea in one sentence
--------------------
A secret message of 1-3 letters decides WHICH 7 objects (out of 26 possible
object types) must appear in an AI-generated photograph. The receiver runs an
object detector on the photo, sees which objects are present, and turns that
set of objects back into the message using the shared secret key.

Nothing is hidden inside the pixels; the image is never modified after it is
generated. Only the *content* of the scene carries the message.

This file contains five parts:
    Part 1  Settings (vocabulary, alphabet, thresholds)
    Part 2  Key-derived secrets (object ranks, residue, codebook)
    Part 3  Message <-> number conversion
    Part 4  Sender: message -> required objects -> text prompt
    Part 5  Receiver: image -> detections -> message (the blind decoder)

Run `python stego_core.py` to execute a small self-test.
"""

import hashlib
import hmac
from itertools import combinations

# =============================================================================
# Part 1. Settings
# =============================================================================

# 26 object types that the YOLO detector (trained on the COCO dataset) knows.
# They were chosen because they are detected reliably, do not look alike, can
# plausibly appear together in a park scene, and image generators rarely add
# them by themselves. The list is public; only the key is secret.
VOCABULARY = [
    "bicycle", "motorcycle", "horse", "dog", "umbrella", "backpack", "suitcase",
    "frisbee", "kite", "skateboard", "surfboard", "tennis racket", "teddy bear",
    "pizza", "cake", "banana", "laptop", "fire hydrant", "boat", "clock",
    "cat", "sheep", "bus", "airplane", "train", "sports ball",
]
N = len(VOCABULARY)   # 26 object types in total
K = 7                 # every carrier image contains exactly 7 of them

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LETTERS = 3
MESSAGE_COUNT = 26 + 26**2 + 26**3   # all messages of length 1, 2 or 3 = 18,278

# Detector confidence thresholds used by the decoder (fixed, not adjustable,
# so the sender and the receiver always run exactly the same decoder).
LOW = 0.15    # confidence <= LOW  -> object is treated as absent
HIGH = 0.45   # confidence >= HIGH -> object is treated as present
MAX_ERRORS = 1.5   # accept the best match only if it is this close ...
MIN_GAP = 1.0      # ... and at least this much better than the second best

# The same version string as the full StegoArena 3-letter profile, so images
# made with either app decode in the other.
VERSION = "stegoarena-v10-presence-set-3L"

# Object detector settings (also fixed).
DETECTOR_WEIGHTS = "yolo11s.pt"
DETECTOR_SETTINGS = dict(imgsz=640, conf=0.05, iou=0.5, max_det=300, device="cpu", verbose=False)


# =============================================================================
# Part 2. Secrets derived from the shared key
# =============================================================================

def keyed_hash(key, text):
    """HMAC-SHA256: a secret 'fingerprint' of `text` that only key holders can compute."""
    if not key:
        raise ValueError("The shared key must not be empty.")
    return hmac.new(key.encode(), f"{VERSION}|{text}".encode(), hashlib.sha256).digest()


def secret_object_order(key):
    """Step A: give every object type a secret rank 0..25.

    We sort the vocabulary by the keyed hash of each name. Without the key the
    order looks random; with the key it is always the same.
    """
    return sorted(VOCABULARY, key=lambda name: keyed_hash(key, "rank|" + name))


def secret_residue(key):
    """Step B: a secret number a in 0..25 that selects which object sets are valid."""
    return int.from_bytes(keyed_hash(key, "residue"), "big") % N


_codebook_cache = {}


def build_codebook(key):
    """Step C: the secret list of valid 7-object sets (the 'codebook').

    A set of 7 ranks is VALID when (sum of its ranks) mod 26 == a.
    Why this rule? If you replace one object by another, the sum changes by
    something between 1 and 25, so the new set cannot also be valid. Therefore
    any two valid sets differ in at least TWO objects. That is what lets the
    decoder repair one missing or one extra object (like a simple
    error-correcting code).

    The valid sets are then shuffled by the key; message number m uses the
    m-th set in this secret list. There are 25,300 valid sets but only 18,278
    messages, so about 28% of sets are unused - finding one of those means
    "wrong key or damaged image".
    """
    if key in _codebook_cache:
        return _codebook_cache[key]
    order = secret_object_order(key)
    a = secret_residue(key)
    valid_sets = [s for s in combinations(range(N), K) if sum(s) % N == a]
    valid_sets.sort(key=lambda s: keyed_hash(key, "cw|" + ",".join(map(str, s))))
    book = {"order": order, "residue": a, "sets": valid_sets}
    _codebook_cache[key] = book
    return book


# =============================================================================
# Part 3. Message <-> number
# =============================================================================

def clean_message(message):
    message = message.strip().upper()
    if not 1 <= len(message) <= MAX_LETTERS:
        raise ValueError("The message must have 1 to 3 letters.")
    if any(ch not in ALPHABET for ch in message):
        raise ValueError("Only the letters A-Z are allowed.")
    return message


def message_to_number(message):
    """'A'->0 ... 'Z'->25, 'AA'->26 ... 'ZZ'->701, 'AAA'->702 ... 'ZZZ'->18277."""
    message = clean_message(message)
    offset = sum(26**length for length in range(1, len(message)))  # skip shorter messages
    value = 0
    for ch in message:                                             # base-26 number
        value = value * 26 + ALPHABET.index(ch)
    return offset + value


def number_to_message(number):
    """Exact inverse of message_to_number."""
    length = 1
    while number >= 26**length:
        number -= 26**length
        length += 1
    letters = []
    for _ in range(length):
        number, digit = divmod(number, 26)
        letters.append(ALPHABET[digit])
    return "".join(reversed(letters))


# =============================================================================
# Part 4. Sender side
# =============================================================================

def encode(message, key):
    """Message -> the 7 objects that must be in the picture (and the 19 that must not)."""
    message = clean_message(message)
    book = build_codebook(key)
    number = message_to_number(message)
    chosen_ranks = book["sets"][number]
    required = [book["order"][r] for r in chosen_ranks]
    forbidden = [name for name in book["order"] if name not in required]
    return {
        "message": message,
        "number": number,
        "bits": format(number, "015b"),
        "required": required,
        "forbidden": forbidden,
    }


PHRASES = {
    "bicycle": "a bicycle parked on its kickstand",
    "motorcycle": "a parked motorcycle at the edge of the path",
    "horse": "a horse grazing or being led along a bridle path",
    "dog": "a dog relaxing or playing on the grass",
    "umbrella": "an open patio or beach umbrella standing on its own",
    "backpack": "a backpack resting on the ground",
    "suitcase": "a suitcase standing upright on the pavement",
    "frisbee": "a frisbee lying on the grass or caught mid-air",
    "kite": "a colourful kite flying in the sky",
    "skateboard": "a skateboard lying on the path",
    "surfboard": "a surfboard leaning against a tree or railing",
    "tennis racket": "a tennis racket resting on a low wall or blanket",
    "teddy bear": "a teddy bear sitting on a picnic blanket",
    "pizza": "a pizza in an open box on a picnic blanket",
    "cake": "a cake on a picnic blanket",
    "banana": "a bunch of bananas on a picnic blanket",
    "laptop": "an open laptop on a picnic blanket",
    "fire hydrant": "a fire hydrant at the edge of the park path",
    "boat": "a small boat on the lake",
    "clock": "a tall park post clock",
    "cat": "a cat sitting on the low stone wall",
    "sheep": "a sheep grazing in the meadow beyond the park",
    "bus": "a city bus on the road at the far edge of the park",
    "airplane": "an airplane flying low across the sky",
    "train": "a train crossing a bridge in the background",
    "sports ball": "a football (soccer ball) on the grass",
}


def make_prompt(carrier):
    """Text prompt for ANY image generator. It names objects only - no positions."""
    wanted = "\n".join(f"- {PHRASES[o]} (exactly one {o})" for o in sorted(carrier["required"]))
    banned = ", ".join(sorted(carrier["forbidden"]))
    return (
        "A natural, candid, photorealistic photograph of a sunny public park beside a lake, "
        "with grass, trees, a paved path, a low stone wall, and water in the background.\n\n"
        "The scene must naturally include each of the following, clearly visible, reasonably large, "
        "and not heavily occluded or cut off by the frame edge:\n"
        f"{wanted}\n\n"
        "Place these objects wherever they look most natural. There is no required position, order, "
        "size, or arrangement.\n\n"
        "The photograph must NOT contain any of the following anywhere, including in the background, "
        f"on signs, or as toys or pictures: {banned}.\n\n"
        "Avoid people; if any appear they must be small, distant, and carrying nothing. "
        "No written text, captions, logos, or watermarks.\n\n"
        "Style: natural daylight, documentary photography, realistic lens perspective, high detail, "
        "plausible scale, no surrealism, no collage."
    )


# =============================================================================
# Part 5. Receiver side - the blind decoder
# =============================================================================

def detect_objects(image, model):
    """Run YOLO once with fixed settings. Returns a list of (label, confidence, box)."""
    import numpy as np  # imported here so the self-test does not need numpy/YOLO

    pixels = np.asarray(image.convert("RGB"))
    result = model.predict(source=pixels, **DETECTOR_SETTINGS)[0]
    detections = []
    for box in result.boxes:
        label = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append((label, confidence, (x1, y1, x2, y2)))
    return detections


def presence_score(confidence):
    """Turn a detector confidence into a 'how present is it' score between 0 and 1."""
    if confidence <= LOW:
        return 0.0
    if confidence >= HIGH:
        return 1.0
    return (confidence - LOW) / (HIGH - LOW)   # uncertain zone: partial score


def decode(detections, key):
    """Blind decoder: detections + key -> message (or a reason for failure).

    This function never sees the intended message. The sender's verification
    step and the receiver call exactly this function.
    """
    book = build_codebook(key)
    order = book["order"]

    # Step 1: best confidence for each vocabulary object. Duplicates collapse
    # (two dogs = one 'dog'), and everything outside the vocabulary is ignored.
    best_conf = {name: 0.0 for name in order}
    for label, confidence, _box in detections:
        if label in best_conf:
            best_conf[label] = max(best_conf[label], confidence)

    # Step 2: presence score per object, listed in the secret key order.
    score = [presence_score(best_conf[name]) for name in order]
    table = [
        {"rank": r, "object": name, "confidence": round(best_conf[name], 3), "presence": round(score[r], 2)}
        for r, name in enumerate(order)
    ]
    if sum(score) == 0:
        return {"ok": False, "reason": "None of the 26 vocabulary objects was detected.", "table": table}

    # Step 3: compare the evidence with every valid set.
    #   error(set) = sum over objects IN the set of (1 - score)   (missing objects)
    #              + sum over objects NOT in the set of score      (extra objects)
    # This is a 'soft' count of how many objects would have to be wrong.
    best_error, second_error, best_number = float("inf"), float("inf"), None
    total_score = sum(score)
    for number, s in enumerate(book["sets"]):
        # same formula, rearranged so we only loop over the 7 members:
        error = total_score + sum(1 - 2 * score[r] for r in s)
        if error < best_error:
            second_error, best_error, best_number = best_error, error, number
        elif error < second_error:
            second_error = error

    best_objects = [order[r] for r in book["sets"][best_number]]
    info = {
        "table": table,
        "best_objects": best_objects,
        "best_error": round(best_error, 2),
        "second_error": round(second_error, 2),
        "number": best_number,
    }

    # Step 4: accept only a clear, close, assigned match.
    # (1e-9 = tolerance for floating-point rounding, same as the full StegoArena app)
    if best_error > MAX_ERRORS + 1e-9:
        return {"ok": False, "reason": f"Closest valid set needs {best_error:.2f} object corrections (limit {MAX_ERRORS}).", **info}
    if second_error - best_error < MIN_GAP - 1e-9:
        return {"ok": False, "reason": "Two valid sets match almost equally well - refusing to guess.", **info}
    if best_number >= MESSAGE_COUNT:
        return {"ok": False, "reason": "Matched a set that no message uses - wrong key or damaged image.", **info}

    return {"ok": True, "message": number_to_message(best_number), "bits": format(best_number, "015b"), **info}


# =============================================================================
# Self-test (no image needed): python stego_core.py
# =============================================================================

if __name__ == "__main__":
    import random

    key = "UNIGE2026"
    rng = random.Random(0)
    for word in ["A", "HI", "CAT", "SOS", "ZZZ"]:
        carrier = encode(word, key)
        perfect = [(o, 0.9, None) for o in carrier["required"]]
        one_missing = perfect[1:]
        one_extra = perfect + [(carrier["forbidden"][0], 0.8, None)]
        duplicates = perfect + [(carrier["required"][0], 0.5, None)] * 3 + [("person", 0.9, None)]
        two_wrong = one_missing + [(carrier["forbidden"][0], 0.8, None)]
        assert decode(perfect, key)["message"] == word
        assert decode(one_missing, key)["message"] == word
        assert decode(one_extra, key)["message"] == word
        assert decode(duplicates, key)["message"] == word
        assert decode(two_wrong, key)["ok"] is False
        assert decode(perfect, "wrong key").get("message") != word
        print(f"{word:>3}: {', '.join(carrier['required'])}  -> ok")
    for _ in range(2000):
        n = rng.randrange(MESSAGE_COUNT)
        assert message_to_number(number_to_message(n)) == n
    print("All self-tests passed.")
