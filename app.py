"""
StegaGen - Streamlit user interface.

Tabs:
    1. Generate & Verify  - sender: message -> prompt -> generated image -> check
    2. Decode             - receiver: image + key -> message
    3. How It Works       - step-by-step explanation of the algorithm

All algorithm code lives in stego_core.py.
"""

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import stego_core as core

st.set_page_config(page_title="StegaGen", layout="wide")
DEFAULT_KEY = "UNIGE2026"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_resource
def load_detector():
    from ultralytics import YOLO  # downloaded automatically on first use if missing
    return YOLO(core.DETECTOR_WEIGHTS)


def run_blind_decoder(image_bytes, key):
    """The ONLY decoding path. Used by the sender's check and by the receiver."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    detections = core.detect_objects(image, load_detector())
    return image, detections, core.decode(detections, key)


def draw_boxes(image, detections):
    """Draw boxes around vocabulary objects only (green = sure, orange = uncertain)."""
    picture = image.copy()
    pen = ImageDraw.Draw(picture)
    width = max(2, picture.width // 300)
    for label, confidence, (x1, y1, x2, y2) in detections:
        if label in core.VOCABULARY and confidence > core.LOW:
            colour = (0, 170, 80) if confidence >= core.HIGH else (230, 150, 0)
            pen.rectangle([x1, y1, x2, y2], outline=colour, width=width)
            pen.text((x1 + 4, max(0, y1 - 14)), f"{label} {confidence:.2f}", fill=colour)
    return picture


def show_result_details(image, detections, result, expected=None):
    left, right = st.columns(2)
    left.image(image, caption="Received image (unchanged)", width="stretch")
    right.image(draw_boxes(image, detections), caption="Detected vocabulary objects", width="stretch")

    table = pd.DataFrame(result["table"])
    table["state"] = table["presence"].map(lambda p: "present" if p == 1 else "absent" if p == 0 else "uncertain")
    if "best_objects" in result:
        table["in decoded set"] = table["object"].isin(result["best_objects"])
    if expected is not None:
        table["expected"] = table["object"].isin(expected["required"])
    with st.expander("Decoder details", expanded=not result["ok"]):
        st.write("Objects in the secret key order:")
        st.dataframe(table, hide_index=True, width="stretch")
        if "best_error" in result:
            st.write(
                f"Closest valid set: {result['best_error']} corrections needed · "
                f"second closest: {result['second_error']} · secret set number: {result['number']}"
            )


# -----------------------------------------------------------------------------
# Tab 1: Generate & Verify
# -----------------------------------------------------------------------------

def generate_tab():
    st.header("Generate & Verify")
    key = st.text_input("Shared key", value=DEFAULT_KEY, type="password", key="gen_key")
    message = st.text_input("Secret message (1 to 3 letters, A-Z)", max_chars=3, placeholder="CAT")
    if not message or not key:
        return
    try:
        carrier = core.encode(message, key)
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Step 1 - Required scene content")
    st.write(f"Message **{carrier['message']}** is number **{carrier['number']}** (bits `{carrier['bits']}`).")
    st.write("Must be in the picture (one of each): " + ", ".join(sorted(carrier["required"])))
    st.write("Must NOT be in the picture: " + ", ".join(sorted(carrier["forbidden"])))

    st.subheader("Step 2 - Generate the image")
    prompt = core.make_prompt(carrier)
    st.write("Paste this prompt into any image generator (free or paid, local or online).")
    st.code(prompt, language=None)
    st.download_button("Download prompt", prompt, file_name="stegagen_prompt.txt")

    st.subheader("Step 3 - Verify before sending")
    upload = st.file_uploader("Upload the generated image", type=["png", "jpg", "jpeg", "webp"], key="gen_img")
    if upload is None:
        return
    with st.spinner("Running the receiver's blind decoder..."):
        image, detections, result = run_blind_decoder(upload.getvalue(), key)

    # The comparison with the intended message happens only AFTER blind decoding.
    if result["ok"] and result["message"] == carrier["message"]:
        st.success(f"Verified: the decoder reads '{result['message']}'. You can send this exact image file.")
    elif result["ok"]:
        st.error(f"The decoder reads '{result['message']}' instead of '{carrier['message']}'. Generate a new image.")
    else:
        st.error(f"The decoder could not read this image: {result['reason']} Generate a new image.")
    show_result_details(image, detections, result, expected=carrier)


# -----------------------------------------------------------------------------
# Tab 2: Decode
# -----------------------------------------------------------------------------

def decode_tab():
    st.header("Decode")
    st.write("You need only the received image and the shared key.")
    key = st.text_input("Shared key", value=DEFAULT_KEY, type="password", key="dec_key")
    upload = st.file_uploader("Upload the received image", type=["png", "jpg", "jpeg", "webp"], key="dec_img")
    if upload is None or not key:
        return
    with st.spinner("Decoding..."):
        image, detections, result = run_blind_decoder(upload.getvalue(), key)
    if result["ok"]:
        st.success("Secret message recovered")
        st.markdown(f"## {result['message']}")
    else:
        st.error(f"No message found: {result['reason']}")
    show_result_details(image, detections, result)


# -----------------------------------------------------------------------------
# Tab 3: How It Works
# -----------------------------------------------------------------------------

def how_it_works_tab():
    st.header("How It Works")
    st.markdown(
        f"""
### 1. The idea

Classical steganography hides bits inside the pixels of an existing image.
StegaGen instead uses **generative, coverless steganography**: the secret
decides *what the picture shows*, and an AI image generator paints a
completely normal photograph with that content. The pixels are never edited
afterwards, so there is nothing hidden "inside" the file to find.

The only property we rely on is **which objects are present**. Positions,
sizes, number of copies and layout are all ignored, because image generators
reproduce *what* you ask for far more reliably than *where* you ask for it.

### 2. Building blocks

| Item | Value |
|---|---|
| Object vocabulary | {core.N} public object types known to the YOLO detector |
| Objects per image | exactly {core.K} different ones |
| Message | 1-3 letters A-Z, {core.MESSAGE_COUNT:,} possible messages (15 bits) |
| Shared key | any secret password known to both sides |
| Detector | `{core.DETECTOR_WEIGHTS}`, 640 px, CPU, fixed settings |

### 3. What the shared key does

The key is fed into **HMAC-SHA256**, a keyed hash function. From it we derive
three secrets:

1. **Secret object order.** Sort the {core.N} object names by `HMAC(key, name)`.
   Each object gets a secret rank from 0 to {core.N - 1}.
2. **Secret residue a.** `a = HMAC(key, "residue") mod {core.N}`.
3. **Secret codebook.** Take every set of {core.K} ranks whose sum satisfies
   `sum mod {core.N} = a`, then shuffle these sets by their HMAC value.
   Message number *m* is represented by the *m*-th set in this list.

Without the key an observer does not know the order, the residue, or which set
means which message.

### 4. Why "sum mod 26 = a"? (the error-correcting trick)

Suppose one object in a valid set is swapped for another. The sum then changes
by the difference of their ranks, which is between 1 and 25, so the new sum
cannot again equal *a* (mod 26). Therefore **no two valid sets differ by a
single swap**; they always differ in at least two objects.

This gives us a small error-correcting code:

* if the generator **forgets one object**, the picture has 6 of the 7 objects.
  Exactly one valid set is 1 step away, so it is still found.
* if the generator **adds one forbidden object**, the picture has 8. Again exactly one valid set is 1 step away.
* if **two things go wrong**, several valid sets are equally close, so the decoder refuses to guess
  instead of returning a wrong message.

About {core.MESSAGE_COUNT:,} of the ~25,300 valid sets are used for messages. The rest are
intentionally unused. Landing on one of them means "wrong key or damaged image".

### 5. Sender algorithm

```
INPUT: message (1-3 letters), key
1. m   <- message_to_number(message)          # 'A'=0 ... 'ZZZ'=18277
2. S   <- codebook(key)[m]                     # 7 secret ranks
3. required  <- objects with ranks in S
   forbidden <- the other 19 objects
4. prompt <- "photo of a park with <required>, and NO <forbidden>"
5. image  <- any_image_generator(prompt)
6. result <- BLIND_DECODE(image, key)          # exactly the receiver's code
7. if result == message: send image
   else: go back to step 5
```

**Message to number.** Messages are counted in order A..Z, AA..ZZ, AAA..ZZZ. For
length L the value is `(26 + ... + 26^(L-1)) + base26(message)`.
Example: `CAT` = 26 + 676 + (2*676 + 0*26 + 19) = 2073.

### 6. Receiver algorithm (the blind decoder)

```
INPUT: image, key            # the expected message is NOT an input
1. detections <- YOLO(image) with fixed settings
2. for each vocabulary object v:
       c_v <- highest confidence among all detections of v (0 if none)
       p_v <- presence score:
              0                       if c_v <= {core.LOW}
              1                       if c_v >= {core.HIGH}
              (c_v-{core.LOW})/({core.HIGH}-{core.LOW})     otherwise
3. for each valid set S in codebook(key):
       error(S) = sum_(v in S) (1 - p_v)  +  sum_(v not in S) p_v
4. S* <- set with the smallest error; E1 = error(S*), E2 = second smallest
5. accept S* only if  E1 <= {core.MAX_ERRORS}   (at most about one wrong object)
                 and  E2 - E1 >= {core.MIN_GAP}  (clear winner)
                 and  index(S*) < {core.MESSAGE_COUNT}   (a real message)
6. return number_to_message(index(S*))
```

**Why the error formula?** `p_v` says how strongly object *v* is seen. If a
candidate set says *v* should be present, we pay `1 - p_v` when it is missing.
If the set says *v* should be absent, we pay `p_v` when it is visible. The
total is a "soft" count of wrong objects. It is 0 for a perfect image, 1 for
one missing or one extra object, and fractional when the detector is unsure.

**Why take the highest confidence per object?** Two dogs, or the same dog
detected twice, still mean just "dog is present". Repeated objects therefore
can never confuse the decoder, and neither can the order in which YOLO lists
them.

### 7. Why the sender check matters

The sender runs **the same function** as the receiver (`decode` in
`stego_core.py`) on the exact file that will be sent. If it decodes correctly
on the sender's machine, the receiver, running the same software on the same
file, gets the same result.

### 8. Worked example (key = "{DEFAULT_KEY}")
"""
    )
    book = core.build_codebook(DEFAULT_KEY)
    example = core.encode("CAT", DEFAULT_KEY)
    st.write(f"Secret residue a = **{book['residue']}**, valid sets = **{len(book['sets']):,}**.")
    st.dataframe(
        pd.DataFrame({"secret rank": range(core.N), "object": book["order"]}).T,
        width="stretch",
    )
    st.write(
        f"Message **CAT** → number **{example['number']}** → set of ranks "
        f"{list(book['sets'][example['number']])} → objects **{', '.join(example['required'])}**."
    )
    st.markdown(
        """
### 9. Limitations

* About 15 bits per image, so the message is short by design. Reliability was chosen over capacity.
* The generator must include 7 objects and avoid 19. Some attempts fail and need regenerating.
* The key scrambles a mapping. This is not encryption, and a wrong key occasionally (≈3%) yields some other word.
* Educational prototype.
"""
    )


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------

st.title("StegaGen")
st.caption("Generative coverless steganography: the secret decides which objects appear in an AI-generated photo.")

tab_generate, tab_decode, tab_how = st.tabs(["Generate & Verify", "Decode", "How It Works"])
with tab_generate:
    generate_tab()
with tab_decode:
    decode_tab()
with tab_how:
    how_it_works_tab()
