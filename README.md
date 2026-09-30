# StegaGen

StegaGen is a short version of StegoArena. The code is two Python files:

* `stego_core.py`: the whole algorithm (key secrets, codebook, prompt, blind decoder) plus a self-test.
* `app.py`: the Streamlit interface with the tabs **Generate & Verify**, **Decode** and **How It Works**.

A 1–3 letter secret decides which 7 of 26 objects appear in an AI-generated photo. The receiver needs only the image and the shared key. The protocol is identical to the full StegoArena `3letter` profile, so images work in either app.

## Run locally
```bash
pip install -r requirements.txt
python stego_core.py      # self-test, no image or model needed
streamlit run app.py
```
`yolo11s.pt` is downloaded automatically on first use.

## Streamlit Community Cloud
Set the main file to `StegaGen/app.py`. `requirements.txt` and `packages.txt` in this folder are used.
