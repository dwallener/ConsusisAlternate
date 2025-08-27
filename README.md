## New and Improved Version

# Status

Tech marches on. This is so much simpler to make now than it was 18 months ago.

This generates NoSAS and StopBang! scores. All code is in Python The key bits of those are BMI, age, gender estimation. Previously training models to do that, based on this dataset:

https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset

It is ~68k in size, covers many ethnicities, and is well-annotated. Using transformers it is possible to train on this data and get reasonable results. However...the latest LLMs also now do this, at least as well and much more simply. This is basically it...

```
SYSTEM_MESSAGE = """You are a clinical screening assistant. The user provides a frontal and a profile face photo.
Task:
1) Estimate: age range (years), BMI range, sex-at-birth (best estimate), and neck circumference range (cm).
2) ...
3) Compute a NoSAS score *estimate*. Use: neck ≥ 40cm = +4; BMI 25–<30 = +3; BMI ≥ 30 = +5; snoring +2 (assume unknown unless clearly visible from context),
   age > 55 = +4; male = +2. Return the score and 'High' if ≥8; else 'Low/Intermediate'.
4) Compute STOP-Bang *range* given your best estimates and marking items you cannot infer (snoring/tiredness/observed apnea/high blood pressure are typically unknown from images).
   Items: snoring, tiredness, observed apnea, high BP, BMI>35, age>50, neck>40cm, male. Return min/max and inferred categories for low/high assumptions.
5) Output only the JSON defined by SCHEMA_JSON. Keep ranges realistic and state assumptions briefly.
6) DO NOT diagnose. This is for research. Be conservative with claims.
```

This was NOT possible to do well when I started working on Consusis - training my own was the only feasible option. What's more interesting is that this, now, also works:

```
2) Identify the three most important morphological features related to obstructive sleep apnea (OSA) visible in these images,
   and give their qualitative values (e.g., 'thick neck', 'retrognathia/mandibular retrusion', 'crowded oropharynx', 'midface hypoplasia', etc.).
```

That, frankly, blew my mind. Because there is NO WAY someone at OpenAI decided to add morphology for apnea to the GPT4+ training dataset. Which means it figured out how to do this simply by reading the published papers. This raises so many questions...

# Usage

To use this in a field setting requires:

- First, make sure the repo is private
- Create Streamlit account
- On Streamlit account, create a new app and link it to the private repo
- > With a private account, there are some credentials to take care of, it's straightforward

Optional:

- Create a domain name and point it at the Streamlit deployment

Streamlit handles virtually all aspects of hosting python code. Honestly, it's close to magic. 

# Other repos of interest:

Contains the pricing/costing model along with Monte Carlo simulation:

https://github.com/dwallener/ConsusisShareable


