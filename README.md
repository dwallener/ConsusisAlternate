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

This was NOT possible to do well when I started working on Consusis - training my own was the only feasible option. 'schema_json' is defined in the code itself - it is self-contained and extensible.

What's more interesting is this, now, also works:

```
2) Identify the three most important morphological features related to obstructive sleep apnea (OSA) visible in these images,
   and give their qualitative values (e.g., 'thick neck', 'retrognathia/mandibular retrusion', 'crowded oropharynx', 'midface hypoplasia', etc.).
```

That, frankly, blew my mind. Because there is NO WAY someone at OpenAI decided to add morphology for apnea to the GPT4+ training dataset. Which means it figured out how to do this simply by reading the published papers. And it is, again, doing it at least as well as a reasonable-size, custom-trained transformer/ML model can do it. This raises so many questions...

Regarding AHI: the code used a simple method for AHI prediction. Each component - NoSAS, StopBang, Morphology - contributes a point. AHI has four levels of severity - none, light, medium, high - the scores map to those values from 0 to 3. 

Regarding LLMs: this has only been tested with ChatGPT. The latest xAI is also a very promising candidate for this.

# Usage

To use this in a field setting requires:

- First, make sure the repo is private
- Create Streamlit account
- On Streamlit account, create a new app and link it to the private repo
- > With a private account, there are some credentials to take care of, it's straightforward
- Create and link OpenAI (or xAI etc) developer/API account and link credentials in the python code
- Streamlit provides a public direct link to the app

Optional:

- Create a domain name and point it at the Streamlit deployment

Streamlit handles virtually all aspects of hosting python code. Honestly, it's close to magic. 

# Note on Snorealyzer:

The learnings from above should raise an obvious question - since you can now "talk" with all the frontier models. Yes, I tried this. Yes, it works, based on a limited number of samples. Yes, absolutely, this path should also be pursued for Snoralyzer. It would take literally 10 minutes to put up a site that does this...there is IMO no longer any point at all in pursuing a custom-trainined model for this.

The work-in-progress code for localized training is in 'snoralyzer-wip'. The audio files are too large for repo storage, they will have to be fetched separately from the link provided earlier. There is code for moving between the format in the dataset and audio-useable.

# Other links of interest:

Contains the pricing/costing model along with Monte Carlo simulation:

https://github.com/dwallener/ConsusisShareable

PSG/Audio recordings (1200 hours, x2) for future Snorealyzer work:

https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios

Streamlit Cloud: fastest way to deploy to "any" device

https://streamlit.io/cloud

OpenAI Developer:

https://platform.openai.com/docs/overview

Azure AI Credits:

https://portal.startups.microsoft.com/signup






