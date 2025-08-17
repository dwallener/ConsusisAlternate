
import streamlit as st
from PIL import Image
import io, base64, json, os
from typing import Dict, Any

# -------------------------
# Helpers
# -------------------------

def img_to_data_url(
    img: Image.Image,
    max_side: int = 1024,
    min_side: int = 640,
    quality: int = 85,
    target_bytes: int = 600_000,   # ~0.6 MB
) -> str:
    import io, base64
    from PIL import Image, ImageOps

    # Normalize orientation & mode
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downscale with good resampling
    w, h = img.size
    if max(w, h) > max_side:
        img = img.copy()  # avoid in-place surprises
        img.thumbnail((max_side, max_side), resample=Image.LANCZOS)
    elif max(w, h) < min_side:
        # don't upscale; just proceed
        pass

    def encode(q: int) -> bytes:
        buf = io.BytesIO()
        # strip metadata (no exif/icc) for size
        img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        return buf.getvalue()

    # First try at requested quality
    data = encode(quality)

    # If still large, step quality down to hit target_bytes
    q = quality
    while len(data) > target_bytes and q > 55:
        q -= 5
        data = encode(q)

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def indicator_to_score(level: str) -> float:
    """
    Map indicator string to numeric contribution.
    Expected levels: "low", "none", "moderate", "high", "severe"
    """
    if not level:
        return 0.0
    level = level.lower()
    if level in ("high", "severe"):
        return 1.0
    if level == "moderate":
        return 0.5
    return 0.0

SCHEMA_JSON = {
  "type": "object",
  "properties": {
    "age_years_estimate": {"type": "object", "properties": {"min": {"type":"integer"}, "max":{"type":"integer"}}, "required":["min","max"]},
    "bmi_estimate": {"type": "object", "properties": {"min": {"type":"number"}, "max":{"type":"number"}}, "required":["min","max"]},
    "sex_at_birth_estimate": {"type": "string", "enum": ["male","female","unknown"]},
    "neck_circumference_cm_estimate": {"type": "object", "properties": {"min": {"type":"number"}, "max":{"type":"number"}}, "required":["min","max"]},
    "morphology": {
      "type":"object",
      "properties":{
        "feature_1_name":{"type":"string"},
        "feature_1_value":{"type":"string"},
        "feature_2_name":{"type":"string"},
        "feature_2_value":{"type":"string"},
        "feature_3_name":{"type":"string"},
        "feature_3_value":{"type":"string"},
        "notes":{"type":"string"}
      },
      "required":["feature_1_name","feature_1_value","feature_2_name","feature_2_value","feature_3_name","feature_3_value"]
    },
    "nosas": {
      "type":"object",
      "properties": {
        "score":{"type":"integer"},
        "risk_category":{"type":"string", "enum":["High","Low/Intermediate"]},
        "explanation":{"type":"string"}
      },
      "required":["score","risk_category"]
    },
    "stopbang": {
      "type":"object",
      "properties": {
        "min_score":{"type":"integer"},
        "max_score":{"type":"integer"},
        "category_low_assumption":{"type":"string","enum":["Low","Intermediate","High"]},
        "category_high_assumption":{"type":"string","enum":["Low","Intermediate","High"]},
        "assumptions":{"type":"string"}
      },
      "required":["min_score","max_score","category_low_assumption","category_high_assumption"]
    }
  },
  "required": ["age_years_estimate","bmi_estimate","sex_at_birth_estimate","neck_circumference_cm_estimate","morphology","nosas","stopbang"]
}

SYSTEM_MESSAGE = """You are a clinical screening assistant. The user provides a frontal and a profile face photo.
Task:
1) Estimate: age range (years), BMI range, sex-at-birth (best estimate), and neck circumference range (cm).
2) Identify the three most important morphological features related to obstructive sleep apnea (OSA) visible in these images,
   and give their qualitative values (e.g., 'thick neck', 'retrognathia/mandibular retrusion', 'crowded oropharynx', 'midface hypoplasia', etc.).
3) Compute a NoSAS score *estimate*. Use: neck ≥ 40cm = +4; BMI 25–<30 = +3; BMI ≥ 30 = +5; snoring +2 (assume unknown unless clearly visible from context),
   age > 55 = +4; male = +2. Return the score and 'High' if ≥8; else 'Low/Intermediate'.
4) Compute STOP-Bang *range* given your best estimates and marking items you cannot infer (snoring/tiredness/observed apnea/high blood pressure are typically unknown from images).
   Items: snoring, tiredness, observed apnea, high BP, BMI>35, age>50, neck>40cm, male. Return min/max and inferred categories for low/high assumptions.
5) Output only the JSON defined by SCHEMA_JSON. Keep ranges realistic and state assumptions briefly.
6) DO NOT diagnose. This is for research. Be conservative with claims.
"""


def call_openai(images: Dict[str,str], model: str, api_key: str) -> Dict[str, Any]:

    from openai import AzureOpenAI

    # Configuration - old
    endpoint = "https://damir-mvp-01.openai.azure.com"
    deployment = "gpt-4.1-mini"
    subscription_key = os.getenv("AZURE_OPENAI_KEY", "EnRH032rZqu5kR1benrMSSiYN9c7lK85hpnjJLRyQYFC48Lm6npOJQQJ99BFAC4f1cMXJ3w3AAABACOGf7hY")
    api_version = "2024-12-01-preview"

    model_id_used = deployment
    llm_backend = deployment

    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version
    )

    content = [
        {"type": "text", "text": "Front and profile photos are provided below."},
        {"type": "image_url", "image_url": {"url": images["front"]}},
        {"type": "image_url", "image_url": {"url": images["profile"]}},
        # NEW: user-provided answers override image inference
        {"type": "text", "text": (
            f"User-provided items for scoring — "
            f"snoring: {'yes' if ui_snoring else 'no'}, "
            f"tiredness: {'yes' if ui_tired else 'no'}, "
            f"high_bp: {'yes' if ui_highbp else 'no'}. "
            "Use these values for STOP-Bang and NoSAS instead of inferring from images.\n"
            "Return only JSON matching the required schema."
        )},
        {"type": "text", "text": "Return only JSON matching the required schema."},
    ]

    # Make the rule explicit in the system prompt
    sys_msg = SYSTEM_MESSAGE + "\n7) You MUST output JSON that exactly matches SCHEMA_JSON (field names, types). Do not invent other keys or formats."

    resp = client.chat.completions.create(
        model=model_id_used,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content": sys_msg},
            {"role":"user","content": content}
        ],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content
    return json.loads(txt)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="LLM OSA Risk Indicator (Experimental)", page_icon="😴", layout="wide")
st.title("LLM‑only Sleep Apnea Risk Indicator (Experimental)")
st.caption("This prototype sends your two photos to an OpenAI GPT vision model and asks it to estimate age/BMI/neck circumference ranges, "
           "three OSA‑relevant morphology features, and to compute NoSAS and STOP‑Bang estimates. "
           "**Not medical advice. Research/education only.**")

with st.sidebar:
    st.subheader("API & Model")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    st.caption("These models accept images and text and can output structured JSON.")
    st.divider()
    st.subheader("Privacy note")
    st.caption("Images are sent to OpenAI to obtain estimates. Do not upload sensitive photos without consent.")

import io
from PIL import Image, ImageOps

def _to_pil(file_or_bytes) -> Image.Image:
    # Accepts an UploadedFile, Camera bytes, or BytesIO → normalized, auto-rotated PIL image
    if hasattr(file_or_bytes, "read"):  # UploadedFile
        img = Image.open(file_or_bytes)
    elif isinstance(file_or_bytes, (bytes, bytearray)):  # camera bytes
        img = Image.open(io.BytesIO(file_or_bytes))
    else:
        raise ValueError("Unsupported input type")
    return ImageOps.exif_transpose(img.convert("RGB"))

# --- In your UI section, replace the two uploader blocks with this ---
use_cam = st.toggle("Use device camera instead of file upload", value=False)

# --- user-provided symptom buttons (default NO unless toggled) ---
bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    ui_snoring = st.toggle("Snoring", value=False, help="Loud snoring (binary for scoring).")
with bcol2:
    ui_tired = st.toggle("Daytime tiredness", value=False, help="Excessive daytime sleepiness.")
with bcol3:
    ui_highbp = st.toggle("High BP", value=False, help="Diagnosed or treated hypertension.")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Frontal photo")
    if use_cam:
        cam_front = st.camera_input("Take frontal photo")
        front_img = _to_pil(cam_front.getvalue()) if cam_front else None
    else:
        front_file = st.file_uploader("Upload frontal photo", type=["jpg","jpeg","png"], key="front")
        front_img = _to_pil(front_file) if front_file else None
    if front_img:
        st.image(front_img, caption="Front", use_container_width=True)

with col2:
    st.subheader("Profile (side) photo")
    if use_cam:
        cam_profile = st.camera_input("Take profile photo")
        profile_img = _to_pil(cam_profile.getvalue()) if cam_profile else None
    else:
        profile_file = st.file_uploader("Upload profile photo", type=["jpg","jpeg","png"], key="profile")
        profile_img = _to_pil(profile_file) if profile_file else None
    if profile_img:
        st.image(profile_img, caption="Profile", use_container_width=True)

# The rest of your button/Analyze code stays the same:
# - build data URLs from `front_img` and `profile_img`
# - call the OpenAI API

go = st.button("Analyze with GPT")

if go:
#    if not api_key:
#        st.error("Please enter your OpenAI API key in the sidebar.")
#    elif not front_img or not profile_img:
#        st.error("Please upload both front and profile photos.")
    if not front_img or not profile_img:
        st.error("Please upload both front and profile photos.")
    else:
        with st.spinner("Calling model…"):
            try:
                payload = {
                    "front": img_to_data_url(front_img),
                    "profile": img_to_data_url(profile_img),                
                }
                result = call_openai(payload, model, api_key)
                st.success("Done")
                st.json(result)

                # Example: pull categories/levels from model result JSON
                nosas_level = result.get("nosas", {}).get("risk_category")          # "High" / "Low/Intermediate"
                stopbang_level = result.get("stopbang", {}).get("category_high_assumption")  # "Low"/"Intermediate"/"High"
                morpho_level = result.get("morphology", {}).get("severity", "")     # you can have the LLM return "low/moderate/severe"

                # Convert each to score
                scores = [
                    indicator_to_score(nosas_level if nosas_level != "Low/Intermediate" else "low"),
                    indicator_to_score(stopbang_level if stopbang_level == "Intermediate" else stopbang_level),
                    indicator_to_score(morpho_level),
                ]

                ahi_est = round(sum(scores))

                st.divider()

                def fmt_val(val, fallback="—"):
                    return f"**{val}**" if val not in (None, "?", "") else f"*{fallback}*"

                def range_str_from(result_dict: dict, key: str) -> str:
                    """Return 'min–max' from result_dict[key] per SCHEMA_JSON, else ''."""
                    r = result_dict.get(key)
                    if not isinstance(r, dict):
                        return ""
                    mn = r.get("min")
                    mx = r.get("max")
                    if mn is None or mx is None:
                        return ""
                    return f"{mn}–{mx}"

                # --- Display ---

                st.divider()
                st.markdown("### Preliminary AHI (heuristic)")
                st.write(f"Estimated AHI category score: **{ahi_est}** (0–3 scale)")
                st.caption("0 = all low risk; higher values reflect stacked indicators. "
                        "This is a simple heuristic, not a clinical AHI.")

            except Exception as e:
                st.exception(e)
else:
    st.info("Upload both images, paste your API key, and click **Analyze with GPT**.")
