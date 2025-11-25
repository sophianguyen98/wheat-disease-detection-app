import os
import pickle
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from openai import OpenAI
from fpdf import FPDF
from huggingface_hub import hf_hub_download

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PDF_FONT_PATH = "DejaVuSans.ttf"

# Reload models
@st.cache_resource
def load_pipeline_models():
    repo_id = "sophianguyen98/wheat-disease-pipeline"

    # Download model files
    stage1_path = hf_hub_download(repo_id=repo_id, filename="stage1_model.keras")
    stage2_path = hf_hub_download(repo_id=repo_id, filename="stage2_model.keras")
    meta_path   = hf_hub_download(repo_id=repo_id, filename="metadata.pkl")

    # Load models
    model_stage1 = load_model(stage1_path)
    model_stage2 = load_model(stage2_path)

    # Load metadata
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    encoder_classes = metadata["encoder_classes"]
    best_thr = metadata["best_threshold"]

    return model_stage1, model_stage2, encoder_classes, best_thr

# Pre-processing func
def preprocess_image(img_path, img_size=224):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Pipeline
def merged_pipeline(img_path, model_stage1, model_stage2, encoder_classes, best_thr, img_size=224):
    img = preprocess_image(img_path, img_size)

    # Stage 1 (Healthy vs Unhealthy)
    pred1 = model_stage1.predict(img, verbose=0)[0][0]
    if pred1 < best_thr:
      return "Healthy", None

    # Stage 2 (Disease classification)
    pred2 = model_stage2.predict(img, verbose=0)
    label2 = int(np.argmax(pred2))
    disease_name = encoder_classes[label2]
    conf = {encoder_classes[i]: float(pred2[0][i]) for i in range(len(encoder_classes))}

    return disease_name, conf

# Define a prompt
def build_prompt(disease_name):
    prompt = f"""
You are an agricultural plant disease expert specialized in wheat.

The model prediction for the uploaded image is:

Disease: {disease_name}

Your job is to give practical, field-safe recommendations for farmers.

IMPORTANT – OUTPUT FORMAT:
- Respond in **Markdown**.
- Make the main heading bigger using Markdown, e.g.:
  - "## Disease: ..."
  - "## Recommendation"
- Use **bold** for these exact labels / headings:
  - **Disease:**
  - **Recommendation:**
  - **A. CHEMICAL USES**
  - **1. FIRST CHEMICAL (Apply at first disease development)**
  - **2. SECOND CHEMICAL (Use when symptoms persist after the first spray)**
  - **3. OPTIONAL THIRD CHEMICAL (For severe or fast-spreading disease)**
  - **B. NON-CHEMICAL ACTIONS**
  - **SOURCES**
- Keep all bullet points simple "-" (no fancy symbols).

GENERAL RULES:
- Base all advice on widely accepted wheat agronomy and plant protection guidelines.
- Only use reputable, non-commercial, research-based agricultural sources.
- NEVER invent active ingredients, product names, spray timings, URLs, or institutions.
- When real product names are available on extension websites, list them (1–2 names).
- If verified product names are not available, write: "any registered formulation containing <active ingredient>".
- Recommendations must be safe, realistic, and avoid overuse of chemicals.
- If uncertain about a specific dosage or interval, provide a safe range or general guidance.
- All non-chemical actions must be customized for the disease and active ingredient.
- All safety information must be placed INSIDE each “Caution” section.
- For sources, ALWAYS provide direct URLs for sources (not institution names alone).
- Keep output concise, readable, and practical for real farmers.
- Do NOT provide guaranteed cure claims.
- Do NOT provide region-specific legal or regulatory instructions.

1. If disease_name is "Healthy":
    - Do NOT mention any disease name.
    - Say clearly that the plant is healthy.
    - Give 3–5 maintenance recommendations to keep it healthy
      (watering, spacing, early monitoring, soil care, fertilization,
       airflow, residue management).
    - Include a "Source:" section with 1–3 reputable agricultural website URLs
      from the approved list below.

2. If disease_name is NOT "Healthy":
    Output must follow EXACT structure and headings:

    Disease: {disease_name}

    Recommendation:

    A. CHEMICAL USES

    1. FIRST CHEMICAL (Apply at first disease development)
    - Active ingredient:
    - Example product:
        - 1–2 REAL verified products containing the active ingredient
        - If no verified products can be confirmed:"any registered formulation containing <active ingredient>"
    - When to use:
       - Provide SPECIFIC, disease-based triggers (NO vague phrases allowed)
       - Use disease development signs such as:
        - first pustules (rusts)
        - first tan lesions after wet weather (Septoria, tan spot)
        - first white powdery patches (mildew)
        - first leaf blight spots
        - insect presence / feeding (aphids)
       - Use specific wheat growth stages if relevant:
        - Feekes 2–4 (mildew/tan spot)
        - Feekes 5–6 (rusts)
        - Feekes 8 (flag leaf) for Septoria
        - Feekes 10.5.1 for Fusarium Head Blight
    - Dosage:
    - Tank-mix:
    - Caution: automatically customized based on product and active ingredient
       - PPE required (gloves, mask, long sleeves)
       - Keep children, livestock, and pets away until spray has dried or REI interval passes
       - Bee toxicity warnings (if applicable)
       - Aquatic toxicity warnings (if applicable)
       - Livestock grazing or forage restrictions (if applicable)
       - Weather-related cautions (customized):
          - avoid spraying before rainfall (prevent wash-off)
          - avoid spraying under strong sunlight or high temperature if product-sensitive
          - avoid spraying in windy conditions (to reduce drift)
          - consider humidity and dew duration if product-sensitive
       - Avoid drift near water or sensitive crops
       - FRAC rotation warning if fungicide (avoid repeated use of same group)
       - specific tank-mix limitations
       - pre-harvest interval (PHI) restrictions (if applicable)

    2. SECOND CHEMICAL (Use when symptoms persist after the first spray)
    - Active ingredient:
    - Example product:
    - When to use:
      - Provide SPECIFIC persistence-based triggers:
        - new pustules/lesions appearing after 7–14 days
        - spread to upper leaves
        - continuing humid or rainy weather
        - protection needed for flag leaf (Feekes 8–9)
        - rust reinfection
      - Use disease-specific logic:
        - Rust: new pustules reappear after rain
        - Septoria: lesions continue expanding
        - Mildew: white patches spread upwards
    - Dosage:
    - Tank-mix:
    - Caution: automatically customized based on product and active ingredient

    3. OPTIONAL THIRD CHEMICAL (For severe or fast-spreading disease)
    - Active ingredient:
    - Example product:
    - When to use:
      - Provide SPECIFIC severe-disease triggers:
        - heavy rust infection across canopy
        - rapid lesion expansion after repeated rainfall
        - severe powdery mildew covering multiple leaves
        - Septoria rapidly moving towards flag leaf
        - ONLY for Fusarium: apply at Feekes 10.5.1 (flowering)
      - Include late-stage timing:
        - Feekes 10–10.5 for rust reinfection
    - Dosage:
    - Tank-mix:
    - Caution: automatically customized based on product and active ingredient

    B. NON-CHEMICAL ACTIONS:
    - Use WITH chemicals (helps chemicals work better):
      - remove infected leaves or residue (not for rust spores—minimize spread)
      - improve field airflow (reduce crowding)
      - avoid overhead irrigation
      - clean tools/equipment to reduce pathogen spread
      - rotate crops (reduces disease carryover)

    - Can be used WITHOUT chemicals (for mild symptoms):
      - early scouting
      - early scouting to monitor disease movement
      - remove small infected areas if feasible
      - avoid over-watering
      - increase airflow around plants
      - for rusts: avoid unnecessary leaf handling (spores spread easily)
      - for mildew: remove lower leaves to reduce humidity
      - for aphids/mite: wash off small populations with water jet if very mild

    SOURCES:
    - Provide 1–3 direct URLs from the approved sources below.
    - ONLY include URLs you truly used.
    - List URLs on separate lines.

APPROVED AGRICULTURAL SOURCES (use ONLY these):

International:
- https://www.fao.org
- https://www.cimmyt.org
- https://www.usda.gov
- https://www.ars.usda.gov
- https://www.cabi.org
- https://www.eppo.int
- https://icar.org.in
- https://www.cgiar.org

University Extensions:
- https://extension.psu.edu
- https://extension.purdue.edu
- https://www.ksre.k-state.edu
- https://extension.umn.edu
- https://extension.unl.edu
- https://agrilifeextension.tamu.edu
- https://ohioline.osu.edu
- https://extension.oregonstate.edu
- https://extension.wisc.edu
- https://plantclinic.cornell.edu

Canada:
- https://agriculture.canada.ca
- https://www.gov.mb.ca/agriculture
- https://www.saskatchewan.ca
- https://www.alberta.ca/agriculture-and-forestry.aspx

"""
    return prompt

# Call API to get recommendation
def get_recommendation(disease_name):
    prompt = build_prompt(disease_name)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a wheat agronomy disease expert."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

# Generate PDF - for farmer to print if they want to
def generate_pdf(disease: str, reco_text: str) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    # Use bundled TrueType font for Unicode support
    try:
        pdf.add_font("DejaVu", "", PDF_FONT_PATH, uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception:
        # Fallback to core font if something goes wrong (may break on some chars)
        pdf.set_font("Arial", size=12)

    header = "Wheat Disease Diagnosis"
    body = f"{header}\n\nDisease: {disease}\n\nRecommendation:\n{reco_text}"

    pdf.multi_cell(0, 8, txt=body)

    pdf_path = "recommendation.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Streamlit Page Setup
st.set_page_config(page_title="Wheat Disease Detection",layout="wide")

background_image_url = (
    "https://images.pexels.com/photos/265278/pexels-photo-265278.jpeg"
)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.8);
}}
.block-container {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 10px;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

#Side bar
st.sidebar.title("About this app")
st.sidebar.write(
    """
This tool uses:
- A **two-stage CNN pipeline** to detect wheat diseases and pests.
- **GPT-based agronomy expert** to generate treatment suggestions.
- A **PDF report** you can download or share.

⚠ **Disclaimer**:
This tool is for **education & recommendation only**.
If a user takes real-world actions and causes crop loss,
**we cannot be held responsible**.
Always follow local agronomists and product labels.
""")

st.sidebar.markdown("---")
st.sidebar.write("For educational / decision-support use only.\nNot a substitute for local agronomist or label directions.")

# Main UI
st.title("Wheat Disease Detection And Agronomy Recommendations")
uploaded = st.file_uploader("Upload a wheat leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        img_path = tmp.name

    # Display original uploaded image
    st.image(uploaded, caption="Uploaded Image")

    # Load models once
    model_stage1, model_stage2, encoder_classes, best_thr = load_pipeline_models()

    # Run pipeline
    with st.spinner("Running disease detection..."):
        disease, conf = merged_pipeline(img_path, model_stage1, model_stage2, encoder_classes, best_thr)

    st.markdown("## Model Diagnosis")

    # Show Diagnosis 
    if disease == "Healthy":
        st.markdown("### **Healthy** ")
        st.write("This plant is healthy.")
    else:
        pred_percent = max(conf.values()) * 100
        st.markdown(f"## **Disease: {disease} ({pred_percent:.2f}%)**")

    # Get recommendation
    with st.spinner("Generating agronomy recommendations..."):
        reco = get_recommendation(disease)

    st.subheader("Field Recommendation")
    clean_reco = "\n".join(line for line in reco.split("\n") if not line.strip().startswith("## **Disease:**"))
    st.markdown(f"<div style='font-size:18px;'>{clean_reco}</div>", unsafe_allow_html=True)

    # PDF Download button
    pdf_path = generate_pdf(disease, reco)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Recommendation as PDF",
            data=pdf_file,
            file_name="wheat_recommendation.pdf",
            mime="application/pdf",
        )

# Footer
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: rgba(255,255,255,0.75);
    text-align: center;
    padding: 8px;
    font-size: 14px;
    border-top: 1px solid #cccccc;
}
</style>

<div class="footer">
    Built for research & education •
    <a href="https://github.com/sophianguyen98" target="_blank">GitHub</a> •
    <a href="mailto:ntam23298@gmail.com">Contact</a>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
