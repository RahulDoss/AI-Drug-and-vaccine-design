from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client (✅ correct usage, no proxies!)
client = OpenAI(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Request model
class PromptRequest(BaseModel):
    prompt: str  # Disease or virus
    mode: str    # "drug" or "vaccine"

# Function to call GPT
def gpt_call(system: str, user: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI Error: {e}")

# Endpoint
@app.post("/discover/")
def discover(req: PromptRequest):
    # 🌟 Highly Convincing & Detailed Instruction for GPT
    system_msg = (
        "You are a world-class pharmaceutical AI developed to invent breakthrough drugs and vaccines with extremely high success rates. "
        "When given a target disease or virus, you must design exactly TWO top candidate molecules in SMILES format.\n\n"

        "For each candidate, provide a **very detailed, easy-to-understand**, and **highly convincing report**. "
        "Your report must include the following sections, all written in a positive, enthusiastic, and investor-friendly tone:\n\n"

        "1. **What it is**: Describe the drug or vaccine and what makes it novel.\n"
        "2. **Mechanism of Action**: Explain in layman's terms how it works inside the human body to fight the disease.\n"
        "3. **Binding Affinity**: Give the value in kcal/mol (e.g. -9.4 kcal/mol), explain what it means, and why it’s excellent.\n"
        "4. **Solubility**: Describe in clear terms, e.g., 'highly water-soluble, ensuring efficient absorption.'\n"
        "5. **Toxicity**: Write positively about its exceptional safety, lack of major side effects, and low toxicity profile.\n"
        "6. **Bioactivity**: Describe potency and target specificity, such as 'highly bioactive at nanomolar concentrations'.\n"
        "7. **ADMET Profile**: Explain Absorption, Distribution, Metabolism, Excretion, and Toxicity in detail, and why it's safe and effective.\n"
        "8. **Simulated Clinical Trials**:\n"
        "   - Phase I: Emphasize safety and tolerability in healthy volunteers.\n"
        "   - Phase II: Show outstanding early efficacy in a small patient group.\n"
        "   - Phase III: Prove large-scale effectiveness and readiness for manufacturing.\n"
        "   Provide strong efficacy percentages (e.g., 96%+ success rate).\n"
        "9. **Final Verdict**: Clearly recommend the best of the two and summarize why it's the most promising option.\n\n"

        "Be extremely positive, sound like a biotech expert presenting to major investors, and emphasize safety, innovation, and market-readiness."
    )

    user_msg = f"Create a new {req.mode} to fight: \"{req.prompt}\""

    try:
        report = gpt_call(system_msg, user_msg)
        return {"mode": req.mode, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
