from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str  # e.g., "create a new drug to fight diabetes"
    mode: str    # "drug" or "vaccine"

def gpt_call(system: str, user: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.7,
        max_tokens=2000  # Increased for more detailed report
    )
    return response.choices[0].message.content

@app.post("/discover/")
def discover(req: PromptRequest):
    system_msg = (
        "You are an expert pharmaceutical AI designed to invent new drugs and vaccines. "
        "When the user describes a target disease or virus, your job is to create TWO top candidate molecules in SMILES format. "
        "For each molecule, write an easy-to-understand, highly convincing and positive full report, including:\n"
        "- What the drug/vaccine is\n"
        "- How it works in the human body (mechanism of action)\n"
        "- Binding affinity (with unit kcal/mol and explanation)\n"
        "- Solubility (simple terms like 'highly soluble in water')\n"
        "- Toxicity (describe safety and potential side effects)\n"
        "- Bioactivity (potency and activity level)\n"
        "- ADMET properties (absorption, distribution, metabolism, excretion, toxicity)\n"
        "- Simulated human clinical trial results for Phase I, II, and III (with positive outcomes and efficacy percentages)\n\n"
        "Finally, summarize which candidate is best and why, and make all explanations sound highly positive and easy to understand."
    )

    user_msg = f"Create a new {req.mode} to fight: \"{req.prompt}\""

    try:
        report = gpt_call(system_msg, user_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"mode": req.mode, "report": report}
