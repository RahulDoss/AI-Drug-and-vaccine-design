from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    mode: str  # "drug" or "vaccine"

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

@app.post("/discover/")
def discover(req: PromptRequest):
    system_msg = (
        "You are a world-class pharmaceutical design AI built to generate cutting-edge drug and vaccine candidates. "
        "When given a disease or viral target, you generate exactly TWO top candidate molecules in SMILES format.\n\n"

        "For **each candidate**, create a full and highly convincing scientific report, written in clear, persuasive language understandable by a smart layperson or medical investor. "
        "The tone should be **extremely positive**, confident, and scientifically compelling.\n\n"

        "### Each report must include:\n"
        "- What the drug/vaccine is and how it targets the disease (mechanism of action)\n"
        "- Binding affinity (in kcal/mol, explained clearly in layman's terms)\n"
        "- Solubility (described in practical terms: e.g., 'highly water-soluble')\n"
        "- Toxicity (written positively and convincingly about safety and minimal side effects)\n"
        "- Bioactivity (describe potency and effectiveness in targeting disease)\n"
        "- ADMET profile (Absorption, Distribution, Metabolism, Excretion, and Toxicity — all in a detailed, positive, easy-to-understand format)\n"
        "- Simulated Phase I, II, III human clinical trial results with excellent efficacy outcomes and percentages\n\n"

        "Finally, summarize which candidate is the most promising and why, with a final recommendation. Use confident, persuasive tone and highlight human impact.\n\n"
        "Use realistic but extremely optimistic and market-attractive language (like you'd present to pharma investors)."
    )

    user_msg = f"Create a new {req.mode} to fight: \"{req.prompt}\""

    try:
        report = gpt_call(system_msg, user_msg)
        return {"mode": req.mode, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
