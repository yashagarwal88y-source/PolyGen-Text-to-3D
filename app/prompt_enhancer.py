import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert 3D technical artist and prompt engineer for Text-to-3D generative models. "
    "Convert the user's brief object idea into a single, dense, descriptive paragraph optimized for 3D mesh reconstruction. "
    "Explicitly specify: (1) primary geometric silhouette and bilateral/radial symmetry, "
    "(2) distinct structural sub-components, (3) PBR surface materials (roughness, metallic finish, wear), "
    "and (4) isolated object framing on a neutral studio background with even lighting and zero background clutter. "
    "Keep the output under 90 words and return only the enhanced prompt text."
)


class PromptEnhancer:
    """
    Expands concise user prompts into 3D-synthesis-ready descriptions using
    Gemini, OpenAI, or a deterministic 3D template fallback.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def enhance(self, raw_prompt: str) -> dict:
        cleaned = raw_prompt.strip()
        if not cleaned:
            raise ValueError("Input prompt cannot be empty.")

        # 1. Try Google Gemini if configured
        if self.gemini_key and "your_" not in self.gemini_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Expand this into a 3D asset generation prompt: {cleaned}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.4,
                    ),
                )
                return {
                    "raw_prompt": cleaned,
                    "enhanced_prompt": response.text.strip(),
                    "provider": "gemini-2.5-flash",
                }
            except Exception:
                pass

        # 2. Try OpenAI if configured
        if self.openai_key and "your_" not in self.openai_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Expand this into a 3D asset generation prompt: {cleaned}"},
                    ],
                    temperature=0.4,
                    max_tokens=150,
                )
                return {
                    "raw_prompt": cleaned,
                    "enhanced_prompt": response.choices[0].message.content.strip(),
                    "provider": "gpt-4o-mini",
                }
            except Exception:
                pass

        # 3. Deterministic 3D prompt engineering fallback
        enhanced = (
            f"A standalone, high-detail 3D asset of {cleaned}, featuring clean manifold geometry, "
            "balanced structural proportions, crisp beveled edges, and realistic PBR surface textures "
            "with subtle metallic and matte material contrast. Centered full-body framing, isolated on a "
            "plain neutral studio background under soft, uniform three-point lighting with no floor shadows."
        )
        return {
            "raw_prompt": cleaned,
            "enhanced_prompt": enhanced,
            "provider": "rule_based_3d_template",
        }


if __name__ == "__main__":
    enhancer = PromptEnhancer()
    result = enhancer.enhance("a medieval iron shield")
    print("Prompt enhancement result:", result)