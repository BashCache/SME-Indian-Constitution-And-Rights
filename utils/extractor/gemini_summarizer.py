import google.generativeai as genai
import os
from PIL import Image

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

async def summarize_visual_content(image_path: str, prompt: str = None) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        img = Image.open(image_path)
        full_prompt = prompt or "Describe the content of this image or table in detail. Do not leave any details."
        response = await model.generate_content_async([full_prompt, img])
        # response = ""
        print("*"*80)
        print(f"Gemini response: {response.text}")
        return response.text.strip()
    except Exception as e:
        return f"[Gemini summarization error: {e}]"
