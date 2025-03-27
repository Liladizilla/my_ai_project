from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO

app = FastAPI()

# Load the AI model (this may take time on first run)
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

# Request model for AI image generation
class ImageRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "AI Image Generator is running!"}

@app.post("/generate-image/")
def generate_image(request: ImageRequest):
    prompt = request.prompt
    image = model(prompt).images[0]

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image_base64": img_str}
