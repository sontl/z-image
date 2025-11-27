import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import ZImagePipeline
from io import BytesIO
from fastapi.responses import Response
import os

app = FastAPI(title="Z-Image Generation API")

# Global variable for the pipeline
pipe = None

@app.on_event("startup")
async def startup_event():
    global pipe
    print("Initializing Z-Image Pipeline...")
    try:
        # Load the pipeline
        # Use bfloat16 for optimal performance on supported GPUs
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")

        # [Optional] Attention Backend
        # Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported
        try:
            # Attempt to enable Flash Attention 3 as requested
            pipe.transformer.set_attention_backend("_flash_3")
            print("Enabled Flash Attention 3 backend.")
        except Exception as e:
            print(f"Failed to set Flash Attention 3 backend, falling back to default/Flash 2: {e}")
            try:
                pipe.transformer.set_attention_backend("flash")
                print("Enabled Flash Attention 2 backend.")
            except Exception as e2:
                print(f"Failed to set Flash Attention 2 backend: {e2}")

        # [Optional] Model Compilation
        # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
        # pipe.transformer.compile()
        
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        # We don't raise here to allow the app to start, but requests will fail
        pass

class GenerateRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    seed: int = 42

@app.post("/generate", responses={200: {"content": {"image/png": {}}}})
async def generate_image(req: GenerateRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

    try:
        generator = torch.Generator("cuda").manual_seed(req.seed)
        
        # Generate Image
        result = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        
        # Save to buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        return Response(content=buffer.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": pipe is not None}