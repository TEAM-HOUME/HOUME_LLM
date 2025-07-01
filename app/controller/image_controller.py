from fastapi import APIRouter
from app.model.image_model import PromptRequest
from app.service.image_service import generate_image
from app.service.image_with_rag_service import generate_image_with_rag

router = APIRouter()

@router.post("/generate-image")
async def generate(prompt: PromptRequest):
    return await generate_image(prompt.prompt)

@router.post("/generate-image-smart")
async def generate_with_rag(request: PromptRequest):
    return await generate_image_with_rag(request.prompt)

