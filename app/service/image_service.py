import httpx
import base64
import logging
from io import BytesIO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from app.config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def generate_image(prompt: str):
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "high"
    }

    timeout = httpx.Timeout(120.0, connect=30.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Sending prompt to OpenAI: {prompt[:40]}...")
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)

            result = response.json()
            image_data = result.get("data", [])[0]

            if "b64_json" in image_data:
                image_bytes = base64.b64decode(image_data["b64_json"])
                return StreamingResponse(
                    BytesIO(image_bytes),
                    media_type="image/png",
                    headers={"Content-Disposition": "inline; filename=generated_image.png"}
                )
            elif "url" in image_data:
                return {"image_url": image_data["url"]}

            else:
                raise HTTPException(status_code=500, detail="Unexpected image data format")

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="OpenAI API timeout")
    except httpx.RequestError as e:
        logger.error(f"HTTPX error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API request failed")
