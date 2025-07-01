from fastapi import FastAPI
from app.controller.image_controller import router as image_router
from app.controller.rag_controller import router as rag_router

app = FastAPI()

# 라우터 등록
app.include_router(image_router)
app.include_router(rag_router, prefix="/rag")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
