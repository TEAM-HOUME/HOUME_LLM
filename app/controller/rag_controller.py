from fastapi import APIRouter
from app.model.rag_model import QueryRequest
from app.service.rag_service import rag_query

router = APIRouter()

@router.post("/rag")
async def run_rag(request: QueryRequest):
    answer = await rag_query(request.query)
    return {"answer": answer}
