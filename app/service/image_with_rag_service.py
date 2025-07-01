from app.config.settings import settings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import base64
from fastapi.responses import StreamingResponse
from io import BytesIO
import httpx
from fastapi import HTTPException

def get_vectorstore():
    with open("app/vector_store/my_docs.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=t) for t in splitter.split_text(raw_text)]

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeddings)

async def generate_image_with_rag(user_prompt: str):
    # 1. 문서 기반으로 프롬프트 강화
    vs = get_vectorstore()
    retriever = vs.as_retriever()
    llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model="gpt-4")

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    enriched_prompt = rag_chain.run(f"다음 프롬프트를 구체화해줘: {user_prompt}")

    # 2. gpt-image-1로 이미지 생성 요청
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-image-1",
        "prompt": enriched_prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "high"
    }

    timeout = httpx.Timeout(120.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        image_data = result["data"][0]["b64_json"]
        image_bytes = base64.b64decode(image_data)

        return StreamingResponse(BytesIO(image_bytes), media_type="image/png")
