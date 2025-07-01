from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

# 벡터 DB 캐싱
vectorstore = None

def load_vectorstore():
    global vectorstore
    if vectorstore is not None:
        return vectorstore

    # 문서 불러오기
    with open("app/vector_store/my_docs.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 텍스트 나누기
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(raw_text)

    docs = [Document(page_content=t) for t in texts]

    # 임베딩 생성
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

async def rag_query(query: str) -> str:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model=settings.GPT_MODEL, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.run(query)
    return result
