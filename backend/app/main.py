
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router

app = FastAPI(
    title="Enterprise AI Knowledge & Data Agent",
    description="Production-style AI knowledge and data agent API.",
    version="0.1.0",
)

# Configure CORS for the local React/Vite frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Register API routers.
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-ai-agent",
        "version": "0.1.0",
    }
