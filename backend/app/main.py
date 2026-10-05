from fastapi import FastAPI

from app.api.auth import router as auth_router


app = FastAPI(
    title="Enterprise AI Knowledge & Data Agent",
    description="Production-style AI knowledge and data agent API.",
    version="0.1.0",
)


app.include_router(auth_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-ai-agent",
        "version": "0.1.0",
    }