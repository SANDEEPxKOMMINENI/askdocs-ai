from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import pdf_router, qa_router

app = FastAPI(title="AskDocs AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf_router.router, prefix="/api/pdf", tags=["PDF"])
app.include_router(qa_router.router, prefix="/api/qa", tags=["QA"])

@app.get("/")
async def root():
    return {"message": "Welcome to AskDocs AI API"}