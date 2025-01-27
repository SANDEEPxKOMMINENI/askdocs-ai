from fastapi import APIRouter, UploadFile, File, HTTPException
from app.pdf_parser import extract_text_from_pdf, chunk_text
from app.embeddings import embeddings_model
from app.faiss_search import search_index
import uuid
import os
import tempfile
from typing import List, Dict
from datetime import datetime

router = APIRouter()

# Store document metadata
documents = {}

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Extract text with enhanced metadata
        sections = extract_text_from_pdf(temp_path)
        
        doc_id = str(uuid.uuid4())
        documents[doc_id] = {
            'title': file.filename,
            'uploadedAt': datetime.utcnow().isoformat(),
            'sections': len(sections)
        }
        
        # Process each section with improved chunking
        for section in sections:
            chunks = chunk_text(section['content'], chunk_size=1000, overlap=200)
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                embedding = embeddings_model.get_embedding(chunk)
                
                # Store with enhanced metadata
                metadata = {
                    'document_id': doc_id,
                    'title': file.filename,
                    'page': section.get('page'),
                    'section_type': section.get('metadata', {}).get('type'),
                    'chunk_id': chunk_id
                }
                
                search_index.add_document(chunk_id, chunk, embedding, metadata)
        
        return {"message": "PDF processed successfully", "document_id": doc_id}
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/list")
async def list_documents() -> List[Dict]:
    return [
        {
            "id": doc_id,
            "title": data["title"],
            "uploadedAt": data["uploadedAt"],
            "sections": data["sections"]
        }
        for doc_id, data in documents.items()
    ]