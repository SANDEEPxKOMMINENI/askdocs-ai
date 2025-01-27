from fastapi import APIRouter, HTTPException
from app.models import Question, Answer
from app.qa_system import qa_system
import logging
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        logger.info(f"Processing question: {question.question}")
        answer, confidence, sources = qa_system.process_question(question.question)
        
        # Format sources for better readability
        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                'id': source['id'],
                'relevance': source['relevance'],
                'excerpt': source['excerpt']
            })
        
        return Answer(
            answer=answer,
            confidence=confidence,
            source_documents=formatted_sources
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )