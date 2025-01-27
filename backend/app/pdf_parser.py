import fitz  # PyMuPDF
from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace while preserving paragraph breaks
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation and structure
    text = re.sub(r'[^\w\s.,!?;:()\-\n]', '', text)
    return text.strip()

def extract_text_from_pdf(file_path: str) -> List[Dict[str, str]]:
    """Extract text from PDF file and return list of sections with metadata"""
    doc = fitz.open(file_path)
    sections = []
    
    # Extract title from first page
    first_page = doc[0].get_text()
    title_match = re.search(r'^(.+?)(?:\n|$)', first_page)
    document_title = title_match.group(1) if title_match else "Untitled Document"
    
    current_section = []
    current_heading = ""
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            # Try to identify section headings
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                # Enhanced heading detection (all caps, numbered sections, etc.)
                if (re.match(r'^(?:[0-9.]+\s+)?[A-Z][^a-z]{0,}$', line) and len(line) < 100) or \
                   (re.match(r'^(?:CHAPTER|SECTION)\s+[0-9]+', line, re.I)) or \
                   (re.match(r'^[0-9]+\.[0-9]+\s+[A-Z]', line)):
                    # Save previous section if exists
                    if current_section:
                        cleaned_content = clean_text(' '.join(current_section))
                        if len(cleaned_content) > 50:  # Only keep substantial sections
                            sections.append({
                                'content': cleaned_content,
                                'page': page_num + 1,
                                'metadata': {
                                    'title': current_heading or f"Page {page_num + 1}",
                                    'type': 'section_content',
                                    'document_title': document_title
                                }
                            })
                        current_section = []
                    current_heading = line
                else:
                    current_section.append(line)
            
    # Add final section
    if current_section:
        cleaned_content = clean_text(' '.join(current_section))
        if len(cleaned_content) > 50:
            sections.append({
                'content': cleaned_content,
                'page': doc.page_count,
                'metadata': {
                    'title': current_heading or f"Page {doc.page_count}",
                    'type': 'section_content',
                    'document_title': document_title
                }
            })
    
    doc.close()
    return sections

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks intelligently"""
    # First split by paragraphs to preserve structure
    paragraphs = text.split('\n\n')
    
    # Then split paragraphs into sentences if needed
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        # Split paragraph into sentences if it's too long
        if len(paragraph) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        else:
            sentences = [paragraph]
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep last sentence for overlap
                    if len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]
                        current_length = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk = []
                        current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks