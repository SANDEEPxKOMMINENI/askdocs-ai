from app.qa_system import qa_system
from app.pdf_parser import extract_text_from_pdf, chunk_text
from app.embeddings import embeddings_model
from app.faiss_search import search_index

def test_qa_system(pdf_path: str, question: str):
    print("1. Extracting text from PDF...")
    sections = extract_text_from_pdf(pdf_path)
    
    print("2. Processing sections...")
    for section in sections:
        chunks = chunk_text(section['content'])
        for chunk in chunks:
            # Get embedding for the chunk
            embedding = embeddings_model.get_embedding(chunk)
            # Add to search index
            search_index.add_document(
                str(hash(chunk)),  # Simple document ID
                chunk,
                embedding,
                section['metadata']
            )
    
    print("3. Processing question...")
    answer, confidence, sources = qa_system.process_question(question)
    
    print("\nResults:")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("\nSources:")
    for source in sources:
        print(f"\nRelevance: {source['relevance']}")
        print(f"Excerpt: {source['excerpt']}")

if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = "path/to/your/document.pdf"
    question = "What is this document about?"
    test_qa_system(pdf_path, question)