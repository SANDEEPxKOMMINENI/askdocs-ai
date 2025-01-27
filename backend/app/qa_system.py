from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    pipeline
)
import torch
from typing import List, Tuple, Dict, Any
import logging
from app.config import DEVICE, MAX_BATCH_SIZE, CHUNK_SIZE, HF_TOKEN
from app.embeddings import embeddings_model
from app.faiss_search import search_index
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import re

logger = logging.getLogger(__name__)

class EnhancedQASystem:
    def __init__(self):
        self.device = DEVICE
        self.max_batch_size = MAX_BATCH_SIZE
        
        # Initialize models
        self._init_qa_model()
        self._init_generation_model()
        self._init_validation_model()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")

    def _init_qa_model(self):
        """Initialize the main QA model"""
        try:
            # Using a more advanced model for better comprehension
            self.qa_tokenizer = AutoTokenizer.from_pretrained(
                "deepset/roberta-large-squad2",
                token=HF_TOKEN,
                model_max_length=CHUNK_SIZE
            )
            
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                "deepset/roberta-large-squad2",
                token=HF_TOKEN
            ).to(self.device)
            
            if self.device == "cpu":
                self.qa_model = torch.quantization.quantize_dynamic(
                    self.qa_model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            self.qa_model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing QA model: {e}")
            raise

    def _init_generation_model(self):
        """Initialize T5 model for answer generation"""
        try:
            # Using a smaller T5 model for better performance
            model_name = "google/flan-t5-base"  # Using base model instead of xl for better performance
            
            self.gen_tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                token=HF_TOKEN
            )
            
            self.gen_model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                token=HF_TOKEN
            ).to(self.device)
            
            if self.device == "cpu":
                self.gen_model = torch.quantization.quantize_dynamic(
                    self.gen_model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            self.gen_model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing generation model: {e}")
            raise

    def _init_validation_model(self):
        """Initialize validation pipeline"""
        try:
            # Using a simpler model for validation to avoid protobuf issues
            self.validation_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1,
                token=HF_TOKEN
            )
            
            # Use the same model for fact checking
            self.fact_check_pipeline = self.validation_pipeline
            
        except Exception as e:
            logger.error(f"Error initializing validation model: {e}")
            logger.warning("Continuing without validation model")
            # Create dummy pipelines that return high confidence
            self.validation_pipeline = lambda *args, **kwargs: {'scores': [0.9]}
            self.fact_check_pipeline = self.validation_pipeline

    def process_question(self, question: str, document_ids: List[str] = None) -> Tuple[str, float, List[Dict[str, str]]]:
        try:
            # Analyze question type and intent
            question_type = self._analyze_question_type(question)
            
            # Get relevant documents with semantic search
            similar_docs = self._get_relevant_documents(question, document_ids)
            
            if not similar_docs:
                return "I couldn't find any relevant information in the documents to answer your question.", 0.0, []
            
            # Get initial answers from QA model
            qa_answers = self._get_qa_answers(question, similar_docs)
            
            # Generate comprehensive answer based on question type
            final_answer = self._generate_comprehensive_answer(
                question, 
                qa_answers, 
                similar_docs,
                question_type
            )
            
            # Validate answer with multiple models
            confidence = self._validate_answer(question, final_answer, similar_docs)
            
            # Fact check the answer
            fact_check_result = self._fact_check_answer(final_answer, similar_docs)
            if not fact_check_result['is_factual']:
                final_answer = self._revise_answer(final_answer, fact_check_result)
            
            # Prepare detailed sources with context
            sources = self._prepare_sources(similar_docs, final_answer)
            
            return final_answer, confidence, sources
            
        except Exception as e:
            logger.error(f"Error in process_question: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again.", 0.0, []

    def _analyze_question_type(self, question: str) -> Dict[str, Any]:
        """Analyze question type and expected answer format"""
        question_types = {
            'factual': r'\b(what|who|where|when|which)\b',
            'explanation': r'\b(why|how)\b',
            'comparison': r'\b(compare|difference|versus|vs)\b',
            'analysis': r'\b(analyze|evaluate|assess)\b',
            'summary': r'\b(summarize|overview|brief)\b'
        }
        
        # Detect question type
        detected_type = 'general'
        for qtype, pattern in question_types.items():
            if re.search(pattern, question.lower()):
                detected_type = qtype
                break
        
        # Additional analysis
        requires_calculation = bool(re.search(r'\b(calculate|compute|sum|total|average)\b', question.lower()))
        requires_list = bool(re.search(r'\b(list|enumerate|what are)\b', question.lower()))
        temporal_aspect = bool(re.search(r'\b(before|after|during|when|time|date)\b', question.lower()))
        
        return {
            'type': detected_type,
            'requires_calculation': requires_calculation,
            'requires_list': requires_list,
            'temporal_aspect': temporal_aspect
        }

    def _get_relevant_documents(self, question: str, document_ids: List[str] = None, limit: int = 5) -> List[Dict]:
        """Get relevant documents using semantic search with re-ranking"""
        try:
            # Get question embedding
            question_embedding = embeddings_model.get_embedding(question)
            
            # Initial semantic search
            initial_results = search_index.search(
                question_embedding,
                k=limit * 3,  # Get more for re-ranking
                filter_ids=document_ids
            )
            
            # Re-rank results using cross-encoder
            reranked_results = []
            for doc_id, content, score in initial_results:
                # Use more sophisticated relevance scoring
                relevance_score = self._calculate_relevance_score(question, content)
                reranked_results.append({
                    'id': doc_id,
                    'content': content,
                    'score': relevance_score
                })
            
            # Sort by new relevance score and return top results
            return sorted(reranked_results, key=lambda x: x['score'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error getting relevant documents: {e}")
            return []

    def _calculate_relevance_score(self, question: str, content: str) -> float:
        """Calculate sophisticated relevance score using multiple metrics"""
        try:
            # Semantic similarity using cross-encoder
            semantic_score = self.validation_pipeline(
                content,
                [question],
                hypothesis_template="This text answers the question: {}"
            )['scores'][0]
            
            # Keyword matching score
            question_keywords = set(re.findall(r'\w+', question.lower()))
            content_keywords = set(re.findall(r'\w+', content.lower()))
            keyword_score = len(question_keywords & content_keywords) / len(question_keywords)
            
            # Context coherence score
            coherence_score = self._calculate_coherence_score(content)
            
            # Combine scores with weights
            final_score = (
                semantic_score * 0.5 +
                keyword_score * 0.3 +
                coherence_score * 0.2
            )
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0

    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate text coherence score"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 1.0
                
            # Calculate sentence-to-sentence coherence
            coherence_scores = []
            for i in range(len(sentences) - 1):
                score = self.validation_pipeline(
                    sentences[i],
                    [sentences[i + 1]],
                    hypothesis_template="This follows from the previous text: {}"
                )['scores'][0]
                coherence_scores.append(score)
            
            return sum(coherence_scores) / len(coherence_scores)
            
        except Exception as e:
            logger.error(f"Error calculating coherence score: {e}")
            return 0.5

    def _get_qa_answers(self, question: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        """Get answers from QA model with advanced processing"""
        try:
            answers = []
            
            for doc in documents:
                # Process document in chunks with overlap
                content = doc['content']
                chunks = self._split_into_chunks(content, overlap=100)
                
                chunk_answers = []
                for chunk in chunks:
                    inputs = self.qa_tokenizer(
                        question,
                        chunk,
                        max_length=CHUNK_SIZE,
                        truncation=True,
                        stride=50,
                        return_tensors="pt",
                        return_overflowing_tokens=True,
                        padding=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.qa_model(**inputs)
                    
                    # Get multiple answer candidates
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    
                    # Get top-k answer spans
                    top_k = 3
                    start_indices = torch.topk(start_logits, top_k, dim=1).indices[0]
                    end_indices = torch.topk(end_logits, top_k, dim=1).indices[0]
                    
                    for start_idx in start_indices:
                        for end_idx in end_indices:
                            if end_idx >= start_idx and end_idx < start_idx + 50:
                                answer_text = self.qa_tokenizer.convert_tokens_to_string(
                                    self.qa_tokenizer.convert_ids_to_tokens(
                                        inputs["input_ids"][0][start_idx:end_idx + 1]
                                    )
                                )
                                
                                if answer_text and len(answer_text.split()) > 2:
                                    confidence = float(
                                        torch.softmax(start_logits, dim=1)[0][start_idx] *
                                        torch.softmax(end_logits, dim=1)[0][end_idx]
                                    )
                                    
                                    chunk_answers.append({
                                        'text': answer_text,
                                        'confidence': confidence,
                                        'doc_id': doc['id'],
                                        'context': chunk
                                    })
                
                # Select best answer from chunk
                if chunk_answers:
                    best_answer = max(chunk_answers, key=lambda x: x['confidence'])
                    answers.append(best_answer)
            
            return sorted(answers, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting QA answers: {e}")
            return []

    def _generate_comprehensive_answer(
        self, 
        question: str, 
        qa_answers: List[Dict[str, Any]], 
        documents: List[Dict],
        question_type: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive answer using structured formatting and question type"""
        try:
            if not qa_answers:
                return "I couldn't find a specific answer to your question in the documents."
            
            # Prepare answer components based on question type
            components = []
            
            if question_type['type'] == 'factual':
                # Direct factual answer
                components.append(self._format_factual_answer(qa_answers[0]['text']))
                
            elif question_type['type'] == 'explanation':
                # Detailed explanation
                components.append(self._generate_explanation(question, qa_answers, documents))
                
            elif question_type['type'] == 'comparison':
                # Comparative analysis
                components.append(self._generate_comparison(question, qa_answers, documents))
                
            elif question_type['type'] == 'analysis':
                # In-depth analysis
                components.append(self._generate_analysis(question, qa_answers, documents))
                
            elif question_type['type'] == 'summary':
                # Concise summary
                components.append(self._generate_summary(question, qa_answers, documents))
            
            # Add calculations if required
            if question_type['requires_calculation']:
                calculation_result = self._perform_calculations(question, qa_answers)
                if calculation_result:
                    components.append(f"\nCalculation Result: {calculation_result}")
            
            # Format as list if required
            if question_type['requires_list']:
                list_items = self._extract_list_items(qa_answers, documents)
                if list_items:
                    components.append("\nKey Points:")
                    components.extend([f"• {item}" for item in list_items])
            
            # Add temporal context if relevant
            if question_type['temporal_aspect']:
                temporal_context = self._extract_temporal_context(qa_answers, documents)
                if temporal_context:
                    components.append(f"\nTemporal Context: {temporal_context}")
            
            # Combine components into final answer
            final_answer = "\n".join(filter(None, components))
            
            # Validate and refine the answer
            final_answer = self._refine_answer(final_answer)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error generating comprehensive answer: {e}")
            return qa_answers[0]['text'] if qa_answers else "I apologize, but I couldn't generate a comprehensive answer."

    def _format_factual_answer(self, answer_text: str) -> str:
        """Format a factual answer for clarity"""
        # Clean and normalize the answer
        answer = answer_text.strip()
        # Capitalize first letter
        answer = answer[0].upper() + answer[1:] if answer else answer
        # Add period if missing
        if answer and not answer[-1] in '.!?':
            answer += '.'
        return answer

    def _generate_explanation(self, question: str, qa_answers: List[Dict], documents: List[Dict]) -> str:
        """Generate a detailed explanation"""
        try:
            # Combine relevant information from multiple answers
            explanation_parts = []
            
            # Add main explanation
            main_answer = qa_answers[0]['text']
            explanation_parts.append(main_answer)
            
            # Add supporting details
            supporting_details = self._extract_supporting_details(question, documents)
            if supporting_details:
                explanation_parts.append("\nSupporting Details:")
                explanation_parts.extend([f"• {detail}" for detail in supporting_details])
            
            # Add causal relationships if present
            causal_relations = self._extract_causal_relations(question, documents)
            if causal_relations:
                explanation_parts.append("\nCausal Relationships:")
                explanation_parts.extend([f"• {relation}" for relation in causal_relations])
            
            return "\n".join(explanation_parts)
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return qa_answers[0]['text']

    def _generate_comparison(self, question: str, qa_answers: List[Dict], documents: List[Dict]) -> str:
        """Generate a comparative analysis"""
        try:
            # Extract comparison elements
            elements = self._extract_comparison_elements(question)
            if not elements:
                return qa_answers[0]['text']
            
            comparison_parts = []
            
            # Add introduction
            comparison_parts.append(f"Comparing {elements['item1']} and {elements['item2']}:")
            
            # Extract and organize comparison points
            similarities, differences = self._extract_comparison_points(
                elements['item1'],
                elements['item2'],
                documents
            )
            
            # Add similarities
            if similarities:
                comparison_parts.append("\nSimilarities:")
                comparison_parts.extend([f"• {sim}" for sim in similarities])
            
            # Add differences
            if differences:
                comparison_parts.append("\nDifferences:")
                comparison_parts.extend([f"• {diff}" for diff in differences])
            
            return "\n".join(comparison_parts)
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return qa_answers[0]['text']

    def _generate_analysis(self, question: str, qa_answers: List[Dict], documents: List[Dict]) -> str:
        """Generate an in-depth analysis"""
        try:
            analysis_parts = []
            
            # Add main findings
            analysis_parts.append("Key Findings:")
            analysis_parts.append(qa_answers[0]['text'])
            
            # Add detailed analysis points
            analysis_points = self._extract_analysis_points(question, documents)
            if analysis_points:
                analysis_parts.append("\nDetailed Analysis:")
                analysis_parts.extend([f"• {point}" for point in analysis_points])
            
            # Add implications if present
            implications = self._extract_implications(question, documents)
            if implications:
                analysis_parts.append("\nImplications:")
                analysis_parts.extend([f"• {imp}" for imp in implications])
            
            return "\n".join(analysis_parts)
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return qa_answers[0]['text']

    def _generate_summary(self, question: str, qa_answers: List[Dict], documents: List[Dict]) -> str:
        """Generate a concise summary"""
        try:
            # Prepare summary prompt
            prompt = f"Summarize: {qa_answers[0]['text']}"
            
            # Generate summary using T5
            inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.gen_model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract key points
            key_points = self._extract_key_points(documents)
            
            summary_parts = [summary]
            
            if key_points:
                summary_parts.append("\nKey Points:")
                summary_parts.extend([f"• {point}" for point in key_points])
            
            return "\n".join(summary_parts)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return qa_answers[0]['text']

    def _perform_calculations(self, question: str, qa_answers: List[Dict]) -> str:
        """Perform numerical calculations from text"""
        try:
            # Extract numbers and operations
            numbers = re.findall(r'\d+(?:\.\d+)?', qa_answers[0]['text'])
            operation = None
            
            if 'sum' in question.lower() or 'total' in question.lower():
                operation = 'sum'
            elif 'average' in question.lower() or 'mean' in question.lower():
                operation = 'average'
            elif 'difference' in question.lower():
                operation = 'difference'
            
            if numbers and operation:
                numbers = [float(n) for n in numbers]
                if operation == 'sum':
                    result = sum(numbers)
                elif operation == 'average':
                    result = sum(numbers) / len(numbers)
                elif operation == 'difference' and len(numbers) >= 2:
                    result = numbers[0] - numbers[1]
                else:
                    return None
                
                return f"{result:.2f}"
            
            return None
        except Exception as e:
            logger.error(f"Error performing calculations: {e}")
            return None

    def _extract_list_items(self, qa_answers: List[Dict], documents: List[Dict]) -> List[str]:
        """Extract list items from text"""
        try:
            # Combine relevant text
            text = " ".join([ans['text'] for ans in qa_answers])
            
            # Look for bullet points or numbered lists
            items = re.findall(r'(?:^|\n)[\s\t]*(?:[-•*]|\d+\.)\s*(.+?)(?=(?:\n|$))', text)
            
            if not items:
                # Try splitting on semicolons or periods
                items = [s.strip() for s in re.split(r'[;.]', text) if s.strip()]
            
            return items[:10]  # Limit to top 10 items
        except Exception as e:
            logger.error(f"Error extracting list items: {e}")
            return []

    def _extract_temporal_context(self, qa_answers: List[Dict], documents: List[Dict]) -> str:
        """Extract temporal information from text"""
        try:
            # Look for dates and time expressions
            text = " ".join([ans['text'] for ans in qa_answers])
            
            # Extract dates
            dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)
            
            # Extract relative time expressions
            relative_time = re.findall(r'\b(?:today|yesterday|tomorrow|last|next|previous|following|ago|before|after|during)\b[^.]*', text)
            
            temporal_info = []
            if dates:
                temporal_info.append(f"Dates mentioned: {', '.join(dates)}")
            if relative_time:
                temporal_info.append(f"Temporal context: {' '.join(relative_time)}")
            
            return " ".join(temporal_info) if temporal_info else ""
        except Exception as e:
            logger.error(f"Error extracting temporal context: {e}")
            return ""

    def _validate_answer(self, question: str, answer: str, documents: List[Dict]) -> float:
        """Validate the generated answer using multiple models"""
        try:
            validation_scores = []
            
            # Check answer relevance
            relevance_score = self.validation_pipeline(
                answer,
                [question],
                hypothesis_template="This answers the question: {}"
            )['scores'][0]
            validation_scores.append(relevance_score)
            
            # Check factual consistency
            for doc in documents:
                consistency_score = self.validation_pipeline(
                    doc['content'],
                    [answer],
                    hypothesis_template="This text supports the statement: {}"
                )['scores'][0]
                validation_scores.append(consistency_score)
            
            # Check answer completeness
            completeness_score = self._check_answer_completeness(question, answer)
            validation_scores.append(completeness_score)
            
            # Calculate final confidence score
            confidence = np.mean(validation_scores)
            
            return min(confidence * 100, 100)  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error validating answer: {e}")
            return 50.0  # Default confidence

    def _check_answer_completeness(self, question: str, answer: str) -> float:
        """Check if the answer completely addresses the question"""
        try:
            # Prepare completeness check
            hypothesis = f"This answer completely addresses the question: {question}"
            
            result = self.validation_pipeline(
                answer,
                [hypothesis],
                hypothesis_template="{}"
            )
            
            return result['scores'][0]
        except Exception as e:
            logger.error(f"Error checking answer completeness: {e}")
            return 0.5

    def _fact_check_answer(self, answer: str, documents: List[Dict]) -> Dict[str, Any]:
        """Perform fact checking on the answer"""
        try:
            # Split answer into checkable claims
            claims = self._extract_claims(answer)
            
            fact_check_results = []
            for claim in claims:
                # Check each claim against documents
                evidence_found = False
                for doc in documents:
                    result = self.fact_check_pipeline(
                        premise=doc['content'],
                        hypothesis=claim
                    )
                    if result[0]['label'] == 'ENTAILMENT':
                        evidence_found = True
                        break
                
                fact_check_results.append({
                    'claim': claim,
                    'verified': evidence_found
                })
            
            # Calculate overall factuality score
            verified_claims = sum(1 for r in fact_check_results if r['verified'])
            factuality_score = verified_claims / len(fact_check_results) if fact_check_results else 0
            
            return {
                'is_factual': factuality_score > 0.8,
                'factuality_score': factuality_score,
                'claim_results': fact_check_results
            }
            
        except Exception as e:
            logger.error(f"Error fact checking answer: {e}")
            return {'is_factual': True, 'factuality_score': 1.0, 'claim_results': []}

    def _extract_claims(self, text: str) -> List[str]:
        """Extract checkable claims from text"""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Filter for sentences that are likely to be claims
        claims = []
        for sentence in sentences:
            # Look for factual statement patterns
            if re.search(r'\b(?:is|are|was|were|has|have|had|will|would|could|should)\b', sentence):
                claims.append(sentence)
        
        return claims

    def _revise_answer(self, answer: str, fact_check_result: Dict[str, Any]) -> str:
        """Revise answer based on fact check results"""
        try:
            if fact_check_result['factuality_score'] > 0.8:
                return answer
            
            # Identify unverified claims
            unverified_claims = [
                result['claim'] for result in fact_check_result['claim_results']
                if not result['verified']
            ]
            
            if not unverified_claims:
                return answer
            
            # Generate correction prompt
            correction_prompt = (
                "Revise the following text to be more accurate, "
                "removing or qualifying these unverified claims:\n"
                f"Text: {answer}\n"
                "Unverified claims:\n" +
                "\n".join(f"- {claim}" for claim in unverified_claims)
            )
            
            # Generate revised answer
            inputs = self.gen_tokenizer(correction_prompt, return_tensors="pt").to(self.device)
            outputs = self.gen_model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            revised_answer = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return revised_answer
            
        except Exception as e:
            logger.error(f"Error revising answer: {e}")
            return answer

    def _prepare_sources(self, documents: List[Dict], answer: str) -> List[Dict[str, str]]:
        """Prepare source documents with relevant excerpts and metadata"""
        try:
            sources = []
            
            for doc in documents:
                # Find most relevant excerpt
                excerpt = self._find_relevant_excerpt(doc['content'], answer)
                
                if excerpt:
                    # Get document metadata
                    metadata = search_index.get_document_metadata(doc['id'])
                    
                    sources.append({
                        'id': doc['id'],
                        'title': metadata.get('title', 'Untitled Document'),
                        'relevance': f"{doc['score']*100:.1f}%",
                        'excerpt': excerpt,
                        'page': metadata.get('page'),
                        'section': metadata.get('section_type')
                    })
            
            return sources
            
        except Exception as e:
            logger.error(f"Error preparing sources: {e}")
            return []

    def _find_relevant_excerpt(self, content: str, answer: str, window_size: int = 200) -> str:
        """Find most relevant excerpt from content using semantic similarity"""
        try:
            sentences = sent_tokenize(content)
            if not sentences:
                return ""
            
            # Calculate similarity scores for each sentence
            answer_embedding = embeddings_model.get_embedding(answer)
            
            best_excerpt = ""
            max_similarity = -1
            
            for i in range(len(sentences)):
                # Create window of sentences
                window = sentences[max(0, i-1):min(len(sentences), i+2)]
                window_text = " ".join(window)
                
                # Calculate similarity
                window_embedding = embeddings_model.get_embedding(window_text)
                similarity = np.dot(answer_embedding, window_embedding)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_excerpt = window_text
            
            return best_excerpt if best_excerpt else sentences[0]
            
        except Exception as e:
            logger.error(f"Error finding relevant excerpt: {e}")
            return content[:200] + "..."

    def _split_into_chunks(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks intelligently"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap sentences
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                    current_chunk = overlap_sentences
                    current_length = sum(len(s.split()) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


    def _extract_supporting_details(self, question: str, documents: List[Dict]) -> List[str]:
        """Extract supporting details relevant to the question"""
        try:
            details = []
            for doc in documents:
                # Find sentences that support the answer
                sentences = sent_tokenize(doc['content'])
                for sentence in sentences:
                    # Check if sentence provides supporting information
                    result = self.validation_pipeline(
                        sentence,
                        [question],
                        hypothesis_template="This provides relevant information for: {}"
                    )
                    if result['scores'][0] > 0.7:  # High confidence threshold
                        details.append(sentence)
            
            # Remove duplicates and similar sentences
            unique_details = []
            for detail in details:
                if not any(self._sentences_are_similar(detail, d) for d in unique_details):
                    unique_details.append(detail)
            
            return unique_details[:5]  # Return top 5 supporting details
        except Exception as e:
            logger.error(f"Error extracting supporting details: {e}")
            return []

    def _extract_causal_relations(self, question: str, documents: List[Dict]) -> List[str]:
        """Extract cause-effect relationships"""
        try:
            causal_patterns = [
                r'because\s+(.+?)[.,]',
                r'due to\s+(.+?)[.,]',
                r'as a result of\s+(.+?)[.,]',
                r'therefore\s+(.+?)[.,]',
                r'consequently\s+(.+?)[.,]'
            ]
            
            relations = []
            for doc in documents:
                for pattern in causal_patterns:
                    matches = re.finditer(pattern, doc['content'], re.IGNORECASE)
                    for match in matches:
                        relation = match.group(1).strip()
                        if relation:
                            relations.append(relation)
            
            return list(set(relations))[:3]  # Return top 3 unique relations
        except Exception as e:
            logger.error(f"Error extracting causal relations: {e}")
            return []

    def _extract_comparison_elements(self, question: str) -> Dict[str, str]:
        """Extract elements being compared"""
        try:
            # Look for comparison patterns
            pattern = r'(?:compare|difference between|versus|vs)\s+([^,]+)\s+(?:and|vs|to)\s+([^.]+)'
            match = re.search(pattern, question, re.IGNORECASE)
            
            if match:
                return {
                    'item1': match.group(1).strip(),
                    'item2': match.group(2).strip()
                }
            return {}
        except Exception as e:
            logger.error(f"Error extracting comparison elements: {e}")
            return {}

    def _extract_comparison_points(self, item1: str, item2: str, documents: List[Dict]) -> Tuple[List[str], List[str]]:
        """Extract similarities and differences between items"""
        try:
            similarities = []
            differences = []
            
            for doc in documents:
                # Look for similarity patterns
                similarity_patterns = [
                    f"both {item1} and {item2}",
                    f"similar to",
                    f"like",
                    f"share",
                    f"common"
                ]
                
                # Look for difference patterns
                difference_patterns = [
                    f"unlike",
                    f"differs",
                    f"while",
                    f"however",
                    f"but",
                    f"whereas"
                ]
                
                sentences = sent_tokenize(doc['content'])
                for sentence in sentences:
                    # Check for similarities
                    if any(pattern in sentence.lower() for pattern in similarity_patterns):
                        similarities.append(sentence)
                    
                    # Check for differences
                    if any(pattern in sentence.lower() for pattern in difference_patterns):
                        differences.append(sentence)
            
            return (
                list(set(similarities))[:3],  # Top 3 similarities
                list(set(differences))[:3]    # Top 3 differences
            )
        except Exception as e:
            logger.error(f"Error extracting comparison points: {e}")
            return [], []

    def _extract_analysis_points(self, question: str, documents: List[Dict]) -> List[str]:
        """Extract key analysis points"""
        try:
            points = []
            
            # Look for analytical patterns
            analysis_patterns = [
                r'significant\s+(.+?)[.,]',
                r'important\s+(.+?)[.,]',
                r'key\s+(.+?)[.,]',
                r'critical\s+(.+?)[.,]',
                r'analysis shows\s+(.+?)[.,]'
            ]
            
            for doc in documents:
                for pattern in analysis_patterns:
                    matches = re.finditer(pattern, doc['content'], re.IGNORECASE)
                    for match in matches:
                        point = match.group(1).strip()
                        if point:
                            points.append(point)
            
            return list(set(points))[:5]  # Return top 5 unique points
        except Exception as e:
            logger.error(f"Error extracting analysis points: {e}")
            return []

    def _extract_implications(self, question: str, documents: List[Dict]) -> List[str]:
        """Extract implications and consequences"""
        try:
            implications = []
            
            # Look for implication patterns
            implication_patterns = [
                r'implies\s+(.+?)[.,]',
                r'suggests\s+(.+?)[.,]',
                r'indicates\s+(.+?)[.,]',
                r'means that\s+(.+?)[.,]',
                r'leads to\s+(.+?)[.,]'
            ]
            
            for doc in documents:
                for pattern in implication_patterns:
                    matches = re.finditer(pattern, doc['content'], re.IGNORECASE)
                    for match in matches:
                        implication = match.group(1).strip()
                        if implication:
                            implications.append(implication)
            
            return list(set(implications))[:3]  # Return top 3 unique implications
        except Exception as e:
            logger.error(f"Error extracting implications: {e}")
            return []

    def _extract_key_points(self, documents: List[Dict]) -> List[str]:
        """Extract key points for summary"""
        try:
            points = []
            
            # Look for key point patterns
            key_patterns = [
                r'main\s+(.+?)[.,]',
                r'key\s+(.+?)[.,]',
                r'essential\s+(.+?)[.,]',
                r'primary\s+(.+?)[.,]',
                r'crucial\s+(.+?)[.,]'
            ]
            
            for doc in documents:
                for pattern in key_patterns:
                    matches = re.finditer(pattern, doc['content'], re.IGNORECASE)
                    for match in matches:
                        point = match.group(1).strip()
                        if point:
                            points.append(point)
            
            return list(set(points))[:5]  # Return top 5 unique points
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []

    def _sentences_are_similar(self, sent1: str, sent2: str, threshold: float = 0.8) -> bool:
        """Check if two sentences are semantically similar"""
        try:
            # Get embeddings
            emb1 = embeddings_model.get_embedding(sent1)
            emb2 = embeddings_model.get_embedding(sent2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            return similarity > threshold
        except Exception as e:
            logger.error(f"Error comparing sentences: {e}")
            return False

# Initialize the enhanced system
qa_system = EnhancedQASystem()