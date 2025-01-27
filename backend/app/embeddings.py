from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from app.config import DEVICE, CHUNK_SIZE, HF_TOKEN

class OptimizedEmbeddings:
    def __init__(self):
        self.device = DEVICE
        # Use a smaller model for better performance
        self.model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN)
        self.model = AutoModel.from_pretrained(self.model_name, token=HF_TOKEN)
        
        self.model = self.model.to(self.device)
        if self.device == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str) -> list[float]:
        try:
            # Tokenize and handle long texts
            if len(text.split()) > CHUNK_SIZE:
                chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
                embeddings = []
                
                for chunk in chunks:
                    encoded_input = self.tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        max_length=CHUNK_SIZE,
                        return_tensors='pt'
                    ).to(self.device)

                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                        embedding = F.normalize(embedding, p=2, dim=1)
                        embeddings.append(embedding)

                # Average the embeddings from chunks
                final_embedding = torch.mean(torch.stack(embeddings), dim=0)
                return final_embedding.cpu().numpy().tolist()[0]
            else:
                encoded_input = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=CHUNK_SIZE,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    embedding = F.normalize(embedding, p=2, dim=1)
                    return embedding.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default embedding size for MiniLM

embeddings_model = OptimizedEmbeddings()