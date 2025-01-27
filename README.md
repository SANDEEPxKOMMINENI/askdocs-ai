# AskDocs AI 🤖

An intelligent document assistant powered by advanced AI technology that helps users extract insights and answers from their documents.

## 🌟 Features

- **Document Upload**: Securely upload and process PDF documents
- **Intelligent Search**: Advanced semantic search capabilities
- **Smart Q&A**: Get accurate answers with context from your documents
- **Advanced Analysis**: Multiple question types supported (factual, explanatory, comparative, etc.)
- **Source Citations**: Transparent answer sources with relevance scores
- **Real-time Processing**: Fast and efficient document processing

## 🚀 Tech Stack

### Frontend
- React 18
- TypeScript
- Vite
- TailwindCSS
- Lucide Icons
- React Router DOM
- Axios

### Backend
- FastAPI
- PyTorch
- Hugging Face Transformers
- FAISS for vector search
- SQLite with SQLAlchemy
- NLTK for text processing

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- Hugging Face API token

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Hugging Face token
```

4. Run the backend server:
```bash
uvicorn app.main:app --reload
```

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API URL
```

3. Run the development server:
```bash
npm run dev
```

## 🎯 Usage

1. **Upload Documents**
   - Navigate to the Documents page
   - Drag and drop or click to upload PDF files
   - Wait for processing completion

2. **Ask Questions**
   - Go to the Chat page
   - Type your question in the input field
   - View answers with source citations and confidence scores

3. **View Documents**
   - Access uploaded documents in the Documents section
   - Filter and manage your document library

## 🧠 AI Features

### Question Types Supported
- Factual questions (What, Who, Where, When)
- Explanatory questions (Why, How)
- Comparative analysis
- Analytical queries
- Summary requests

### Answer Processing
- Multi-stage answer generation
- Fact checking and validation
- Source verification
- Confidence scoring
- Context-aware responses

## 🔒 Security

- Secure document processing
- No permanent storage of sensitive data
- Local database for document management
- Environment variable protection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework
- [TailwindCSS](https://tailwindcss.com/) for styling
