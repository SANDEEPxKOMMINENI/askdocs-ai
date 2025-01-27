import { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Search, Book, AlertCircle, ChevronRight, RefreshCw, Trash2, Loader2, MessageSquare } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { cn } from '../utils/cn';

interface Source {
  id: string;
  relevance: string;
  excerpt: string;
  title?: string;
}

interface Answer {
  answer: string;
  confidence: number;
  source_documents: Source[];
}

interface Document {
  id: string;
  title: string;
  uploadedAt: string;
}

export default function Chat() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [messages, setMessages] = useState<Array<{ type: 'user' | 'ai'; content: string }>>([]);

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await axios.get(`${import.meta.env.VITE_API_URL}/api/pdf/list`);
        setDocuments(response.data);
      } catch (err) {
        console.error('Error fetching documents:', err);
      }
    };
    fetchDocuments();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    
    setLoading(true);
    setError(null);

    setMessages(prev => [...prev, { type: 'user', content: question }]);

    try {
      const response = await axios.post(`${import.meta.env.VITE_API_URL}/api/qa/ask`, {
        question: question.trim(),
        documentIds: selectedDocs.length ? selectedDocs : undefined
      });
      setAnswer(response.data);
      setMessages(prev => [...prev, { type: 'ai', content: response.data.answer }]);
    } catch (err) {
      setError('Failed to get answer. Please try again.');
      console.error('Error asking question:', err);
    } finally {
      setLoading(false);
      setQuestion('');
    }
  };

  const clearChat = () => {
    setMessages([]);
    setAnswer(null);
    setQuestion('');
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="relative mb-8 rounded-2xl overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-violet-500/20 blur-3xl" />
        <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 p-8 rounded-2xl border border-slate-700">
          <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
            AI Document Assistant
          </h1>
          <p className="text-lg text-slate-400">
            Ask questions about your documents and get intelligent, context-aware answers
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="md:col-span-1">
          <Card glass className="sticky top-8">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-blue-400 flex items-center">
                  <Book className="w-5 h-5 mr-2" />
                  Documents
                </h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => window.location.reload()}
                  className="h-8 text-slate-400 hover:text-slate-300"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
              <div className="space-y-2">
                {documents.map((doc) => (
                  <label
                    key={doc.id}
                    className="flex items-center p-3 rounded-lg hover:bg-slate-800/50 transition-colors cursor-pointer group"
                  >
                    <input
                      type="checkbox"
                      checked={selectedDocs.includes(doc.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedDocs([...selectedDocs, doc.id]);
                        } else {
                          setSelectedDocs(selectedDocs.filter(id => id !== doc.id));
                        }
                      }}
                      className="rounded border-slate-700 bg-slate-800 text-blue-500 focus:ring-blue-500"
                    />
                    <span className="ml-3 text-slate-300 font-medium flex-1">{doc.title}</span>
                  </label>
                ))}
              </div>
            </div>
          </Card>
        </div>

        <div className="md:col-span-3">
          <Card glass>
            <div className="p-6">
              <form onSubmit={handleSubmit} className="mb-6">
                <div className="flex gap-4">
                  <div className="relative flex-1">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400 h-5 w-5" />
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Ask any question about your documents..."
                      className="w-full p-4 pl-12 bg-slate-800/50 border border-slate-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg text-slate-200 placeholder-slate-400"
                      disabled={loading}
                    />
                  </div>
                  <Button
                    type="submit"
                    disabled={loading || !question.trim()}
                    className="px-6 bg-gradient-to-r from-blue-500 to-violet-600 hover:from-blue-600 hover:to-violet-700"
                  >
                    {loading && <Loader2 className="h-5 w-5 animate-spin mr-2" />}
                    {loading ? 'Analyzing...' : 'Ask AI'}
                    {!loading && <ChevronRight className="ml-2 h-5 w-5" />}
                  </Button>
                </div>
              </form>

              <div className="flex justify-end mb-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearChat}
                  className="h-8 text-slate-400 hover:text-slate-300"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear Chat
                </Button>
              </div>

              {messages.length === 0 && (
                <div className="text-center py-12">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 text-slate-600" />
                  <p className="text-slate-400">No messages yet. Start by asking a question!</p>
                </div>
              )}

              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={cn(
                    "mb-6 p-4 rounded-xl",
                    msg.type === 'user' 
                      ? 'bg-blue-500/10 border border-blue-500/20 ml-12' 
                      : 'bg-slate-800/50 border border-slate-700 mr-12'
                  )}
                >
                  <div className="prose prose-invert max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              ))}

              {error && (
                <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-6 rounded-xl mb-6 flex items-center">
                  <AlertCircle className="h-6 w-6 mr-3" />
                  <p className="text-lg">{error}</p>
                </div>
              )}

              {answer && (
                <div className="space-y-6 border-t border-slate-700 pt-6">
                  <div className="flex items-center mb-6">
                    <div className="h-3 w-32 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(answer.confidence * 100, 100)}%` }}
                      />
                    </div>
                    <span className="ml-4 text-base text-slate-400">
                      {Math.min(answer.confidence * 100, 100).toFixed(1)}% confidence
                    </span>
                  </div>

                  {answer.source_documents.length > 0 && (
                    <div>
                      <h3 className="text-xl font-semibold text-slate-200 mb-4">Sources</h3>
                      <div className="space-y-4">
                        {answer.source_documents.map((source, index) => (
                          <Card key={source.id} glass className="border border-slate-700">
                            <div className="p-4">
                              <div className="flex items-center justify-between mb-3">
                                <span className="text-base font-semibold text-blue-400">
                                  {source.title || `Source ${index + 1}`}
                                </span>
                                <span className="text-sm px-3 py-1 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded-full">
                                  Relevance: {source.relevance}
                                </span>
                              </div>
                              <p className="text-slate-300 leading-relaxed">{source.excerpt}</p>
                            </div>
                          </Card>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}