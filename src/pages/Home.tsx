import { Link } from 'react-router-dom';
import { FileText, MessageSquare, ArrowRight, Sparkles, Zap, Shield } from 'lucide-react';
import { Card } from '../components/Card';

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-16 relative">
        <div className="absolute inset-0 -z-10 bg-gradient-to-r from-blue-500/20 to-violet-500/20 rounded-3xl blur-3xl" />
        <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-violet-400 to-purple-400 bg-clip-text text-transparent animate-gradient">
          Welcome to AskDocs AI
        </h1>
        <p className="text-2xl text-slate-400 max-w-3xl mx-auto">
          Your intelligent document assistant powered by advanced AI technology
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
        <Link to="/documents">
          <Card 
            hover 
            className="group h-full bg-gradient-to-br from-blue-500 to-violet-600 p-8"
          >
            <div className="relative z-10">
              <div className="mb-6 p-3 bg-white/20 rounded-lg w-fit">
                <FileText className="h-8 w-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-3">Upload Documents</h2>
              <p className="text-blue-100 text-lg mb-6 min-h-[4rem]">
                Securely upload your PDF documents for instant AI-powered analysis and insights
              </p>
              <div className="flex items-center text-white font-semibold">
                Get Started 
                <ArrowRight className="ml-2 h-5 w-5 transition-transform duration-300 group-hover:translate-x-2" />
              </div>
            </div>
            <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl" />
          </Card>
        </Link>
        
        <Link to="/chat">
          <Card 
            hover 
            className="group h-full bg-gradient-to-br from-violet-500 to-purple-600 p-8"
          >
            <div className="relative z-10">
              <div className="mb-6 p-3 bg-white/20 rounded-lg w-fit">
                <MessageSquare className="h-8 w-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-3">Ask Questions</h2>
              <p className="text-violet-100 text-lg mb-6 min-h-[4rem]">
                Get instant, accurate answers from your document library using advanced AI
              </p>
              <div className="flex items-center text-white font-semibold">
                Start Chatting 
                <ArrowRight className="ml-2 h-5 w-5 transition-transform duration-300 group-hover:translate-x-2" />
              </div>
            </div>
            <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl" />
          </Card>
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        <Card glass hover className="p-6 dark-card">
          <div className="flex items-center mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Sparkles className="h-6 w-6 text-blue-400" />
            </div>
            <h3 className="text-xl font-semibold ml-3 text-slate-200">Smart Analysis</h3>
          </div>
          <p className="text-slate-400">
            Advanced AI algorithms analyze your documents to provide accurate, context-aware responses
          </p>
        </Card>

        <Card glass hover className="p-6 dark-card">
          <div className="flex items-center mb-4">
            <div className="p-2 bg-violet-500/20 rounded-lg">
              <Zap className="h-6 w-6 text-violet-400" />
            </div>
            <h3 className="text-xl font-semibold ml-3 text-slate-200">Instant Answers</h3>
          </div>
          <p className="text-slate-400">
            Get immediate responses to your questions with relevant context and source citations
          </p>
        </Card>

        <Card glass hover className="p-6 dark-card">
          <div className="flex items-center mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Shield className="h-6 w-6 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold ml-3 text-slate-200">Secure & Private</h3>
          </div>
          <p className="text-slate-400">
            Your documents are processed securely with state-of-the-art privacy protection
          </p>
        </Card>
      </div>
    </div>
  );
}