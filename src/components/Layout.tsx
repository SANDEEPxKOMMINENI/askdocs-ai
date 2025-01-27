import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FileText, MessageSquare, Home, Command } from 'lucide-react';
import { cn } from '../utils/cn';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))] opacity-[0.02] pointer-events-none" />
      
      <nav className="fixed top-0 left-0 right-0 z-50 glass-effect">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between h-16">
            <Link 
              to="/" 
              className="flex items-center space-x-2"
            >
              <div className="w-10 h-10 rounded-xl button-gradient flex items-center justify-center">
                <Command className="h-6 w-6 text-white" />
              </div>
              <span className="font-bold text-xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-violet-400 to-purple-400">
                AskDocs AI
              </span>
            </Link>
            
            <div className="flex space-x-1">
              <Link
                to="/"
                className={cn(
                  "nav-item",
                  isActive('/') && "active"
                )}
              >
                <Home className="h-5 w-5 mr-2" />
                <span className="font-medium">Home</span>
              </Link>
              
              <Link
                to="/documents"
                className={cn(
                  "nav-item",
                  isActive('/documents') && "active"
                )}
              >
                <FileText className="h-5 w-5 mr-2" />
                <span className="font-medium">Documents</span>
              </Link>
              
              <Link
                to="/chat"
                className={cn(
                  "nav-item",
                  isActive('/chat') && "active"
                )}
              >
                <MessageSquare className="h-5 w-5 mr-2" />
                <span className="font-medium">Chat</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>
      
      <main className="pt-20 min-h-screen relative z-10">
        <div className="max-w-7xl mx-auto px-4 py-6">
          {children}
        </div>
      </main>
    </div>
  );
}