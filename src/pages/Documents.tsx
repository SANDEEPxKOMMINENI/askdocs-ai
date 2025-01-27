import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { Card } from '../components/Card';

export default function Documents() {
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const formData = new FormData();
    formData.append('file', acceptedFiles[0]);

    setUploadStatus('uploading');
    setErrorMessage('');

    try {
      await axios.post(`${import.meta.env.VITE_API_URL}/api/pdf/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadStatus('success');
      setTimeout(() => setUploadStatus('idle'), 3000);
    } catch (error) {
      console.error('Error uploading document:', error);
      setUploadStatus('error');
      setErrorMessage('Failed to upload document. Please try again.');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: false,
  });

  return (
    <div className="max-w-4xl mx-auto">
      <div className="relative mb-12 rounded-2xl overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-violet-500/20 blur-3xl" />
        <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 p-8 rounded-2xl border border-slate-700">
          <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
            Document Upload
          </h1>
          <p className="text-lg text-slate-400">
            Upload your PDF documents to get started with AI-powered analysis
          </p>
        </div>
      </div>
      
      <Card
        glass
        className={`
          relative overflow-hidden transition-all duration-300
          ${isDragActive ? 'border-blue-500 bg-slate-900/80' : 'border-slate-700 hover:border-blue-500/50'}
        `}
      >
        <div
          {...getRootProps()}
          className="p-16 text-center cursor-pointer"
        >
          <input {...getInputProps()} />
          
          {uploadStatus === 'uploading' && (
            <div className="animate-pulse">
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-blue-500/20 flex items-center justify-center">
                <Loader2 className="h-8 w-8 text-blue-400 animate-spin" />
              </div>
              <p className="text-xl font-semibold text-blue-400">
                Uploading document...
              </p>
            </div>
          )}

          {uploadStatus === 'success' && (
            <div>
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle className="h-8 w-8 text-emerald-400" />
              </div>
              <p className="text-xl font-semibold text-emerald-400">
                Document uploaded successfully!
              </p>
            </div>
          )}

          {uploadStatus === 'error' && (
            <div>
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
                <XCircle className="h-8 w-8 text-red-400" />
              </div>
              <p className="text-xl font-semibold text-red-400">
                {errorMessage}
              </p>
            </div>
          )}

          {uploadStatus === 'idle' && (
            <>
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-slate-800 flex items-center justify-center">
                <FileText className="h-8 w-8 text-slate-400" />
              </div>
              <p className="text-xl font-semibold text-slate-300 mb-2">
                {isDragActive
                  ? 'Drop your PDF here...'
                  : 'Drag and drop your PDF here, or click to browse'}
              </p>
              <p className="text-slate-400">
                Supported format: PDF
              </p>
            </>
          )}
        </div>
      </Card>

      <div className="mt-8 text-center">
        <Card glass className="inline-block p-4">
          <p className="text-slate-400">
            Your documents will be processed and analyzed by our AI system for instant querying
          </p>
        </Card>
      </div>
    </div>
  );
}