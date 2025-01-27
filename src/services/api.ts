import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.data);
      throw new Error(error.response.data.detail || 'An error occurred');
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.request);
      throw new Error('Network error - please check your connection');
    } else {
      // Other errors
      console.error('Error:', error.message);
      throw new Error('An unexpected error occurred');
    }
  }
);

export const uploadDocument = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await api.post('/api/pdf/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
        console.log(`Upload Progress: ${percentCompleted}%`);
      },
    });
    return response.data;
  } catch (error) {
    console.error('Upload Error:', error);
    throw error;
  }
};

export const askQuestion = async (question: string, documentIds?: string[]) => {
  try {
    const response = await api.post('/api/qa/ask', { 
      question,
      documentIds 
    });
    return response.data;
  } catch (error) {
    console.error('Question Error:', error);
    throw error;
  }
};

export const listDocuments = async () => {
  try {
    const response = await api.get('/api/pdf/list');
    return response.data;
  } catch (error) {
    console.error('List Documents Error:', error);
    throw error;
  }
};