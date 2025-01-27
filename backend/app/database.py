from sqlalchemy import create_engine, Column, String, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL
import json

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    title = Column(String)
    content = Column(String)
    embedding = Column(String)  # Store as JSON string
    created_at = Column(DateTime)
    
    def set_embedding(self, embedding_list):
        self.embedding = json.dumps(embedding_list)
        
    def get_embedding(self):
        return json.loads(self.embedding) if self.embedding else []

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
Base.metadata.create_all(bind=engine)