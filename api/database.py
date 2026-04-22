"""
IrisFlow — Conexão com banco de dados
SQLite em desenvolvimento, PostgreSQL (Supabase) em produção.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./irisflow.db")

# SQLite precisa de connect_args especial para uso com FastAPI
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency para injetar a sessão do banco nas rotas FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
