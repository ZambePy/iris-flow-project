"""
IrisFlow — API REST
Gerencia usuários, licenças, assinaturas e configurações de calibração.
Fase 2 (Mês 3+).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(
    title="IrisFlow API",
    description="API de comunicação assistiva por rastreamento de íris",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "IrisFlow API", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# TODO (Mês 3): Registrar rotas
# from api.routes import auth, users, subscriptions
# app.include_router(auth.router, prefix="/auth", tags=["auth"])
# app.include_router(users.router, prefix="/users", tags=["users"])
# app.include_router(subscriptions.router, prefix="/subscriptions", tags=["subscriptions"])
