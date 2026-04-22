# IrisFlow

> Comunicar vai além das palavras.

Software SaaS de comunicação assistiva que usa visão computacional para rastrear o movimento
da íris via webcam comum e traduzir esse movimento em ações digitais — texto, voz e controle
de interface — para pessoas com tetraplegia, ELA, AVC e doenças neurodegenerativas.

---

## Por que o IrisFlow?

Soluções de eye-tracking existentes custam entre **R$ 15.000 e R$ 80.000**, inacessíveis para
a maioria das famílias e instituições públicas de saúde. O IrisFlow funciona em qualquer
webcam convencional (a partir de R$ 80), custando **10x menos** que os concorrentes globais,
com interface e síntese de voz nativas em **português brasileiro**.

---

## Funcionalidades

- **Rastreamento de íris em tempo real** via MediaPipe + OpenCV (latência < 30ms)
- **Teclado virtual** controlado pelo olhar
- **Síntese de voz PT-BR** (Coqui TTS)
- **Reconhecimento de voz PT-BR** para usuários com controle vocal parcial (Whisper)
- **Calibração adaptativa por IA** — aprende as características do olho de cada usuário
- **Dashboard web** para assinantes (Next.js) — ativa licença, gerencia plano
- **Privacidade total:** processamento 100% local, nenhum dado biométrico vai para a nuvem

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Visão computacional | MediaPipe Face Mesh + OpenCV 4.x |
| Modelo de calibração | TensorFlow Lite / ONNX Runtime (LSTM) |
| TTS | Coqui TTS (PT-BR) |
| STT | OpenAI Whisper (self-hosted) |
| API | FastAPI (Python 3.11+) |
| Banco de dados | SQLite (dev) → PostgreSQL via Supabase (prod) |
| Auth | Supabase Auth (JWT + OAuth) |
| Pagamentos | Stripe |
| Frontend | Next.js 14 + Tailwind CSS + shadcn/ui |
| Infra | Docker + Railway + Vercel |
| CI/CD | GitHub Actions |

---

## Como rodar localmente

### Pré-requisitos

- Python 3.11+
- pip
- Webcam conectada
- Git

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/irisflow.git
cd irisflow

# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com suas credenciais
```

### Rodando o engine

```bash
python engine/iris_tracker.py
```

### Rodando a API

```bash
uvicorn api.main:app --reload
# Acesse: http://localhost:8000/docs
```

### Rodando com Docker

```bash
docker-compose up
```

### Rodando os testes

```bash
pytest tests/ -v
```

---

## Estrutura do projeto

```
irisflow/
├── engine/                  # Engine de IA e visão computacional
│   ├── iris_tracker.py      # Detecção de íris (MediaPipe + OpenCV)
│   ├── calibration.py       # Calibração manual e adaptativa
│   ├── tts.py               # Text-to-speech PT-BR
│   ├── stt.py               # Speech-to-text (Whisper)
│   └── virtual_keyboard.py  # Teclado virtual
├── api/                     # FastAPI backend
│   ├── main.py              # Entrypoint da API
│   ├── models.py            # Schemas do banco
│   ├── database.py          # Conexão com banco de dados
│   └── routes/              # Rotas organizadas por domínio
│       ├── auth.py
│       ├── users.py
│       └── subscriptions.py
├── dashboard/               # Frontend Next.js (Fase 3)
├── tests/                   # Testes automatizados (pytest)
├── docs/                    # Documentação técnica
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── CLAUDE.md                # Contexto para o Claude Code
```

---

## Cronograma de desenvolvimento

| Mês | Entregas |
|-----|---------|
| 1 | Engine de detecção, landmarks do olho mapeados |
| 2 | Teclado virtual + voz PT-BR + demo interna |
| 3 | API REST, autenticação, Whisper integrado |
| 4 | Stripe, calibração IA, demo com clínica piloto |
| 5 | Dashboard web, landing page, deploy em produção |
| 6 | Painel B2B, primeiros assinantes pagantes |
| 7 | App TV + React Native mobile, pitch final |

---

## Contribuindo

1. Crie um branch: `git checkout -b feature/nome-da-feature`
2. Faça suas alterações com commits claros: `feat: descrição`
3. Abra um Pull Request para `main`
4. Aguarde revisão antes de mergear

---

## Licença

Proprietária — IrisFlow Tecnologia Ltda. © 2026
