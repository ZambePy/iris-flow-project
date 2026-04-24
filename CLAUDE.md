# IrisFlow — Contexto do Projeto para Claude Code

## O que é o IrisFlow

Software SaaS de comunicação assistiva que usa visão computacional para rastrear o movimento
da íris via webcam comum (a partir de R$ 80) e traduzir esse movimento em ações digitais:
seleção de letras, respostas sim/não, controle de interface e síntese de voz em PT-BR.

**Público-alvo:** Pessoas com tetraplegia, ELA, AVC e outras condições que impedem movimento
voluntário. Clínicas de reabilitação, hospitais e famílias de pacientes.

**Diferencial principal:** Funciona em webcam comum (sem hardware proprietário). Custo 10x menor
que concorrentes como Tobii Dynavox (R$ 15k–80k).

---

## Time

| Pessoa | Papel |
|--------|-------|
| Dev 1 (você) | Backend Python, engine de IA, API, integração |
| Marcus | Backend Python, engine de IA, API, integração |
| Vinicius | Modelo de negócios, marketing, vendas |

---

## Stack tecnológica

### Engine de IA (Python)
- **Visão computacional:** MediaPipe Face Mesh + OpenCV 4.x
- **Calibração adaptativa:** TensorFlow Lite / ONNX Runtime (modelo LSTM leve)
- **TTS PT-BR:** Coqui TTS (principal), pyttsx3 (fallback)
- **STT PT-BR:** OpenAI Whisper (self-hosted, modelo "small")
- **Linguagem:** Python 3.11+

#### Pipeline de rastreamento (iris_tracker.py — implementado)

**Captura:** câmera a 640×480 para reduzir carga do MediaPipe sem perda de precisão
(landmarks retornam coordenadas normalizadas [0,1] independente da resolução).

**Feature combinada por frame:**
`ratio = 0.7 × eye_ratio + 0.3 × face_ratio`
- `eye_ratio`: posição da íris relativa aos cantos ósseos do olho (baseline estável)
- `face_ratio`: posição da íris relativa ao span lateral da face (~3× mais largo)
- Ambos suavizados por EMA (α=0.20) antes da combinação

**Suavização do cursor:**
1. EMA α=0.20 sobre o ratio combinado (reduz ruído de alta frequência)
2. Deadzone adaptativa em espaço normalizado — trava cursor quando olho está parado;
   válvula de escape após MAX_FROZEN=25 frames para não bloquear movimentos lentos
3. Filtro de Kalman 2D (velocidade constante + amortecimento 0.85) em pixels
4. LERP fator 0.10 entre posição atual e alvo do Kalman (deslizamento suave)
5. Cursor mantém última posição conhecida quando íris não é detectada (sem indicação visual de pausa)

**Calibração híbrida em 2 etapas (~2 minutos no total):**
- *Etapa 1 — Pontos fixos:* grade 3×3 (9 pontos) + ponto central dedicado com peso 3×
  e 60 frames de coleta; regressão polinomial (afim/bilinear/quadrática por nº de pontos)
  com mínimos quadrados ponderados
- *Etapa 2 — Trajetórias dinâmicas:* 6 padrões de movimento (cruz horizontal, cruz vertical,
  diagonal Z, figura-8, borda da tela, espiral); coleta 1 amostra a cada 2 frames com
  compensação de lag oculomotor de 4 frames (~130 ms); ~500 pares (íris→tela) combinados
  com os 10 pontos fixos (peso 0.10 por amostra de trajetória)

**Óculos:** toggle manual `[G]` na tela de setup — eleva confiança mínima do MediaPipe
de 0.50 para 0.70, melhorando estabilidade de landmarks sob reflexo de lente.
Usuários de óculos devem garantir boa iluminação frontal (tela de setup exibe indicador).

### Backend / API
- **Framework:** FastAPI
- **Banco dev:** SQLite → PostgreSQL (Supabase em produção)
- **Cache:** Redis (a partir do Mês 4, se necessário)
- **Auth:** Supabase Auth (JWT + OAuth Google)
- **Pagamentos:** Stripe (assinaturas recorrentes)

### Frontend (Mês 5+)
- **Framework:** Next.js 14 (React) + Tailwind CSS + shadcn/ui
- **Deploy:** Vercel (gratuito)

### Infra
- **Containerização:** Docker + Docker Compose
- **CI/CD:** GitHub Actions
- **Deploy backend:** Railway ou Render
- **Versionamento:** Git + GitHub

---

## Arquitetura do sistema

```
Edge (dispositivo do usuário)
  └── MediaPipe + OpenCV + modelo LSTM
      └── Processa câmera localmente (nenhum dado biométrico vai para nuvem)

Backend (nuvem — Railway + Supabase)
  └── FastAPI + PostgreSQL + Redis
      └── Gerencia usuários, licenças, configurações, webhooks Stripe

Frontend (Vercel)
  └── Next.js — dashboard de assinantes: ativa licença, gerencia plano, baixa software

Mobile/TV (Mês 5+)
  └── React Native (mobile) + Android TV app
      └── Comunicam com edge via WebSocket na mesma rede Wi-Fi
```

**Regra crítica de privacidade (LGPD):** Dados da câmera e biometria são processados 100%
localmente no dispositivo. NUNCA enviar frames ou dados biométricos para a nuvem.

---

## Estrutura de pastas

```
irisflow/
├── engine/                  # Engine de IA e visão computacional (Fase 1)
│   ├── iris_tracker.py      # MediaPipe + OpenCV: detecção de íris
│   ├── calibration.py       # Calibração manual e adaptativa (LSTM)
│   ├── tts.py               # Text-to-speech PT-BR
│   ├── stt.py               # Speech-to-text (Whisper)
│   └── virtual_keyboard.py  # Teclado virtual
├── api/                     # FastAPI backend (Fase 2 — Mês 3)
│   ├── main.py
│   ├── models.py
│   ├── database.py
│   └── routes/
│       ├── auth.py
│       ├── users.py
│       └── subscriptions.py
├── dashboard/               # Next.js (Fase 3 — Mês 5)
├── tests/                   # pytest
├── docs/                    # Documentação técnica
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Cronograma resumido

| Fase | Meses | Foco |
|------|-------|------|
| 1 — Fundação + Engine | 1–2 | MediaPipe, calibração, teclado virtual, TTS |
| 2 — Backend + API | 3–4 | FastAPI, Supabase, Stripe, Whisper, Docker |
| 3 — Dashboard + MVP | 5–6 | Next.js, landing page, deploy produção |
| 4 — Mobile + TV | 7 | React Native, Android TV, pitch final |

---

## Convenções de código

- **Python:** PEP 8, type hints em todas as funções, docstrings em PT-BR
- **Commits:** mensagens em inglês no imperativo (`feat: add iris tracker`, `fix: calibration offset`)
- **Branches:** `feature/nome-da-feature`, `fix/nome-do-bug`, `chore/nome-da-tarefa`
- **Pull Requests:** sempre revisar antes de mergear na `main`
- **Testes:** pytest, manter cobertura > 70% nos módulos críticos
- **Variáveis de ambiente:** nunca hardcodar segredos, sempre usar `.env`

---

## Variáveis de ambiente necessárias

Ver `.env.example` para lista completa. As principais:

- `SUPABASE_URL` e `SUPABASE_KEY` — banco e auth
- `STRIPE_SECRET_KEY` e `STRIPE_WEBHOOK_SECRET` — pagamentos
- `DATABASE_URL` — string de conexão (SQLite em dev, Postgres em prod)

---

## Comandos úteis

```bash
# Rodar engine localmente (Windows — usar sempre o .venv)
.venv/Scripts/python.exe engine/iris_tracker.py

# Rodar API em desenvolvimento
uvicorn api.main:app --reload

# Rodar testes
pytest tests/ -v

# Subir ambiente completo com Docker
docker-compose up

# Instalar dependências
pip install -r requirements.txt
```

---

## Contexto de negócio

- **Modelo:** B2B (clínicas/hospitais) + B2C (famílias)
- **Preço:** ~R$ 79/mês por assinante, licença institucional para clínicas
- **Meta:** 10.000 usuários ativos até 2030
- **Custo total estimado (7 meses):** ~R$ 2.360
- **Infra inicial:** gratuita (Railway free tier + Supabase free tier + Vercel)
