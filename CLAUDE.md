# IrisFlow — Contexto do Projeto para Claude Code

## O que é o IrisFlow

Software SaaS de comunicação assistiva que usa visão computacional para rastrear o movimento
da íris via webcam comum (a partir de R$ 80) e traduzir esse movimento em ações digitais:
seleção de letras, respostas sim/não, controle de interface e síntese de voz em PT-BR.

**Público-alvo:** Pessoas com tetraplegia, ELA, AVC e outras condições que impedem movimento
voluntário. Clínicas de reabilitação, hospitais e famílias de pacientes.

**Diferencial principal:** Funciona em webcam comum (sem hardware proprietário). Custo 10x menor
que concorrentes como Tobii Dynavox (R$ 15k–80k).

**Meta imediata:** Demo funcional na AACD (Associação de Assistência à Criança Deficiente)
em até 70 dias. Paciente real usa o sistema ao vivo na frente da equipe clínica.

---

## Time

| Pessoa | Papel |
|--------|-------|
| Gabriel | Backend Python, engine de IA, API, integração |
| Marcus | Backend Python, engine de IA, API, integração |
| Vinicius | Modelo de negócios, marketing, vendas |

---

## Estado atual do projeto (maio 2026)

O engine de rastreamento está implementado e funcional. O que existe:

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `engine/iris_tracker.py` | ✅ Completo | MediaPipe + head pose 6DoF + gaze 3D + Kalman |
| `engine/calibration.py` | ⚠️ Funcional, otimizar | 3 fases: varredura + 9 pontos + trajetórias |
| `engine/calib_store.py` | ✅ Completo | SQLite — persiste sessões de calibração |
| `engine/virtual_keyboard.py` | ⚠️ Funcional, melhorar | QWERTY com dwell time fixo |
| `engine/tts.py` | ✅ Completo | Coqui TTS + pyttsx3 fallback |
| `api/main.py` | 🔧 Estrutura | FastAPI básico, rotas vazias |
| `engine/user_profile.json` | ✅ Gerado | Perfil ocular por paciente |
| `engine/calibration.db` | ✅ Gerado | Histórico de sessões SQLite |

O que ainda não existe e precisa ser criado:

- `engine/quick_comm.py` — painel de comunicação rápida (8 botões emocionais)
- `engine/word_predict.py` — predição de palavras PT-BR local
- `engine/metrics.py` — métricas clínicas por sessão
- Modo cuidador via FastAPI + HTML simples
- Installer Windows (PyInstaller + NSIS)

---

## Plano de 70 dias — Sprints

### Sprint 1 · Dias 1–17 · Destravar calibração + dwell click

**Problema a resolver:** calibração demora demais para pacientes com condições progressivas.
Fase 0 (10s de varredura) + Fase 2 (trajetórias opcionais mas longas) = frustração clínica.

**Tarefas:**

1. **Calibração rápida** (`engine/calibration.py`)
   - Fase 0: reduzir `SWEEP_FRAMES` de 150 para 90 (3s por sweep em vez de 5s)
   - Fase 1: reduzir `COUNTDOWN` de 2.0s para 1.2s e `COLLECT` de 1.5s para 1.0s
   - Fase 2: tornar pulada por padrão — mostrar `[T]` para ativar trajetórias, `[SPACE]` para pular
   - Auto-recalibração silenciosa: a cada 300 frames no `_run_tracking()`, coletar 1 amostra
     e re-ajustar o modelo com histórico — corrige drift sem interromper o usuário

2. **Calibração cruzada com histórico** (`engine/calibration.py`)
   - Após Fase 1 (e Fase 2 se ativada), chamar `calib_store.load_historical()` e combinar
     pontos históricos com pontos da sessão atual antes de chamar `model.fit()`
   - Sessões mais antigas têm peso menor (decay=0.60 já implementado no calib_store)
   - Salvar sessão no `calib_store` após calibração bem-sucedida (`save_session()`)
   - Isso resolve a imprecisão nas bordas/cantos que a grade 3×3 não cobre bem

3. **Dwell click inteligente** (`engine/virtual_keyboard.py`)
   - Dwell adaptativo por tipo de tecla: letras=1.2s, teclas largas (ESPACO/FALAR/LIMPAR)=0.8s,
     ações destrutivas (APAGAR/LIMPAR)=1.5s
   - Hitbox expandida: +20% na área de hit sem mudar visual (reduz esforço ocular)
   - Intent check: só ativar se olhar ficou nos últimos 3 frames dentro da hitbox
     (evita ativação acidental ao passar por cima)
   - Barra de progresso mais visível: 8px de altura, cor azul→verde conforme progride

**Entregável do Sprint 1:** sistema roda 10 minutos contínuos com 1 paciente real sem travar
ou exigir recalibração manual.

---

### Sprint 2 · Dias 18–34 · Comunicação rápida + predição de palavras

**Tarefas:**

1. **Painel de comunicação rápida** (`engine/quick_comm.py` — criar do zero)
   - 8 botões gigantes (ocupam 70% da tela): Dor, Desconforto, Fome, Água, Frio, Calor,
     Ajuda, Emergência
   - 1 olhar prolongado (dwell 0.8s) → dispara alerta sonoro + mensagem no log
   - Alternável com o teclado via botão de contexto
   - Este módulo tem mais impacto clínico imediato que o teclado completo

2. **Predição de palavras PT-BR** (`engine/word_predict.py` — criar do zero)
   - Modelo n-gram treinado localmente (sem internet, sem API)
   - Sugerir top 3 palavras acima do teclado
   - Memória de frases frequentes por paciente (salvar em `user_profile.json`)
   - Integrar ao `VirtualKeyboard` como linha extra de sugestões

3. **Anti-fadiga ocular** (`engine/calibration.py` — camada 5)
   - Magnetismo de alvo: expandir hitbox invisível em +40% para elementos próximos ao olhar
   - Pausa automática configurável após N minutos de uso contínuo
   - Cooldown visual: reduzir brilho da tela em 20% após 15 min de uso

**Entregável do Sprint 2:** paciente comunica necessidade básica (água, dor) em menos de 3s
sem precisar do cuidador.

---

### Sprint 3 · Dias 35–51 · Perfil adaptativo + modo cuidador

**Tarefas:**

1. **Perfil adaptativo por paciente** (`engine/calib_store.py` + `user_profile.json`)
   - Expandir `user_profile.json` para registrar: dwell ideal por sessão, nível de fadiga
     médio, frases mais usadas, horário de maior precisão
   - O sistema aprende e ajusta automaticamente dwell_time e sensibilidade por sessão
   - Detectar degradação de precisão (drift acima do threshold) e sugerir recalibração

2. **Modo cuidador** (`api/` — usar FastAPI existente)
   - Interface web simples (HTML + FastAPI, sem Next.js ainda) acessível pelo celular do
     cuidador na mesma rede Wi-Fi
   - Funcionalidades: configurar frases rápidas, ajustar dwell_time, ver log de uso em
     tempo real, acionar recalibração remotamente
   - Rota WebSocket já existe no `api/` — usar para live updates

**Entregável do Sprint 3:** cuidador configura frases e sensibilidade pelo celular sem tocar
no PC do paciente.

---

### Sprint 4 · Dias 52–70 · Métricas clínicas + demo AACD

**Tarefas:**

1. **Métricas clínicas** (`engine/metrics.py` — criar do zero)
   - Registrar por sessão: estabilidade ocular (desvio padrão da posição), tempo médio de
     resposta por tecla, número de ativações acidentais, fadiga ao longo do tempo
   - Exportar relatório PDF simples para o clínico (usar reportlab ou fpdf2)
   - Dados ficam 100% locais (LGPD) — nunca sobem para nuvem

2. **Garantia offline** (revisar todos os módulos)
   - Remover qualquer dependência de rede do loop principal (TTS, predição, calibração)
   - TTS: garantir que pyttsx3 funciona como fallback quando Coqui falha
   - Predição: modelo n-gram roda 100% local, sem chamada OpenAI

3. **Installer Windows** (PyInstaller + NSIS)
   - Gerar `.exe` com um clique — sem precisar instalar Python
   - Testar em máquina limpa (sem `.venv`)
   - Script de demo de 5 minutos documentado para o cuidador apresentar à equipe clínica

**Entregável do Sprint 4:** demo ao vivo na AACD — paciente usa o sistema na frente da equipe,
cuidador mostra o painel pelo celular, clínico vê as métricas de precisão.

---

## Fora do escopo dos 70 dias — não implementar agora

As funcionalidades abaixo são importantes mas entram **após** o feedback clínico da AACD:

- Clonagem de voz (TTS com voz do paciente)
- Integração Alexa / smart home
- App mobile React Native
- Dashboard Next.js completo
- Stripe / pagamentos
- Docker em produção
- Avatar facial

---

## Estratégia de dados e calibração cruzada

### Por que cruzar calibração com histórico

O modelo RBF (`GazeModel`) usa `RBFInterpolator` (thin-plate spline) para mapear
feature de olhar → posição na tela. Com apenas 9 pontos (grade 3×3), a interpolação é
fraca nas bordas e cantos. Cruzar com histórico de sessões anteriores resolve isso.

### Como funciona (já arquitetado, falta ligar)

O `calib_store.py` já implementa tudo. A chamada que falta em `calibration.py`:

```python
from calib_store import load_historical, save_session

# Após fase 1 (e fase 2 se ativada):
hist_feats, hist_pos, hist_weights = load_historical(
    max_sessions=5,   # usa até 5 sessões anteriores
    decay=0.60,       # sessões mais antigas valem menos
    max_error_px=120, # ignora sessões ruins
)

# Combina sessão atual + histórico
all_feats   = feats1 + feats2 + hist_feats
all_pos     = pos1   + pos2   + hist_pos
all_weights = weights_atuais  + hist_weights

model.fit(all_feats, all_pos, all_weights)

# Salvar sessão atual após validação
save_session(profile, feats1, pos1, feats2, pos2, error_px=avg_err)
```

### Evolução por sessão

| Sessão | Pontos disponíveis | Efeito |
|--------|--------------------|--------|
| 1ª | 9 (grade atual) | igual ao atual |
| 2ª | ~60 (9 + 9×decay) | bordas melhoram |
| 5ª | ~200+ pontos | modelo já conhece o padrão ocular do paciente |

### Datasets públicos (uso futuro — pós-AACD)

Não usar nos 70 dias. Após ter 20–30 sessões reais de pacientes:

- **GazeCapture** (MIT, gazecapture.csail.mit.edu) — 1.45M frames, 1.450 pessoas.
  Usar como prior fraco (peso 0.03–0.05) para melhorar a primeira sessão de um paciente
  novo antes de ter histórico próprio.
- **MPIIGaze** (perceptualui.org/research/datasets/MPIIGaze) — 213k amostras.
  Útil para validar que o modelo generaliza além dos pacientes da AACD.

O `calib_store.db` gerado nas sessões da AACD é o ativo mais valioso do projeto —
são dados de pacientes brasileiros com ELA/tetraplegia usando webcam comum,
que nenhum concorrente tem.

---

## Stack tecnológica

### Engine de IA (Python)
- **Visão computacional:** MediaPipe Face Mesh + OpenCV 4.x
- **Calibração:** RBFInterpolator (scipy) — thin-plate spline
- **Suavização:** Filtro de Kalman 2D (filterpy) + deadzone adaptativa + LERP
- **TTS PT-BR:** Coqui TTS (principal), pyttsx3 (fallback offline)
- **Linguagem:** Python 3.11+

### Pipeline de rastreamento (`iris_tracker.py` — implementado)

**Captura:** câmera a 640×480, thread dedicada com `queue.Queue` (sem perda de frames).

**Feature combinada por frame:**
- `eye_ratio`: posição da íris relativa aos cantos ósseos do olho (baseline estável)
- `face_ratio`: posição da íris relativa ao span lateral da face
- Combinados: `ratio = 0.7 × eye_ratio + 0.3 × face_ratio`

**Pipeline de suavização (5 camadas):**
1. EMA α=0.20 sobre o ratio combinado
2. Head pose 6DoF via solvePnP — compensa movimento de cabeça
3. Deadzone adaptativa — trava cursor quando olho está parado
4. Filtro de Kalman 2D (velocidade constante + amortecimento 0.85)
5. LERP 0.08 entre posição atual e alvo do Kalman

### Backend / API
- **Framework:** FastAPI
- **Banco dev:** SQLite → PostgreSQL (Supabase em produção)
- **Auth:** Supabase Auth (JWT)
- **Pagamentos:** Stripe (pós-AACD)

### Frontend (pós-AACD)
- **Framework:** Next.js 14 + Tailwind CSS + shadcn/ui
- **Deploy:** Vercel

---

## Arquitetura do sistema

```
Edge (dispositivo do paciente — 100% local)
  └── iris_tracker.py      → captura + head pose + gaze 3D
  └── calibration.py       → calibração + modelo RBF + auto-drift
  └── calib_store.py       → SQLite local (calibration.db)
  └── virtual_keyboard.py  → teclado + dwell click
  └── quick_comm.py        → painel emocional (Sprint 2)
  └── word_predict.py      → predição PT-BR local (Sprint 2)
  └── tts.py               → síntese de voz offline
  └── metrics.py           → métricas clínicas (Sprint 4)

Backend (rede local Wi-Fi — modo cuidador)
  └── api/main.py          → FastAPI WebSocket + rotas de config
      └── acessível pelo celular do cuidador na mesma rede

Nuvem (pós-AACD)
  └── Supabase + Railway    → usuários, licenças, dashboard
  └── LGPD: NUNCA sobe biometria ou frames para nuvem
```

---

## Estrutura de pastas

```
irisflow/
├── engine/
│   ├── iris_tracker.py      # ✅ Rastreamento: MediaPipe + head pose + gaze
│   ├── calibration.py       # ⚠️ Calibração — Sprint 1: acelerar + cruzar histórico
│   ├── calib_store.py       # ✅ SQLite: persiste sessões de calibração
│   ├── virtual_keyboard.py  # ⚠️ Teclado — Sprint 1: dwell inteligente
│   ├── tts.py               # ✅ TTS PT-BR
│   ├── quick_comm.py        # 🔲 Criar no Sprint 2
│   ├── word_predict.py      # 🔲 Criar no Sprint 2
│   ├── metrics.py           # 🔲 Criar no Sprint 4
│   ├── user_profile.json    # ✅ Perfil ocular por paciente
│   └── calibration.db       # ✅ Histórico de sessões
├── api/
│   ├── main.py              # ✅ FastAPI base
│   ├── models.py
│   ├── database.py
│   └── routes/              # 🔲 Modo cuidador — Sprint 3
├── tests/
├── docs/
├── main.py                  # Entry point
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Convenções de código

- **Python:** PEP 8, type hints em todas as funções, docstrings em PT-BR
- **Commits:** mensagens em inglês no imperativo (`feat: add dwell intent check`)
- **Branches:** `feature/nome`, `fix/nome`, `chore/nome`
- **Testes:** pytest, cobertura > 70% nos módulos críticos
- **Segredos:** nunca hardcodar, sempre usar `.env`
- **LGPD:** dados biométricos (frames, posição de íris) NUNCA saem do dispositivo local

---

## Comandos úteis

```bash
# Rodar engine (Windows)
.venv\Scripts\python.exe engine/iris_tracker.py

# Rodar engine (Linux/macOS)
python engine/iris_tracker.py

# Rodar API
uvicorn api.main:app --reload

# Rodar testes
pytest tests/ -v

# Instalar dependências
pip install -r requirements.txt
```

---

## Contexto de negócio

- **Modelo:** B2B (clínicas/hospitais) + B2C (famílias)
- **Preço:** R$ 79/mês familiar, R$ 349/mês clínica, R$ 1.490/mês hospitalar
- **Meta 2030:** 10.000 usuários ativos
- **Custo MVP:** ~R$ 1.555 (desenvolvimento = time próprio)
- **Infra inicial:** gratuita (Railway + Supabase + Vercel free tiers)
- **Próximo marco:** demo AACD em 70 dias → feedback clínico real → iterar