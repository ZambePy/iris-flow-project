# IrisFlow — Arquitetura do Sistema

## Visão geral

```
┌─────────────────────────────────────────────────┐
│                DISPOSITIVO DO USUÁRIO           │
│                  (Edge / Local)                  │
│                                                 │
│  ┌──────────────┐    ┌──────────────────────┐  │
│  │   Webcam     │───▶│   engine/            │  │
│  │  (qualquer)  │    │   iris_tracker.py    │  │
│  └──────────────┘    │   calibration.py     │  │
│                      │   virtual_keyboard.py│  │
│  ┌──────────────┐    │   tts.py             │  │
│  │  Microfone   │───▶│   stt.py (Whisper)   │  │
│  │  (opcional)  │    └──────────┬───────────┘  │
│                                 │               │
│  ◼ Dados biométricos 100% locais               │
│  ◼ Nenhum frame de câmera vai para a nuvem     │
└─────────────────────────────────────────────────┘
                         │
                         │ HTTPS (config + licença)
                         ▼
┌─────────────────────────────────────────────────┐
│                    NUVEM                        │
│                                                 │
│  ┌──────────────────┐   ┌───────────────────┐  │
│  │   FastAPI        │   │   Supabase        │  │
│  │   api/main.py    │──▶│   PostgreSQL      │  │
│  │                  │   │   Auth (JWT)      │  │
│  └──────────┬───────┘   └───────────────────┘  │
│             │                                   │
│             │           ┌───────────────────┐  │
│             └──────────▶│   Stripe          │  │
│                         │   Assinaturas     │  │
│                         └───────────────────┘  │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│                 FRONTEND (Mês 5+)               │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │   Next.js — Vercel                       │  │
│  │   dashboard/ (assinante)                 │  │
│  │   landing page (marketing)               │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Regras de privacidade (LGPD)

- **Dados biométricos:** processados APENAS localmente no dispositivo do usuário
- **O que vai para a nuvem:** email, nome, configurações de calibração (coeficientes matemáticos, não biometria bruta), status de assinatura
- **Nunca enviado:** frames de câmera, posições brutas da íris, dados de saúde do usuário

## Fluxo de uma sessão

1. Usuário abre o software no computador
2. Engine verifica licença ativa via API (HTTPS)
3. Teclado virtual + rastreamento de íris iniciam localmente
4. Dados de calibração do servidor são carregados (coeficientes apenas)
5. Sessão roda 100% offline — sem latência de rede no rastreamento

## Latência alvo

- Rastreamento de íris: < 30ms (MediaPipe em CPU comum)
- Seleção por dwell time padrão: 1.5s (configurável por usuário)
