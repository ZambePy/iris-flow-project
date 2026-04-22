# IrisFlow — Endpoints da API

Base URL: `https://api.irisflow.com.br` (produção) | `http://localhost:8000` (dev)

## Status

| Endpoint | Método | Status |
|----------|--------|--------|
| `/` | GET | ✅ Implementado |
| `/health` | GET | ✅ Implementado |
| `/auth/*` | - | 🔲 Mês 3 |
| `/users/*` | - | 🔲 Mês 3 |
| `/subscriptions/*` | - | 🔲 Mês 4 |

## Endpoints planejados

### Auth
```
POST /auth/register        — Criar conta
POST /auth/login           — Login (retorna JWT)
POST /auth/logout          — Logout
POST /auth/refresh         — Renovar token
POST /auth/forgot-password — Recuperar senha
```

### Usuários
```
GET    /users/me                — Dados do usuário logado
PATCH  /users/me                — Atualizar perfil
GET    /users/me/calibration    — Configurações de calibração
PUT    /users/me/calibration    — Salvar configurações de calibração
```

### Assinaturas
```
POST /subscriptions/checkout   — Criar sessão de pagamento Stripe
POST /subscriptions/webhook    — Webhook do Stripe (ativa/desativa licença)
GET  /subscriptions/status     — Status atual da assinatura
```

## Autenticação

Todos os endpoints protegidos requerem header:
```
Authorization: Bearer <jwt_token>
```
