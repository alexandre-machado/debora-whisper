# Ideia: tools, MCP e controle de um harness por voz

> Ideia estacionada em 2026-10-08. Nada foi implementado ainda.

## Objetivo

No voice chat, o LLM local (Qwen3-8B no OpenVINO) vira o atendente de voz e
repassa o trabalho pesado a um harness de mercado: Claude Code primeiro, Codex
depois. Você fala, o Qwen decide se responde sozinho ou se repassa. Quando o
harness termina, o Qwen resume o resultado em uma ou duas frases faladas.
Diffs, código e logs ficam no overlay e no log, não vão para o TTS.

Uma tarefa no harness leva minutos, então ela roda em segundo plano. A
ferramenta só responde "enviado". Os eventos do harness, como "terminou" ou
"quer rodar `git push`, autoriza?", chegam depois como notificações faladas.

## O que foi verificado (2026-10-08)

- **OpenVINO GenAI 2026.4.1:** `Tokenizer.apply_chat_template(history,
  add_generation_prompt, chat_template, tools=..., extra_context=...)` aceita
  `tools`. O Qwen3 responde com `<tool_call>{"name": ..., "arguments":
  ...}</tool_call>`.
- **Claude Code 2.1.294:** `claude -p --input-format stream-json
  --output-format stream-json --verbose --resume <session-id>` mantém uma
  sessão viva por stdio, com JSON por linha nos dois sentidos. Falta
  confirmar se os pedidos de permissão chegam pelo stdio; o Agent SDK faz
  isso.
- **Codex CLI 0.160:** não tem mais `codex mcp-server`. Tem `codex exec`, com
  `resume`, e `codex app-server`, um JSON-RPC por stdio ainda experimental.
- **SDK `mcp` em Python:** não está no ambiente do app.
- **Limitação:** o app não consegue pilotar um terminal interativo que já
  está aberto. Ele roda a própria sessão headless, que depois pode ser aberta
  no terminal com `claude --resume <id>`.

## Fases propostas

0. **Provas de conceito**, em scripts fora do app:
   - quanto o Qwen3-8B int4 acerta a ferramenta e os argumentos em português,
     e quanto isso custa em latência;
   - uma sessão do Claude Code em stream-json: mensagens seguidas,
     interrupção e permissões;
   - um turno e uma aprovação pelo `codex app-server`.
1. **Tool calling no `llm_server`:**
   - o pedido passa a levar `tools`;
   - ao aparecer `<tool_call>`, o servidor segura o stream e manda
     `{"id", "tool_call": {...}}`, para o TTS nunca falar o JSON;
   - o `voice_chat` ganha um registro de ferramentas e o laço chamada →
     mensagem `tool` → nova geração, com limite de rodadas. Cada rodada custa
     uns 1–2 s.
2. **Adaptador do Claude Code:**
   - uma `HarnessSession` em processo próprio, no mesmo padrão do
     `llm_server`, com `start(pasta, resume)`, `send`, `interrupt` e eventos;
   - ferramentas para o Qwen: `harness_enviar`, `harness_status`,
     `harness_aprovar` e `harness_parar`.
3. **Adaptador do Codex:** pelo `app-server`. Se ele estiver instável, uso o
   `codex exec --json` com `resume`, que não tem aprovação interativa e
   precisa do sandbox `workspace-write`.
4. **Cliente MCP genérico:** um `"mcp_servers"` no `config.json`, no formato
   do Claude, cujas ferramentas entram no mesmo registro.

## Decisões em aberto (com recomendação)

- **Roteamento:** o Qwen decide por tool call, e "Claude, …" ou "Codex, …"
  força o repasse. A alternativa é um modo harness, em que tudo vai direto
  para o harness.
- **Permissões:** `acceptEdits`, com sim ou não por voz para comandos de
  shell. As alternativas são pedir autorização para tudo, ou deixar tudo
  liberado numa pasta isolada.
- **Ordem:** Claude Code antes do Codex.

## Nota relacionada: velocidade do TTS

O T3 do Chatterbox é limitado pelo CPU: cada token leva uns 37 ms, a maior
parte disparando kernels pequenos, com a GPU quase parada.

- Trocar o resto das camadas para sdpa não deu ganho mensurável: o ruído
  entre rodadas iguais foi de ±15%.
- O vazamento de hooks de atenção foi corrigido no commit `3b31210`.
- O ganho grande que resta é capturar o passo do T3 em CUDA graphs com KV
  cache estático. Isso não foi tentado.
