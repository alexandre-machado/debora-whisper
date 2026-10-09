# Plano: conversar com o Claude Code pela Débora

> Revisado em 2026-10-09. Nada foi implementado ainda. A versão de
> 2026-10-08 (Qwen como atendente que repassa trabalho por tool call) foi
> substituída pela decisão abaixo.

## Decisão

A Débora fica só com a voz: STT (Whisper) e TTS (Chatterbox). Toda a
inteligência vai para um harness de mercado, e o primeiro é o **Claude
Code**. O LLM local (Qwen3-8B) deixa de ser o cérebro do chat de voz.

Motivo: o Qwen3-8B se comportava mal como persona, e cada regra a mais no
prompt piorava as respostas (ver `docs/benchmarks/llm-2026-10-09/`). O
harness já conhece o repositório, as ferramentas e a memória do usuário.

## O que aprendemos (2026-10-09)

### Mandar a transcrição crua

Um LLM local limpando a transcrição antes do harness piora o pedido: ele
corrige o óbvio, mas erra o que é do projeto ("dictêixon engine" virou
"Decision Engine"), e soma ~1 s. O Claude Code entendeu os quatro pedidos
crus porque tem o contexto do repositório (`dictation_engine.py`,
`tests/test_voice_chat.py`). **Recomendação:** mandar o texto cru, com um
aviso no prompt de sistema de que ele vem de reconhecimento de voz e pode
ter erros. Tabela completa no README do benchmark.

### Latência dos harnesses

Uma pergunta curta, processo novo a cada chamada:

| CLI | Versão | Tempo |
|---|---|---|
| `claude -p` | 2.1.295 | 6.1 s |
| `codex exec` | 0.161.0 | 6.1 s |
| `copilot -s -p` | 1.0.93 | 7.9 s |
| `agy -p` | 1.3.2 | 9.3 s |

O Qwen local dava o primeiro token em ~0.5 s. Os ~6 s são quase todos de
partida do processo, então o processo do Claude precisa ficar vivo entre
os turnos, e a Débora precisa dar um retorno enquanto espera (um som curto
ou "um instante").

### Como manter o Claude Code vivo

O `claude` 2.1.295 tem tudo para um processo persistente por stdio, com
JSON por linha nos dois sentidos:

```
claude -p --input-format stream-json --output-format stream-json --verbose \
  --include-partial-messages \
  --session-id <uuid> | --resume <id> \
  --permission-mode acceptEdits \
  --permission-prompts host
```

- `--include-partial-messages`: o texto chega aos pedaços, e o TTS pode
  começar pela primeira frase, como faz hoje com o Qwen.
- `--session-id` / `--resume`: a mesma conversa entre reinícios do app; o
  usuário pode abrir a sessão no terminal com `claude --resume <id>`.
- `--permission-mode`: `acceptEdits`, `auto`, `bypassPermissions`,
  `manual`, `dontAsk` ou `plan`.
- `--permission-prompts host`: os pedidos de permissão vão para o
  "host", ou seja, para quem fala o protocolo do SDK pelo stdio (ou para uma
  `--permission-prompt-tool`). `none` nega automaticamente tudo o que
  pediria permissão. **Ainda não testado na prática:** é a primeira coisa a
  provar.

O `agy` também aceita `--input-format stream-json`; o `copilot` só tem
`-p --output-format json --resume`, com um processo por turno.

### Limitações

- O app não pilota um terminal do Claude que já está aberto; ele roda a
  própria sessão headless.
- Respostas do Claude vêm em markdown, com código e listas. Só a parte
  falável vai para o TTS; o resto fica no overlay e no log. O prompt de
  sistema deve pedir respostas curtas e faladas.
- O `language` do `config.json` vale para o Whisper: com `"en"`, fala em
  português sai traduzida para inglês antes de chegar ao harness.

## Fases

0. **Prova de conceito** (script fora do app, `docs/benchmarks/` ou
   scratch):
   - sessão em stream-json: duas mensagens seguidas no mesmo processo,
     tempo até o primeiro pedaço de texto com o processo já quente;
   - interrupção no meio de uma resposta (o usuário fala por cima);
   - `--permission-prompts host`: qual mensagem chega pelo stdout quando o
     Claude quer rodar um comando, e como responder sim ou não pelo stdin.
1. **`HarnessSession`** em `debora_whisper/harness.py`, no mesmo padrão do
   `LLMProcess` em `voice_chat.py`: `start(pasta, resume)`, `send(texto)`,
   `interrupt()`, `stop()` e eventos (texto, ferramenta em uso, pedido de
   permissão, fim do turno).
2. **Ligar ao `VoiceChat`:** um `"voice_chat_backend": "claude"` no
   `config.json` troca o `self._llm` (o construtor já aceita um `llm`
   alternativo) por um que manda o texto ao Claude. O histórico fica no
   Claude, não no `_history` local. O que já existe se reaproveita:
   `speakable`, `without_emoji`, `spoken_numbers`, `split_sentences` e o
   `StreamPlayer`.
3. **Permissões por voz:** a Débora fala "quer rodar `git push`, autoriza?",
   e o próximo turno (sim ou não) responde ao pedido em vez de virar uma
   mensagem nova.
4. **Outros harnesses**, depois: Codex (`codex app-server` ou `codex exec
   --json` com `resume`), `agy`, `copilot`.

## Decisões em aberto (com recomendação)

- **Transcrição crua ou limpa:** crua (ver acima). Falta a confirmação do
  usuário.
- **Permissões:** `acceptEdits`, com sim ou não por voz para comandos de
  shell.
- **Pasta de trabalho:** uma pasta fixa no `config.json`
  (`"harness_cwd"`), com o repositório da Débora como padrão no começo.
- **O Qwen local:** fica como modo offline. Se for mantido, trocar pelo
  Gemma 4 E4B, que foi o melhor no benchmark mas exige `VLMPipeline` no
  `llm_server.py`.

## Fora deste plano, mas relacionado

- **Fim de turno:** testar o Smart Turn v3 junto do Silero
  (`docs/benchmarks/turn-detection-2026-10-09.md`). Com um harness lento,
  cortar o turno cedo demais custa caro.
- **Velocidade do TTS:** o T3 do Chatterbox é limitado pelo CPU (~37 ms por
  token, GPU quase parada). O ganho grande que resta é CUDA graphs com KV
  cache estático; não foi tentado.
