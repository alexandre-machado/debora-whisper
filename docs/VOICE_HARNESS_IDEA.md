# Plano: conversar com o Claude Code pela Débora

> Revisado em 2026-10-09. Fases 0–2 implementadas na branch local
> `feat/claude-harness`. Fases 3–4, testes e revisão de segurança pendentes. A versão de
> 2026-10-08 (Qwen como atendente que repassa trabalho por tool call) foi
> substituída pela decisão abaixo.

## Decisão

A Débora fica só com a voz: STT (Whisper) e TTS (Chatterbox). Toda a
inteligência pode ir para um harness de mercado, e o primeiro é o **Claude
Code**. O LLM local (Qwen3-8B) continua disponível e é o backend padrão;
`voice_chat_backend: "claude"` seleciona o harness.

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

O Qwen local dava o primeiro token em ~0.5 s. A hipótese inicial atribuía
quase todos os ~6 s à partida; a fase 0 confirmou uma redução com processo
persistente, mas não isolou o custo da API. O processo fica vivo entre
turnos, e a Débora dá um retorno enquanto espera ("Um instante.").

### Como manter o Claude Code vivo

O `claude` 2.1.295 tem tudo para um processo persistente por stdio, com
JSON por linha nos dois sentidos:

```
claude -p --input-format stream-json --output-format stream-json --verbose \
  --include-partial-messages \
  --session-id <uuid> | --resume <id> \
  --permission-mode acceptEdits \
  --permission-prompts host --permission-prompt-tool stdio
```

- `--include-partial-messages`: o texto chega aos pedaços, e o TTS pode
  começar pela primeira frase, como faz hoje com o Qwen.
- `--session-id` / `--resume`: a mesma conversa entre reinícios do app; o
  usuário pode abrir a sessão no terminal com `claude --resume <id>`.
- `--permission-mode`: `acceptEdits`, `auto`, `bypassPermissions`,
  `manual`, `dontAsk` ou `plan`.
- `--permission-prompts host` sozinho não emitiu pedidos no stdout e negou
  o Bash. Com `--permission-prompt-tool stdio`, emitiu `control_request`
  com `request.subtype: "can_use_tool"`; respostas `control_response` com
  `behavior: "allow"`/`"deny"` funcionaram. O padrão da Débora é negar.

### Prova real da fase 0

Dois turnos no mesmo processo mantiveram a palavra `jabuticaba`. Primeiro
delta: 3,781 s frio / 1,766 s quente; repetição com stdio e retomada:
3,031 s / 1,453 s. São amostras, não garantia de latência. O fim do turno é
`result`, não `stream_event.message_stop`. Interrupção por `control_request`
com `subtype: "interrupt"` recebeu confirmação e `result` com
`terminal_reason: "aborted_streaming"`; o processo continuou vivo.
`git --version` foi autorizado e o segundo comando foi negado pelo host.
`--append-system-prompt-file` funcionou, inclusive com idioma/data e retomada
usando `--system-prompt-snapshot off`. JSONs e resultados completos, recortados,
no [benchmark](benchmarks/harness-2026-10-09/README.md).

### Contexto: duas camadas

As regras permanentes do canal de voz vêm de `harness_prompt.md`, incluído no
pacote, ou de `harness_prompt_file`. A Débora acrescenta idioma e data/hora
ao iniciar o processo e passa um arquivo temporário via
`--append-system-prompt-file`. Não usa o prompt local nem reenvia histórico.

O conhecimento do projeto permanece na pasta: seus arquivos de instruções
(`CLAUDE.md`/`AGENTS.md` conforme a configuração do Claude), skills e settings
continuam sob responsabilidade do CLI. A Débora não escreve contexto na pasta.
Uma skill é carregada sob demanda pela descrição, por isso não serve como
veículo de regras de voz que precisam estar sempre presentes.

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

0. **Implementada — prova de conceito** (script fora do app, `docs/benchmarks/` ou
   scratch):
   - sessão em stream-json: duas mensagens seguidas no mesmo processo,
     tempo até o primeiro pedaço de texto com o processo já quente;
   - interrupção no meio de uma resposta (o usuário fala por cima);
   - `--permission-prompts host`: qual mensagem chega pelo stdout quando o
     Claude quer rodar um comando, e como responder sim ou não pelo stdin.
1. **Implementada — `HarnessSession`** em `debora_whisper/harness.py`, no mesmo padrão do
   `LLMProcess` em `voice_chat.py`: `start_harness(config, log)`, `send(texto, on_text, stop)`,
   `interrupt()`, `stop()` e eventos (texto, ferramenta em uso, pedido de
   permissão, fim do turno).
2. **Implementada — ligação ao `VoiceChat`:** um `"voice_chat_backend": "claude"` no
   `config.json` troca o `self._llm` (o construtor já aceita um `llm`
   alternativo) por um que manda o texto ao Claude. O histórico fica no
   Claude, não no `_history` local. O que já existe se reaproveita:
   `speakable`, `without_emoji`, `spoken_numbers`, `split_sentences` e o
   `StreamPlayer`.
   Settings oferece backend e pasta, com Browse/Use home; os dois comandos
   aceitam `--voice-chat-backend` e `--harness-cwd`. O processo é iniciado
   antecipadamente só com chat ligado e backend Claude, ou no primeiro turno.
   Configuração alterada reinicia o harness no próximo turno. UUIDs ficam em
   `~/.debora/harness_session.json`, por pasta; `reset()` ou “nova conversa”
   descarta o vínculo para criar outro UUID no próximo envio. O aviso
   “Um instante.” entra na fila de fala após 1,5 s sem texto.
3. **Permissões por voz:** a Débora fala "quer rodar `git push`, autoriza?",
   e o próximo turno (sim ou não) responde ao pedido em vez de virar uma
   mensagem nova.
4. **Outros harnesses**, depois: Codex (`codex app-server` ou `codex exec
   --json` com `resume`), `agy`, `copilot`.

## Decisões atuais e próximos passos

- **Transcrição crua ou limpa:** crua, decisão confirmada e implementada.
- **Permissões:** `acceptEdits` e negação automática dos pedidos pendentes
  (`harness_permission_response: "deny"`). Callback separado permite
  acrescentar sim/não por voz na fase 3.
- **Pasta de trabalho:** uma pasta fixa no `config.json`
  (`"harness_cwd"`); null significa a pasta pessoal do usuário.
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
