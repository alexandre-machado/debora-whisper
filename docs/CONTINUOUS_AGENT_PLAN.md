# Plano para Agente de Voz Contínuo

Este documento descreve a evolução do projeto de ditado para um agente local
que escuta continuamente, identifica quando o usuário está falando com ele e
executa comandos autorizados. O processamento deve permanecer local, sem
enviar áudio ou transcrições para serviços externos.

## Objetivos

O agente deve:

- manter a captura de áudio ativa sem exigir uma hotkey para cada frase;
- detectar início e fim de fala;
- transcrever em blocos curtos com baixa latência;
- distinguir fala dirigida ao agente de conversa ambiente;
- interpretar intenções;
- executar apenas ações explicitamente permitidas;
- pedir confirmação para ações sensíveis;
- informar estado, erro e resultado ao usuário;
- manter o modo atual de ditado como alternativa.

O setup atual não é contínuo: grava enquanto a hotkey está ativa, transcreve ao
parar e cola o texto na janela em foco. A evolução deve ser implementada como
um modo separado, sem alterar esse comportamento até que o novo fluxo esteja
validado.

## Arquitetura proposta (Foco em Eficiência e Resiliência)

```text
Microfone contínuo (com reconexão automática)
        |
        v
Pré-processamento (Noise Suppression, AGC) -> Baixo custo
        |
        v
Buffer circular de áudio
        |
        v
VAD em cascata (Energia -> Neural leve) -> Só avança se for voz
        |
        v
Wake word ou modo de ativação -> Só roda em blocos com voz
        |
        v
Transcrição incremental (com contexto prévio) -> Otimizada (NPU/Quantização)
        |
        v
Roteador de intenção
        |
        +--> resposta informativa
        |
        +--> comando permitido --> confirmação opcional --> executor
        |
        +--> texto comum --> modo ditado ou descarte
```

## Fases de implementação

### Fase 1 — Captura contínua e estados

Criar um serviço de áudio de longa duração baseado no `AudioRecorder` atual,
mas sem acumular a sessão inteira na memória. Usar blocos de áudio e um buffer
circular com limite configurável.

Estados mínimos:

- `IDLE`: microfone ativo, sem fala detectada;
- `SPEECH`: fala em andamento;
- `PENDING_TRANSCRIPTION`: segmento pronto para transcrição;
- `ACTIVE_COMMAND`: agente ativado e aguardando/compreendendo comando;
- `EXECUTING`: comando autorizado em execução;
- `CONFIRMATION`: aguardando confirmação do usuário;
- `ERROR`: falha de captura, modelo ou executor.

Critérios de engenharia e disponibilidade:

- encerrar corretamente o stream ao sair;
- não criar uma thread por bloco;
- limitar memória e evitar alocações dinâmicas no loop de áudio de alta frequência;
- resiliência de hardware: detectar desconexão de dispositivo (ex: fone Bluetooth) e tentar reconexão automática (backoff exponencial);
- registrar perda de frames e erros do dispositivo;
- permitir pausar e retomar a escuta.

### Fase 2 — Detecção de voz

Adicionar VAD antes da transcrição para evitar processar silêncio. Para aliar **baixo consumo** a **alta qualidade**, recomenda-se uma abordagem em cascata:

1. **VAD de Energia (Custo Zero)**: Descarta silêncio absoluto.
2. **VAD Neural Leve (ex: Silero VAD ou WebRTC)**: Roda apenas nos blocos que passaram pelo filtro de energia, para diferenciar voz humana de ruídos constantes (teclado, ventilador).

O VAD deve ter parâmetros configuráveis:

- limiar de energia e threshold de probabilidade do modelo;
- duração mínima de fala;
- silêncio necessário para finalizar um segmento (cuidado para não cortar fonemas longos);
- duração máxima de um segmento;
- margem de áudio anterior ao início da fala (lookback);
- margem posterior ao fim da fala (trailing pad).

**Pré-processamento**: Avaliar a inclusão de Controle Automático de Ganho (AGC) e Supressão de Ruído básica antes do VAD para melhorar a disponibilidade e precisão em ambientes ruidosos.

Cada segmento deve guardar:

- timestamp de início e fim;
- duração;
- motivo do encerramento;
- nível de ruído;
- se foi transcrito, descartado ou interrompido.

### Fase 3 — Transcrição incremental

Não enviar uma gravação infinita ao Whisper. Dividir a entrada em segmentos
com limite de duração e usar sobreposição para reduzir palavras cortadas.

Estratégia inicial para **alta qualidade e performance**:

- segmentos de 5 a 15 segundos;
- sobreposição de aproximadamente 0,5 a 1 segundo;
- passagem de contexto: usar o texto do segmento anterior como `prompt` inicial (`initial_prompt` no Whisper) para manter consistência semântica e reduzir alucinações;
- uma fila de transcrição com backpressure;
- gestão de recursos: manter uma única instância do modelo reutilizada na memória, preferencialmente com quantização (INT8/FP16) dependendo do hardware;
- fallback de hardware: se a NPU falhar, degradar graciosamente para GPU ou CPU ao invés de derrubar o serviço;
- medição de latência e RTF por segmento;
- descarte ou degradação explícita quando a fila crescer.

Modelos recomendados:

| Cenário | Modelo |
|---|---|
| Menor latência em GPU | Whisper `turbo` |
| NPU equilibrado | Whisper `small` |
| Baixo consumo/fallback | Whisper `tiny` ou `base` |
| Avaliação futura | Qwen3-ASR-0.6B |

O Parakeet atual não deve receber áudio contínuo diretamente: a implementação
usa buckets estáticos até aproximadamente 16 segundos e trunca entradas
maiores. Ele só deve ser usado no agente após a implementação de chunking e
recomposição de texto.

### Fase 4 — Ativação do agente

O agente precisa distinguir fala dirigida a ele de fala ambiente. Avaliar,
nesta ordem:

1. **Push-to-talk**, reutilizando a hotkey atual, como modo seguro inicial;
2. **Modo ativo temporário**, ativado por uma hotkey e encerrado após timeout;
3. **Wake word local** (ex: openWakeWord ou Porcupine), otimizado para CPU. **Importante para baixo consumo**: o detector acústico só deve ser acionado nos blocos de áudio que o VAD confirmar como voz humana;
4. desativação automática após período de silêncio.

O wake word não deve ser confundido com simples transcrição contínua. A
transcrição pode detectar texto parecido com a palavra de ativação, mas isso
não oferece a mesma robustez de um detector acústico dedicado, além de desperdiçar recursos computacionais ativando o modelo maior.

Indicadores visuais e sonoros devem mostrar:

- escutando;
- agente ativo;
- processando;
- executando;
- aguardando confirmação;
- erro ou microfone indisponível.

### Fase 5 — Roteamento de intenções

Separar transcrição de interpretação. Criar uma representação estruturada,
por exemplo:

```json
{
  "intent": "open_application",
  "arguments": {
    "name": "notepad"
  },
  "confidence": 0.94,
  "requires_confirmation": false
}
```

O roteador deve começar com intenções determinísticas e explícitas:

- abrir aplicativo permitido;
- abrir URL permitida;
- pesquisar texto;
- copiar ou colar texto;
- controlar o próprio agente;
- iniciar/parar ditado;
- responder com informação local;

Não executar comandos arbitrários derivados diretamente da transcrição. A
transcrição é entrada não confiável.

### Fase 6 — Executor seguro

Implementar um executor com:

- catálogo de comandos permitidos;
- validação de argumentos;
- separação entre comando e parâmetros;
- timeout;
- cancelamento;
- captura de stdout/stderr;
- código de saída;
- logs estruturados;
- limite de processos simultâneos.

Por padrão, bloquear ou exigir confirmação para:

- apagar ou sobrescrever arquivos;
- executar PowerShell, shell ou scripts;
- instalar pacotes;
- alterar configurações do sistema;
- enviar mensagens;
- mover dinheiro ou realizar ações externas;
- comandos com caminhos não validados;
- qualquer ação fora da allowlist.

Confirmações devem mostrar a ação interpretada, e não apenas repetir a
transcrição:

```text
Vou abrir o Notepad. Confirmar? [Sim] [Não]
```

Nunca usar `Invoke-Expression`, `eval`, concatenação de texto em shell ou
execução direta da saída de um modelo.

### Fase 7 — Memória e contexto

Começar sem memória persistente. O agente deve manter apenas o contexto
necessário para concluir a intenção atual (ex: os últimos comandos ou o buffer de transcrição para passagem de prompt). Essa abordagem minimiza o uso de RAM, evita vazamento de memória a longo prazo e simplifica a recuperação em caso de falha do serviço.

Se memória for necessária posteriormente:

- armazenar somente dados explicitamente autorizados;
- separar histórico de áudio, transcrição e comandos;
- oferecer limpeza pela interface;
- não persistir áudio por padrão;
- proteger arquivos locais e registrar retenção.

## Configuração proposta

Adicionar uma seção de configuração sem quebrar arquivos existentes:

```json
{
  "agent_mode": false,
  "continuous_listening": false,
  "activation_mode": "push_to_talk",
  "wake_word": null,
  "vad_enabled": true,
  "vad_min_speech_seconds": 0.25,
  "vad_end_silence_seconds": 0.8,
  "agent_chunk_seconds": 10,
  "agent_overlap_seconds": 0.75,
  "agent_max_queue": 2,
  "command_confirmation": "always",
  "allowed_commands": []
}
```

Valores seguros devem ser o padrão: `agent_mode` desativado,
`push_to_talk`, sem wake word, confirmação sempre ativa e allowlist vazia.

## Interface e ciclo de vida

Adicionar controles para:

- ativar/desativar agente;
- pausar escuta;
- selecionar microfone;
- selecionar modelo e dispositivo;
- configurar ativação;
- visualizar comando interpretado;
- confirmar ou cancelar;
- visualizar último erro;
- limpar histórico de transcrições.

O desligamento deve:

1. impedir novos segmentos;
2. aguardar ou cancelar a transcrição atual;
3. encerrar a fila;
4. parar o stream;
5. liberar o modelo;
6. remover hotkeys e timers;
7. confirmar no log que o agente terminou.

## Testes e benchmark

### Testes unitários

- transições de estado;
- VAD com silêncio, fala e ruído;
- padding e sobreposição;
- limite da fila;
- cancelamento;
- normalização de transcrição;
- roteamento de intenções;
- validação de argumentos;
- allowlist;
- confirmação;
- timeout e códigos de saída.

### Testes de integração

- captura real com dispositivo selecionado;
- transcrição de segmentos;
- fallback NPU → GPU → CPU;
- perda e reconexão do microfone;
- `DEVICE_LOST`;
- encerramento durante transcrição;
- comando permitido e comando bloqueado.

### Métricas

Medir:

- tempo entre fim da fala e texto parcial;
- tempo entre fim da fala e intenção;
- taxa de falsos acionamentos;
- taxa de comandos não reconhecidos;
- WER em português brasileiro e inglês;
- RTF;
- tamanho e atraso da fila;
- uso de CPU, GPU, NPU e RAM;
- falhas por hora de escuta;
- taxa de execução incorreta, que deve ser zero nos testes de segurança.

Testar fala curta, fala longa, ruído, teclado, televisão, múltiplos locutores
e pausas naturais. Não usar apenas TTS para validar ativação ou comandos.

## Segurança e privacidade

- escuta contínua deve ter indicador visível;
- não salvar áudio por padrão;
- limitar logs para não registrar conteúdo sensível sem opt-in;
- comandos devem ser allowlisted;
- ações destrutivas exigem confirmação;
- wake word e modo agente devem ser desligáveis;
- qualquer erro deve aparecer explicitamente;
- o agente não deve simular sucesso quando a execução falhar.

## Ordem recomendada

1. Corrigir e testar captura contínua com buffer limitado.
2. Adicionar VAD e métricas por segmento.
3. Implementar chunking para Whisper e benchmark long-form.
4. Adicionar modo push-to-talk ativo e indicadores de estado.
5. Implementar intenções determinísticas e executor allowlisted.
6. Adicionar confirmações e testes de segurança.
7. Avaliar wake word local.
8. Avaliar Qwen3-ASR e Parakeet com chunking.
9. Só depois considerar memória, contexto prolongado e comandos mais amplos.

## Critério de prontidão

O modo agente só deve ser habilitado por padrão quando:

- não houver execução arbitrária de texto;
- ações sensíveis exigirem confirmação;
- a escuta puder ser pausada claramente;
- o shutdown for limpo;
- a fila não crescer indefinidamente;
- o benchmark demonstrar latência aceitável;
- a taxa de falsos acionamentos for conhecida;
- comandos bloqueados forem testados automaticamente;
- português brasileiro e inglês tiverem resultados separados.
