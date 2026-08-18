# Configurar uma NUC do zero

Passo a passo para levar uma Intel NUC sem nada instalado até rodar `proctor.service`
como estação de prova. Assume-se que o dashboard do professor já existe em outro
lugar (hoje: EC2 separado) — este documento cobre só o papel de **estação**, não o
dashboard.

Cada passo indica se é **manual** (feito fora do repositório) ou **script**
(comando dentro do repo clonado).

## 0. Pré-requisitos de hardware

- Intel NUC (ou equivalente) com pelo menos 4 núcleos, 16 GB RAM recomendado.
- Webcam UVC (Logitech C920/C922 ou compatível), com microfone embutido.
- Conexão de rede até o dashboard (porta configurada — ver `PROCTOR_DASHBOARD_BASE_URL`).
- Acesso a um bucket S3 com fotos de cadastro em `fotos/{turma_id}/{aluno}.png` e
  credenciais AWS com permissão de leitura/escrita nesse bucket.

## 1. Instalar o sistema operacional (manual)

- Grave um ISO do **Ubuntu 24.04 Desktop** (não Server — o projeto depende de
  sessão gráfica GNOME) num pendrive e instale normalmente.
- Crie o usuário operacional `proctor`.
- **Confirme a sessão em X11, não Wayland**: a captura de tela usa `x11grab`, que
  não funciona em Wayland. Na tela de login do GNOME, antes de digitar a senha,
  clique no ícone de engrenagem e escolha **"Ubuntu on Xorg"**. Depois de logado,
  confirme com:
  ```bash
  echo $XDG_SESSION_TYPE   # deve imprimir "x11"
  ```
- Ainda **não precisa conectar a webcam definitiva** nesta etapa — os passos 2 a 5
  funcionam sem ela. Conecte a webcam de produção só no passo 6.

## 2. Clonar o repositório (manual)

```bash
git clone <url-do-repo> ~/proctor-station
cd ~/proctor-station
```

## 3. Rodar o bootstrap (script)

```bash
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
```

O que ele faz, em ordem: instala pacotes apt (build tools, Python 3.12, FFmpeg,
v4l-utils, alsa-utils, Chromium, ferramentas de lockdown X11), instala as policies
de hardening do Chromium, confere se a sessão é X11 (avisa se não for), cria o
venv, instala as dependências Python (`dlib` compila do source — 3 a 5 min), baixa
os modelos dlib (~100 MB), cria o `.env` a partir do `.env.example` **se ainda não
existir**, roda a suíte de testes e testa a câmera em modo headless (se já houver
alguma `/dev/video*` disponível nesse momento).

Ele **não** detecta a webcam definitiva nem preenche credenciais — isso é
proposital, ver passos 4 e 6.

## 4. Preencher o `.env` à mão (manual)

Abra `.env` (criado pelo bootstrap) e preencha o que ficou em branco. Cada campo
em branco no `.env.example` já tem um comentário explicando o que colocar, mas em
resumo, para uma **estação** (NUC):

```dotenv
# Credenciais AWS — acesso ao bucket S3 de fotos/gravações
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Identificação ÚNICA desta estação — nunca repita entre NUCs
PROCTOR_DASHBOARD_STATION_ID=nuc-01
PROCTOR_DASHBOARD_STATION_NAME=Estação 1

# Aponta para o dashboard real (EC2), não para 127.0.0.1 — use https:// se o
# dashboard estiver atrás do nginx/certbot (ver docs/setup_dashboard.md),
# nunca http:// puro em produção (token de estação viajaria em texto claro).
PROCTOR_DASHBOARD_ENABLED=true
PROCTOR_DASHBOARD_BASE_URL=https://<host-do-dashboard>

# Token desta estação — NÃO é a senha do professor. Emitido no dashboard com
# scripts/issue_station_token.py <station_id>; copie o valor impresso lá.
# Sem isso (ou com o token errado), o heartbeat recebe 401 e a estação nunca
# aparece no painel — sem erro na UI, só no log (journalctl -u proctor -f).
PROCTOR_DASHBOARD_STATION_TOKEN=

# Só se a prova precisar sair por um proxy fixo — deixe vazio se não precisar
PROCTOR_APP_PROXY_SERVER=
```

Confira também `PROCTOR_REC_FFMPEG_CPU_CORES` / `PROCTOR_REC_PROCTOR_CPU_CORES`:
deixe em branco a menos que você tenha confirmado o número de núcleos desta
máquina específica com `nproc` (ver comentário no `.env.example`).

## 5. Encodings da turma (script)

A estação precisa do `.pkl` de encodings da turma que vai fazer a prova em
`data/encodings/{turma_id}.pkl`. Gere localmente a partir das fotos no S3:

```bash
source venv/bin/activate
python scripts/enroll.py --turma <turma_id>
```

Alternativa: copie um `.pkl` já gerado em outra máquina para
`data/encodings/{turma_id}.pkl` desta NUC — o formato é local e portátil.

## 6. Conectar a webcam definitiva e detectar (script)

Só agora conecte a webcam de produção na porta USB que ela vai usar durante a
prova (evite trocar de porta depois — índice `/dev/videoN` e card ALSA podem
mudar). Depois:

```bash
bash scripts/detect_camera_audio.sh
```

Ele detecta o `/dev/videoN` que fala MJPG (o nó de captura de imagem de verdade —
webcams UVC costumam expor mais de um `/dev/videoN`) e o card ALSA de áudio da
webcam, e ajusta `PROCTOR_FACE_CAMERA_INDEX` / `PROCTOR_REC_WEBCAM_AUDIO_DEVICE`
no `.env`. Se ele não achar nada automaticamente, ele avisa e mostra o comando
manual (`v4l2-ctl --list-devices`, `arecord -l`) — confira e ajuste à mão nesse
caso.

Rode este script de novo sempre que a webcam for trocada ou reconectada numa
porta USB diferente.

## 7. Instalar o serviço systemd da API local (script)

```bash
cd ~/proctor-station   # o script resolve import a partir do cwd — sempre cd antes
sudo bash scripts/install_systemd_service.sh
```

Isso cria e habilita `proctor.service` (porta `8000`, `--host 0.0.0.0`),
apontando para o Python do venv e o diretório do projeto atual. Reiniciar o
serviço depois de mexer no código é o que recarrega:

```bash
sudo systemctl restart proctor
```

> `proctor-dashboard.service` (`scripts/install_dashboard_service.sh`) é **só
> para a máquina que roda o dashboard** (hoje o EC2) — não instale isso nas
> estações.

## 8. Verificação (script)

```bash
systemctl status proctor --no-pager
curl -sS http://127.0.0.1:8000/health      # espera status=ok, camera_ok=true, s3_ok=true
curl -sS http://127.0.0.1:8000/status

v4l2-ctl --list-devices                    # webcam presente?
arecord -l                                 # mic da webcam no card certo?
fuser /dev/video0                          # livre fora de prova
```

Se `camera_ok=false`: confira se a webcam está conectada e se `detect_camera_audio.sh`
achou o dispositivo certo. Se `s3_ok=false`: confira as credenciais AWS no `.env` e
se o bucket/região batem com `PROCTOR_S3_BUCKET` / `PROCTOR_S3_REGION`.

Com o dashboard acessível, confirme que esta estação aparece lá com o
`station_id`/`station_name` corretos — se aparecer com o nome de outra estação,
alguma NUC está com `PROCTOR_DASHBOARD_STATION_ID` repetido.

## 9. Testar uma prova de ponta a ponta (manual)

Com alguém sentado na frente da webcam:

```bash
curl -sS -X POST http://127.0.0.1:8000/exam-mode/prepare
curl -sS -X POST http://127.0.0.1:8000/exam-mode/enter
```

Isso já aplica o lockdown do GNOME e trava o desktop de verdade (Alt+Tab, Super,
dock). A recuperação sempre funciona pelo shell, mesmo com a GUI travada:

```bash
bash scripts/recover_exam_mode.sh
```

## Problemas conhecidos que vão custar tempo

Antes de estranhar algo, confira `~/Desktop/CONTEXT.md` (seção "Armadilhas
Conhecidas") e `~/Desktop/resume.txt` — cobrem, entre outras coisas: `pkill -f`/
`ps | grep` dando falso positivo consigo mesmos, `PREVIEW_OPEN_TIMEOUT_MS` não
poder ser reduzido sem medir contra o FFmpeg real, o `config.json` da extensão
allowlist sendo artefato versionado reescrito em runtime, e o pacote
`chromium-browser` do apt sendo só um invólucro do snap no Ubuntu 24.04 (congelar
versão de verdade é `snap refresh --hold chromium`, não `apt-mark hold`).
