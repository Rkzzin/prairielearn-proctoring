"""Política de teardown dos componentes de uma sessão.

Quando uma sessão acaba — normalmente, por falha no start, ou por recuperação
manual — parte do ambiente pode ter que **sobreviver**. Se a estação continua
em modo prova esperando o próximo aluno, derrubar lockdown, overlay de espera e
câmera para reconstruí-los em seguida causa flicker no overlay, liga/desliga da
webcam e, pior, reabre por um instante a janela de fuga do GNOME.

Essa decisão era três booleanos calculados à mão dentro do ``except`` de
``start_session`` — o trecho mais fácil de errar do módulo, e testado apenas de
forma indireta. Aqui a regra tem nome, um lugar só e teste próprio.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.states import StationMode

#: ``reason`` usado quando o stop vem da saída do modo prova. Nesse caso o
#: lockdown **não** sobrevive: o operador está voltando para manutenção.
EXIT_EXAM_MODE_REASON = "exit_exam_mode"


@dataclass(frozen=True)
class ShutdownPolicy:
    """O que sobrevive ao encerramento dos componentes.

    Por padrão nada sobrevive — o caminho seguro é derrubar tudo e restaurar o
    GNOME.
    """

    keep_lockdown: bool = False
    keep_waiting_overlay: bool = False
    keep_camera: bool = False

    @classmethod
    def full_teardown(cls) -> ShutdownPolicy:
        """Derruba tudo: saída do modo prova e recuperação manual."""
        return cls()

    @classmethod
    def for_failed_start(cls, *, mode: StationMode, session_started: bool) -> ShutdownPolicy:
        """Rollback de um ``start_session`` que levantou exceção.

        Em ``WAITING_STUDENT`` o lockdown **sempre** fica: a estação segue em
        modo prova, e uma identificação que falhou não pode ser motivo para
        devolver o GNOME destravado ao aluno sentado na frente dela.

        Overlay de espera e câmera só ficam se a sessão nem chegou a existir —
        o caso comum de "ainda não identifiquei ninguém". É o que evita o
        liga/desliga da webcam e o flicker do overlay a cada tentativa de
        auto-start.
        """
        waiting = mode == StationMode.WAITING_STUDENT
        keep_waiting_resources = waiting and not session_started
        return cls(
            keep_lockdown=waiting,
            keep_waiting_overlay=keep_waiting_resources,
            keep_camera=keep_waiting_resources,
        )

    @classmethod
    def for_stopped_session(
        cls,
        *,
        mode: StationMode,
        auto_start: bool,
        reason: str,
        exam_mode_active: bool = False,
    ) -> ShutdownPolicy:
        """Encerramento de uma sessão que estava ativa.

        Enquanto a estação estiver em modo prova, qualquer encerramento que não
        seja a saída explícita desse modo volta a esperar aluno. Isso mantém o
        desktop coberto durante a troca de Chromium, inclusive quando o
        operador encerra a prova manualmente.

        ``keep_lockdown`` é, por construção, o mesmo predicado de "vai voltar
        para ``WAITING_STUDENT``" — quem chama usa esse campo para as duas
        decisões, de propósito, para não haver dois predicados a divergir.
        """
        keep = mode == StationMode.SESSION and exam_mode_active and reason != EXIT_EXAM_MODE_REASON
        return cls(keep_lockdown=keep, keep_waiting_overlay=keep)
