#!/usr/bin/env python3
"""CLI para enrollment de alunos.

Modos de uso:

1. Enrollment interativo via webcam (um aluno por vez):
   python scripts/enroll.py --turma ES2025-T1 --ra 12345 --nome "João Silva"

2. Enrollment em lote via CSV com fotos:
   python scripts/enroll.py --turma ES2025-T1 --csv alunos.csv --fotos-dir ./fotos

3. Listar turmas existentes:
   python scripts/enroll.py --list

4. Listar alunos de uma turma:
   python scripts/enroll.py --turma ES2025-T1 --info

5. Remover aluno de uma turma:
   python scripts/enroll.py --turma ES2025-T1 --remove 12345

Formato do CSV (colunas: ra,nome,foto):
   12345,João Silva,joao_silva.jpg
   12346,Maria Santos,maria_santos.jpg
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import FaceConfig
from src.core.models import TurmaEncodings
from src.face.recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enroll")


def enroll_interactive(
    recognizer: FaceRecognizer,
    student_id: str,
    student_name: str,
    num_samples: int = 5,
    camera_index: int = 0,
) -> bool:
    """Enrollment interativo com preview da webcam.

    Mostra uma janela com o feed da câmera. O operador pressiona
    ESPAÇO para capturar cada sample e Q para cancelar.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        logger.error("Não foi possível abrir a câmera %d", camera_index)
        return False

    frames: list = []
    print(f"\n{'='*60}")
    print(f"  Enrollment: {student_name} ({student_id})")
    print(f"  Capturas necessárias: {num_samples}")
    print(f"  ESPAÇO = capturar  |  Q = cancelar")
    print(f"{'='*60}\n")

    try:
        while len(frames) < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            # Overlay com instruções
            display = frame.copy()
            cv2.putText(
                display,
                f"Capturas: {len(frames)}/{num_samples} | ESPACO=capturar Q=sair",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display,
                f"Aluno: {student_name} ({student_id})",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nEnrollment cancelado pelo operador.")
                return False
            elif key == ord(" "):
                frames.append(frame)
                print(f"  Captura {len(frames)}/{num_samples} OK")
                # Flash visual
                white = frame.copy()
                white[:] = (255, 255, 255)
                cv2.imshow("Enrollment", white)
                cv2.waitKey(100)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Processar enrollment
    result = recognizer.enroll_from_frames(student_id, student_name, frames)

    if result.success:
        print(f"\n  ✓ {result.message}")
    else:
        print(f"\n  ✗ {result.message}")

    return result.success


def enroll_from_csv(
    recognizer: FaceRecognizer,
    csv_path: Path,
    fotos_dir: Path,
    num_jitters: int = 3,
) -> dict[str, bool]:
    """Enrollment em lote a partir de CSV com fotos.

    CSV esperado (com header): ra,nome,foto
    As fotos devem estar no diretório `fotos_dir`.
    """
    results: dict[str, bool] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ra = row["ra"].strip()
            nome = row["nome"].strip()
            foto_filename = row["foto"].strip()
            foto_path = fotos_dir / foto_filename

            if not foto_path.exists():
                logger.warning("Foto não encontrada: %s", foto_path)
                results[ra] = False
                continue

            # Carregar foto
            frame = cv2.imread(str(foto_path))
            if frame is None:
                logger.warning("Não foi possível ler: %s", foto_path)
                results[ra] = False
                continue

            # Usar a mesma foto com jitter para gerar variações
            # (face_recognition gera perturbações aleatórias)
            old_jitters = recognizer.config.num_jitters
            recognizer.config.num_jitters = num_jitters

            result = recognizer.enroll_from_frames(ra, nome, [frame] * 3)
            recognizer.config.num_jitters = old_jitters

            results[ra] = result.success
            status = "✓" if result.success else "✗"
            print(f"  {status} {nome} ({ra}): {result.message}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrollment de alunos para reconhecimento facial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--turma", help="ID da turma (ex: ES2025-T1)")
    parser.add_argument("--ra", help="RA do aluno (enrollment individual)")
    parser.add_argument("--nome", help="Nome do aluno (enrollment individual)")
    parser.add_argument("--csv", help="Caminho do CSV (enrollment em lote)", type=Path)
    parser.add_argument("--fotos-dir", help="Diretório com fotos", type=Path)
    parser.add_argument("--camera", help="Índice da câmera", type=int, default=0)
    parser.add_argument("--samples", help="Capturas por aluno", type=int, default=5)
    parser.add_argument("--list", action="store_true", help="Listar turmas")
    parser.add_argument("--info", action="store_true", help="Info da turma")
    parser.add_argument("--remove", help="Remover aluno (RA) da turma")
    parser.add_argument(
        "--encodings-dir",
        help="Diretório dos encodings",
        type=Path,
        default=Path("src/face/encodings"),
    )
    parser.add_argument(
        "--threshold",
        help="Threshold de match (default: 0.45)",
        type=float,
        default=0.45,
    )

    args = parser.parse_args()

    config = FaceConfig(
        encodings_dir=args.encodings_dir,
        match_threshold=args.threshold,
        camera_index=args.camera,
        samples_per_student=args.samples,
    )
    recognizer = FaceRecognizer(config)

    # ── Listar turmas ──
    if args.list:
        turmas = recognizer.list_turmas()
        if turmas:
            print("Turmas cadastradas:")
            for t in sorted(turmas):
                recognizer.load_turma(t)
                print(f"  {t}: {recognizer.turma.student_count} alunos")
        else:
            print("Nenhuma turma cadastrada.")
        return

    if not args.turma:
        parser.error("--turma é obrigatório (exceto com --list)")

    # Carregar ou criar turma
    try:
        recognizer.load_turma(args.turma)
        print(f"Turma '{args.turma}' carregada ({recognizer.turma.student_count} alunos)")
    except FileNotFoundError:
        recognizer.create_turma(args.turma)
        print(f"Turma '{args.turma}' criada (nova)")

    # ── Info da turma ──
    if args.info:
        if recognizer.turma.student_count == 0:
            print("  (vazia)")
        else:
            for sid, student in sorted(recognizer.turma.students.items()):
                n_enc = len(student.encodings)
                print(f"  {sid}: {student.student_name} ({n_enc} samples)")
        return

    # ── Remover aluno ──
    if args.remove:
        if recognizer.turma.remove_student(args.remove):
            recognizer.save_turma()
            print(f"Aluno {args.remove} removido da turma '{args.turma}'.")
        else:
            print(f"Aluno {args.remove} não encontrado na turma '{args.turma}'.")
        return

    # ── Enrollment em lote via CSV ──
    if args.csv:
        if not args.fotos_dir:
            parser.error("--fotos-dir é obrigatório com --csv")

        print(f"\nEnrollment em lote: {args.csv}")
        results = enroll_from_csv(recognizer, args.csv, args.fotos_dir)
        recognizer.save_turma()

        ok = sum(1 for v in results.values() if v)
        print(f"\nResultado: {ok}/{len(results)} alunos cadastrados com sucesso.")
        return

    # ── Enrollment individual via webcam ──
    if not args.ra or not args.nome:
        parser.error("--ra e --nome são obrigatórios para enrollment individual")

    success = enroll_interactive(
        recognizer,
        student_id=args.ra,
        student_name=args.nome,
        num_samples=args.samples,
        camera_index=args.camera,
    )

    if success:
        recognizer.save_turma()
        print(f"\nTurma '{args.turma}' salva com sucesso.")


if __name__ == "__main__":
    main()
