#!/usr/bin/env python3
"""CLI de enrollment — lê fotos do S3 e gera .pkl local.

O fluxo esperado é:
  1. Alunos sobem suas fotos para o S3 nas primeiras semanas de aula.
       s3://{bucket}/fotos/{turma_id}/{nome_do_aluno}.png
  2. Após o prazo, o operador roda este script para gerar o .pkl.
  3. O .pkl fica salvo localmente na NUC para uso durante as provas.

Uso:

  Enrollment completo de uma turma (baixa todas as fotos do S3):
    python scripts/enroll.py --turma ES2025-T1

  Enrollment de um único aluno (foto já está no S3):
    python scripts/enroll.py --turma ES2025-T1 --aluno joao_silva

  Reprocessar turma do zero (sobrescreve .pkl existente):
    python scripts/enroll.py --turma ES2025-T1 --force

  Listar turmas com .pkl gerado:
    python scripts/enroll.py --list

  Ver alunos cadastrados em uma turma:
    python scripts/enroll.py --turma ES2025-T1 --info

  Remover aluno do .pkl (não remove a foto do S3):
    python scripts/enroll.py --turma ES2025-T1 --remove joao_silva

Variáveis de ambiente relevantes:
  PROCTOR_S3_BUCKET          Nome do bucket (default: proctor-station)
  PROCTOR_S3_REGION          Região AWS (default: us-east-1)
  PROCTOR_S3_PHOTOS_PREFIX   Prefixo das fotos (default: fotos)
  PROCTOR_FACE_MODELS_DIR    Diretório dos modelos dlib (default: models)
  PROCTOR_FACE_ENCODINGS_DIR Diretório dos .pkl (default: data/encodings)
  PROCTOR_FACE_NUM_JITTERS   Jitters por foto (default: 1; use 3-5 para melhor precisão)

AWS credentials devem estar configuradas via ~/.aws/credentials ou variáveis de ambiente.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import FaceConfig, S3Config
from src.core.s3_client import get_s3_client
from src.face.recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enroll")


def _build_recognizer(args: argparse.Namespace) -> FaceRecognizer:
    config = FaceConfig(
        encodings_dir=args.encodings_dir,
        match_threshold=args.threshold,
        num_jitters=args.jitters,
    )
    return FaceRecognizer(config)


def _build_s3(args: argparse.Namespace) -> object:
    s3_config = S3Config(
        bucket=args.bucket,
        photos_prefix=args.photos_prefix,
    )
    return get_s3_client(config=s3_config)


def cmd_enroll_turma(args: argparse.Namespace) -> None:
    """Baixa todas as fotos da turma do S3 e gera o .pkl."""
    recognizer = _build_recognizer(args)
    s3 = _build_s3(args)

    # Verificar se já existe .pkl e se --force não foi passado
    encodings_dir = Path(args.encodings_dir)
    pkl_path = encodings_dir / f"{args.turma}.pkl"
    if pkl_path.exists() and not args.force:
        print(
            f"  .pkl já existe para '{args.turma}'. "
            f"Use --force para reprocessar do zero."
        )
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Enrollment: turma {args.turma}")
    print(f"  Bucket: s3://{args.bucket}/{args.photos_prefix}/{args.turma}/")
    print(f"  Jitters por foto: {args.jitters}")
    print(f"{'='*60}\n")

    # Baixar fotos do S3
    def on_progress(current: int, total: int, name: str) -> None:
        print(f"  [{current:>3}/{total}] Baixando: {name}")

    try:
        photos = s3.download_all_photos(args.turma, on_progress=on_progress)
    except ValueError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

    print(f"\n  {len(photos)} foto(s) baixada(s). Processando encodings...\n")

    # Criar turma e processar cada foto
    recognizer.create_turma(args.turma)
    ok = 0
    failed: list[str] = []

    for photo in photos:
        # Repetir o frame para gerar variações via dlib jitter interno
        frames = [photo.image] * max(args.jitters, 3)

        # student_id = nome do arquivo (stem), student_name = mesmo valor
        # (sem mapeamento RA↔nome aqui; isso virá da API/DB futuramente)
        result = recognizer.enroll_from_frames(
            student_id=photo.student_name,
            student_name=photo.student_name,
            frames=frames,
        )

        status = "✓" if result.success else "✗"
        print(f"  {status} {photo.student_name}: {result.message}")

        if result.success:
            ok += 1
        else:
            failed.append(photo.student_name)

    print(f"\n  Resultado: {ok}/{len(photos)} alunos cadastrados.")

    if failed:
        print(f"\n  Falhas ({len(failed)}):")
        for name in failed:
            print(f"    - {name}")
        print(
            "\n  Dica: verifique se as fotos têm exatamente 1 rosto visível "
            "e boa iluminação."
        )

    if ok > 0:
        path = recognizer.save_turma()
        print(f"\n  .pkl salvo em: {path}")
    else:
        print("\n  Nenhum aluno cadastrado — .pkl não gerado.")
        sys.exit(1)


def cmd_enroll_aluno(args: argparse.Namespace) -> None:
    """Baixa a foto de um único aluno do S3 e adiciona ao .pkl existente."""
    recognizer = _build_recognizer(args)
    s3 = _build_s3(args)

    # Carregar turma existente ou criar nova
    try:
        recognizer.load_turma(args.turma)
        print(f"  Turma '{args.turma}' carregada ({recognizer.turma.student_count} alunos)")
    except FileNotFoundError:
        recognizer.create_turma(args.turma)
        print(f"  Turma '{args.turma}' criada (nova)")

    # Verificar se foto existe no S3
    if not s3.photo_exists(args.turma, args.aluno):
        print(
            f"\n  ✗ Foto não encontrada no S3: "
            f"s3://{args.bucket}/{args.photos_prefix}/{args.turma}/{args.aluno}.png"
        )
        sys.exit(1)

    # Baixar e processar
    prefix = s3.config.photos_prefix_for_turma(args.turma)
    # Tentar extensões em ordem
    key = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = f"{prefix}{args.aluno}{ext}"
        if s3.photo_exists(args.turma, args.aluno):
            key = candidate
            break

    photo = s3.download_student_photo(key)
    frames = [photo.image] * max(args.jitters, 3)

    result = recognizer.enroll_from_frames(
        student_id=photo.student_name,
        student_name=photo.student_name,
        frames=frames,
    )

    if result.success:
        path = recognizer.save_turma()
        print(f"  ✓ {result.message}")
        print(f"  .pkl atualizado: {path}")
    else:
        print(f"  ✗ {result.message}")
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    recognizer = _build_recognizer(args)
    turmas = recognizer.list_turmas()
    if turmas:
        print("Turmas com .pkl gerado:")
        for t in sorted(turmas):
            try:
                recognizer.load_turma(t)
                print(f"  {t}: {recognizer.turma.student_count} alunos")
            except Exception:
                print(f"  {t}: (erro ao carregar)")
    else:
        print("Nenhuma turma cadastrada.")


def cmd_info(args: argparse.Namespace) -> None:
    recognizer = _build_recognizer(args)
    try:
        recognizer.load_turma(args.turma)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    print(f"Turma '{args.turma}' — {recognizer.turma.student_count} aluno(s):\n")
    if recognizer.turma.student_count == 0:
        print("  (vazia)")
    else:
        for sid, student in sorted(recognizer.turma.students.items()):
            print(f"  {sid}: {student.student_name} ({len(student.encodings)} samples)")


def cmd_remove(args: argparse.Namespace) -> None:
    recognizer = _build_recognizer(args)
    try:
        recognizer.load_turma(args.turma)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    if recognizer.turma.remove_student(args.remove):
        recognizer.save_turma()
        print(f"  ✓ Aluno '{args.remove}' removido da turma '{args.turma}'.")
        print("  Nota: a foto no S3 não foi removida.")
    else:
        print(f"  ✗ Aluno '{args.remove}' não encontrado na turma '{args.turma}'.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrollment de alunos via fotos no S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Opções de ação (mutuamente exclusivas)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="Listar turmas com .pkl gerado")
    action.add_argument("--info", action="store_true", help="Ver alunos cadastrados na turma")
    action.add_argument("--remove", metavar="NOME_ALUNO", help="Remover aluno do .pkl")

    # Alvos
    parser.add_argument("--turma", metavar="TURMA_ID", help="ID da turma (ex: ES2025-T1)")
    parser.add_argument(
        "--aluno",
        metavar="NOME_ALUNO",
        help="Stem do arquivo no S3 para enrollment individual (ex: joao_silva)",
    )

    # S3
    parser.add_argument("--bucket", default="proctor-station", help="Nome do bucket S3")
    parser.add_argument(
        "--photos-prefix",
        default="fotos",
        dest="photos_prefix",
        help="Prefixo S3 das fotos (default: fotos)",
    )

    # Paths locais
    parser.add_argument(
        "--encodings-dir",
        type=Path,
        default=Path("data/encodings"),
        dest="encodings_dir",
        help="Diretório local dos .pkl (default: data/encodings)",
    )

    # Parâmetros de encoding
    parser.add_argument(
        "--jitters",
        type=int,
        default=3,
        help="Jitters dlib por foto (mais = mais preciso, mais lento; default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Threshold de distância para match (default: 0.45)",
    )
    parser.add_argument("--force", action="store_true", help="Sobrescrever .pkl existente")

    args = parser.parse_args()

    # Roteamento de comandos
    if args.list:
        cmd_list(args)
        return

    if not args.turma:
        parser.error("--turma é obrigatório (exceto com --list)")

    if args.info:
        cmd_info(args)
    elif args.remove:
        cmd_remove(args)
    elif args.aluno:
        cmd_enroll_aluno(args)
    else:
        # Enrollment completo da turma
        cmd_enroll_turma(args)


if __name__ == "__main__":
    main()