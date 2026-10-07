from pathlib import Path

import cv2

from reconhecimento_facial.config import (
    IMAGE_EXTENSIONS,
    TEST_DIR,
)
from reconhecimento_facial.face_recognition import FaceRecognizer


def get_test_images(directory: Path) -> list[Path]:
    """
    Retorna todas as imagens válidas do diretório de testes.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Diretório de testes não encontrado: {directory}"
        )

    images = [
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


def show_result(
    image_path: Path,
    result: dict,
):
    """
    Exibe a imagem e o resultado do reconhecimento.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"Não foi possível abrir a imagem: "
            f"{image_path}"
        )
        return

    authorized = result["authorized"]
    person = result["person"]
    distance = result["distance"]
    threshold = result["threshold"]

    # ========================================================
    # RESULTADO
    # ========================================================

    if authorized:

        message = "ACESSO PERMITIDO"

        print("\n" + "=" * 60)
        print("ACESSO PERMITIDO")
        print(f"Pessoa: {person}")

        if distance is not None:
            print(f"Distância: {distance:.4f}")

        if threshold is not None:
            print(f"Threshold: {threshold:.4f}")

        print("=" * 60)

    else:

        message = "ACESSO NEGADO"

        print("\n" + "=" * 60)
        print("ACESSO NEGADO")
        print(f"Motivo: {result['reason']}")

        if distance is not None:
            print(f"Distância: {distance:.4f}")

        if threshold is not None:
            print(f"Threshold: {threshold:.4f}")

        print("=" * 60)

    # ========================================================
    # TEXTO NA IMAGEM
    # ========================================================

    color = (0, 255, 0) if authorized else (0, 0, 255)

    cv2.putText(
        image,
        message,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    if authorized and person:

        cv2.putText(
            image,
            f"Pessoa: {person}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # EXIBIR
    # ========================================================

    cv2.imshow(
        "Sistema de Reconhecimento Facial",
        image,
    )

    print("\nPressione qualquer tecla para continuar.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():

    print("=" * 60)
    print("      SISTEMA DE RECONHECIMENTO FACIAL")
    print("=" * 60)

    print(f"\nDiretório de testes:")
    print(TEST_DIR)

    # ========================================================
    # RECONHECEDOR
    # ========================================================

    recognizer = FaceRecognizer()

    # ========================================================
    # IMAGENS DE TESTE
    # ========================================================

    try:

        test_images = get_test_images(TEST_DIR)

    except FileNotFoundError as error:

        print(f"\nErro: {error}")
        return

    if not test_images:

        print(
            "\nNenhuma imagem encontrada "
            "no diretório de testes."
        )

        return

    print(
        f"\nQuantidade de imagens encontradas: "
        f"{len(test_images)}"
    )

    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    for image_path in test_images:

        result = recognizer.recognize(image_path)

        show_result(
            image_path,
            result,
        )

    print("\nProcessamento finalizado.")


if __name__ == "__main__":
    main()