from pathlib import Path

from deepface import DeepFace

from reconhecimento_facial.config import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    MODEL_NAME,
)


class FaceRecognizer:
    """
    Responsável pelo reconhecimento facial.

    A classe compara uma imagem de teste com a base
    de pessoas autorizadas.
    """

    def __init__(
        self,
        authorized_path: Path = AUTHORIZED_DIR,
        model_name: str = MODEL_NAME,
        detector_backend: str = DETECTOR_BACKEND,
        distance_metric: str = DISTANCE_METRIC,
    ):
        self.authorized_path = Path(authorized_path)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric

        self._validate_directories()

    def _validate_directories(self):
        """
        Verifica se o diretório de pessoas autorizadas existe.
        """

        if not self.authorized_path.exists():
            raise FileNotFoundError(
                f"Diretório de autorizados não encontrado: "
                f"{self.authorized_path}"
            )

        if not self.authorized_path.is_dir():
            raise NotADirectoryError(
                f"O caminho de autorizados não é um diretório: "
                f"{self.authorized_path}"
            )

    def recognize(self, test_image: Path) -> dict:
        """
        Compara uma imagem de teste com a base de autorizados.

        Retorna um dicionário contendo:

        - authorized
        - person
        - distance
        - identity
        - threshold
        """

        test_image = Path(test_image)

        if not test_image.exists():
            raise FileNotFoundError(
                f"Imagem de teste não encontrada: {test_image}"
            )

        print(f"\nAnalisando: {test_image.name}")

        try:

            results = DeepFace.find(
                img_path=str(test_image),
                db_path=str(self.authorized_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=True,
                silent=True,
            )

        except ValueError as error:

            print(f"Não foi possível detectar um rosto: {error}")

            return self._denied_result(
                reason="Nenhum rosto detectado."
            )

        except Exception as error:

            print(
                f"Erro durante o reconhecimento:\n"
                f"{type(error).__name__}: {error}"
            )

            return self._denied_result(
                reason="Erro durante o reconhecimento."
            )

        if not results:

            return self._denied_result(
                reason="Nenhum resultado encontrado."
            )

        dataframe = results[0]

        if dataframe.empty:

            return self._denied_result(
                reason="Nenhuma pessoa correspondente encontrada."
            )

        # Melhor correspondência
        best_match = dataframe.iloc[0]

        identity = Path(best_match["identity"])

        distance = float(best_match["distance"])

        # O DeepFace pode fornecer o threshold utilizado.
        threshold = None

        if "threshold" in dataframe.columns:
            threshold = float(best_match["threshold"])

        # Nome da pessoa
        person_name = self._get_person_name(identity)

        # Verifica se a distância está dentro do threshold
        authorized = True

        if threshold is not None:
            authorized = distance <= threshold

        return {
            "authorized": authorized,
            "person": person_name if authorized else None,
            "distance": distance,
            "threshold": threshold,
            "identity": str(identity),
            "reason": (
                "Correspondência encontrada."
                if authorized
                else "Distância acima do threshold."
            ),
        }

    @staticmethod
    def _get_person_name(identity: Path) -> str:
        """
        Obtém o nome da pessoa a partir da estrutura:

        data/interim/
            └── pessoa/
                └── foto.jpg
        """

        return identity.parent.name

    @staticmethod
    def _denied_result(reason: str) -> dict:
        """
        Cria um resultado padrão para acesso negado.
        """

        return {
            "authorized": False,
            "person": None,
            "distance": None,
            "threshold": None,
            "identity": None,
            "reason": reason,
        }