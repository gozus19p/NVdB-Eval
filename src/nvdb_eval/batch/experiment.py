import argparse
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, override
from urllib.parse import quote

import structlog
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from tqdm import tqdm

from model.lemma import Lemma
from model.sensocomune import SensoComune


class SenseExtraction(BaseModel):
    pos: str
    definition: str
    examples: list[str]
    nota: str | None = None


class DefinitionGeneration(BaseModel):
    lemma: str
    senses: list[SenseExtraction]


class GroundTruthVsPrediction(BaseModel):
    ground_truth: int | None
    prediction: int | None


class Judgement(BaseModel):
    true_positive: list[GroundTruthVsPrediction]
    false_positive: list[GroundTruthVsPrediction]
    false_negative: list[GroundTruthVsPrediction]


class Prompt:
    __system_prompt: str
    __user_prompt_template: str

    def __init__(
        self, system_prompt_path: Path, user_prompt_template_path: Path
    ) -> None:
        """Costruttore del `Prompt`.

        Args:
            system_prompt_path (Path): Il `Path` che punta al system prompt
            user_prompt_template_path (Path): Il `Path` che punta al user prompt template
        """
        self.__system_prompt = system_prompt_path.read_text()
        self.__user_prompt_template = user_prompt_template_path.read_text()

    def get_system_prompt(self) -> str:
        """Restituisce il system prompt.

        Returns:
            str: Il system prompt
        """
        return self.__system_prompt

    def prepare_user_prompt(self, **kwargs) -> str:
        """Prepara il prompt user da eseguire in funzione del contesto fornito, rappresentato dai `**kwargs`.

        Returns:
            str: Il prompt "user" pronto da usare
        """
        rendered_prompt: str = self.__user_prompt_template
        for key, value in kwargs.items():
            rendered_prompt = rendered_prompt.replace(f"{{{{{key}}}}}", str(value))
            rendered_prompt = rendered_prompt.replace(f"{{{key}}}", str(value))
        return rendered_prompt.replace("{{", "{").replace("}}", "}")


class PromptType(Enum):
    """Enumeratore che rappresenta la tipologia di prompt da eseguire. Il campo `Enum.value` è l'oggetto `Prompt`.
    Il costrutto dell'enumeratore viene utilizzato come proxy del pattern singleton.

    Args:
        Enum (Prompt): Enumeratore per il prompt
    """

    SENSE_EXTRACTION = Prompt(
        system_prompt_path=Path(__file__).parent
        / "prompt"
        / "definition_it_simple-system.md",
        user_prompt_template_path=Path(__file__).parent
        / "prompt"
        / "definition_it_simple.md",
    )
    JUDGEMENT_ALIGNMENT = Prompt(
        system_prompt_path=Path(__file__).parent
        / "prompt"
        / "judgement-alignment-system.md",
        user_prompt_template_path=Path(__file__).parent
        / "prompt"
        / "judgement-alignment-user.md",
    )
    DEFINITION_V2 = Prompt(
        system_prompt_path=Path(__file__).parent
        / "prompt"
        / "definition_it_simple-system.md",
        user_prompt_template_path=Path(__file__).parent
        / "prompt"
        / "definition_it_simple_poly.md",
    )

    def get_system_prompt(self) -> str:
        return self.value.get_system_prompt()

    def prepare_user_prompt_template(self, **kwargs) -> str:
        return self.value.prepare_user_prompt(**kwargs)


class Batch(ABC):

    __openrouter_model_name: str
    __openrouter_api_key: str
    __logger: structlog.BoundLogger
    __cache: dict[str, Any]
    __cache_dir: Path
    __enable_multithreading: bool
    __max_workers: int
    __perimeter_path: Path
    __error_log_path: Path

    def __init__(
        self,
        openrouter_model_name: str,
        openrouter_api_key: str,
        enable_multithreading: bool = False,
        max_workers: int = 8,
        perimeter_path: Path | None = None,
    ) -> None:
        if not openrouter_model_name or not openrouter_api_key:
            raise ValueError(
                "Both arguments `openrouter_model_name` and `openrouter_api_key` are mandatory"
            )
        self.__openrouter_model_name = openrouter_model_name
        self.__openrouter_api_key = openrouter_api_key
        self.__logger = structlog.get_logger(self.__class__.__name__)
        self.__enable_multithreading = enable_multithreading
        self.__max_workers = max_workers
        self.__perimeter_path = perimeter_path or (
            Path(__file__).parent.parent.parent
            / "resources"
            / "sensocomune"
            / "sensocomune.base.json"
        )
        self.__cache_dir = self.__build_cache_dir(
            model_name=self.__openrouter_model_name,
            batch_class_name=self.__class__.__name__,
        )
        self.__error_log_path = self.__cache_dir / "errors.jsonl"
        self.__cache = self.__load_cache_from_filesystem()

    @property
    def _enable_multithreading(self) -> bool:
        return self.__enable_multithreading

    @property
    def _max_workers(self) -> int:
        return self.__max_workers

    def _get_cache(self) -> dict[str, Any]:
        return self.__cache

    def _get_openrouter_model_name(self) -> str:
        return self.__openrouter_model_name

    def _get_perimeter_path(self) -> Path:
        return self.__perimeter_path

    def __build_cache_dir(self, model_name: str, batch_class_name: str) -> Path:
        escaped_model_name: str = quote(model_name, safe="-_.")
        cache_dir: Path = (
            Path(__file__).parent.parent.parent
            / "resources"
            / "generated"
            / batch_class_name
            / escaped_model_name
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def __load_cache_from_filesystem(self) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        for cache_file in sorted(self.__cache_dir.glob("*.json")):
            if not cache_file.is_file():
                continue
            try:
                cache[cache_file.stem] = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                self.__logger.warning(
                    "Skipping malformed cache file",
                    cache_file=str(cache_file),
                )
        return cache

    def __persist_cache_entry(self, cache_key: str, result: Any) -> None:
        cache_file: Path = self.__cache_dir / f"{cache_key}.json"
        cache_file.write_text(
            json.dumps(
                result, ensure_ascii=False, indent=2, default=self.__json_default
            )
        )

    def __json_default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def __log_iteration_error(
        self,
        index: int,
        item: Any,
        error: Exception,
        cache_key: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "index": index,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "item": str(item),
            "cache_key": cache_key,
        }
        self.__error_log_path.write_text(
            self.__error_log_path.read_text()
            + json.dumps(payload, ensure_ascii=False)
            + "\n"
            if self.__error_log_path.exists()
            else json.dumps(payload, ensure_ascii=False) + "\n"
        )
        self.__logger.error(
            "Error processing perimeter item",
            index=index,
            cache_key=cache_key,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def _get_client(self) -> OpenAI:
        """Restituisce il client OpenAI specifico per OpenRouter.

        Returns:
            OpenAI: Il client per chiamare OpenRouter
        """
        return OpenAI(
            api_key=self.__openrouter_api_key, base_url="https://openrouter.ai/api/v1"
        )

    def _call_openrouter(self, user_prompt: str, system_prompt: str) -> ChatCompletion:
        return self._get_client().chat.completions.create(
            model=self.__openrouter_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    @abstractmethod
    def _get_perimeter(self) -> list[Any]:
        pass

    @abstractmethod
    def analyse(self, item: Any) -> Any:
        pass

    @abstractmethod
    def _get_cache_key(self, item: Any) -> str:
        pass

    def execute(self) -> dict:

        perimeter: list[Any] = self._get_perimeter()
        if not self.__enable_multithreading:
            for index, item in enumerate(
                tqdm(perimeter, desc="Analyzing items sequentially"),
                start=1,
            ):
                try:
                    cache_key: str = self._get_cache_key(item)
                    if cache_key in self.__cache:
                        continue

                    result: Any = self.analyse(item)
                    self.__cache[cache_key] = result
                    self.__persist_cache_entry(cache_key=cache_key, result=result)
                except Exception as exc:
                    self.__log_iteration_error(
                        index=index,
                        item=item,
                        error=exc,
                    )
            return self.__cache
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.__max_workers) as executor:
                future_to_ctx: dict[Any, tuple[int, Any, str]] = {}
                for index, item in enumerate(perimeter, start=1):
                    try:
                        cache_key: str = self._get_cache_key(item)
                        if cache_key in self.__cache:
                            continue
                        future = executor.submit(self.analyse, item)
                        future_to_ctx[future] = (index, item, cache_key)
                    except Exception as exc:
                        self.__log_iteration_error(
                            index=index,
                            item=item,
                            error=exc,
                        )

                for future in tqdm(
                    as_completed(future_to_ctx),
                    total=len(future_to_ctx),
                    desc="Analyzing items with multithreading",
                ):
                    index, item, cache_key = future_to_ctx[future]
                    try:
                        result: Any = future.result()
                        self.__cache[cache_key] = result
                        self.__persist_cache_entry(cache_key=cache_key, result=result)
                    except Exception as exc:
                        self.__log_iteration_error(
                            index=index,
                            item=item,
                            error=exc,
                            cache_key=cache_key,
                        )
            return self.__cache


class DefinitionGeneratorBatch(Batch):

    def __init__(
        self,
        openrouter_model_name: str,
        openrouter_api_key: str,
        enable_multithreading: bool = False,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            openrouter_model_name=openrouter_model_name,
            openrouter_api_key=openrouter_api_key,
            enable_multithreading=enable_multithreading,
            max_workers=max_workers,
            perimeter_path=Path(__file__).parent.parent.parent
            / "resources"
            / "sensocomune"
            / "tdm.sampled.json",
        )

    @override
    def execute(self) -> dict:
        return super().execute()

    @override
    def _get_perimeter(self) -> list[Lemma]:
        data: list[dict[str, Any]] = json.loads(self._get_perimeter_path().read_text())
        return SensoComune.from_json_list(data).lemmas

    @override
    def analyse(self, item: Lemma) -> dict[str, Any]:
        if not isinstance(item, Lemma):
            raise TypeError("DefinitionGeneratorBatch.analyse expects a Lemma item")

        user_prompt: str = PromptType.SENSE_EXTRACTION.prepare_user_prompt_template(
            lemma=item.lemma
        )
        system_prompt: str = PromptType.SENSE_EXTRACTION.get_system_prompt()
        response: ChatCompletion = self._call_openrouter(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        sense_extraction_content: str = (
            response.choices[0]
            .message.content.removeprefix("```json")
            .removesuffix("```")
        )
        try:
            definition_generation: DefinitionGeneration = (
                DefinitionGeneration.model_validate_json(sense_extraction_content)
            )
            return {
                "lemma": definition_generation.lemma,
                "lemma_data": item.to_dict(),
                "senses": [
                    sense.model_dump() for sense in definition_generation.senses
                ],
            }
        except ValueError:
            raise

    @override
    def _get_cache_key(self, item: Lemma) -> str:
        if not isinstance(item, Lemma):
            raise TypeError("DefinitionGeneratorBatch cache key expects a Lemma item")
        payload: str = json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DefinitionGeneratorV2Batch(Batch):
    """Senses-aware batch: the LLM receives the ground-truth sense count as min_senses input."""

    def __init__(
        self,
        openrouter_model_name: str,
        openrouter_api_key: str,
        enable_multithreading: bool = False,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            openrouter_model_name=openrouter_model_name,
            openrouter_api_key=openrouter_api_key,
            enable_multithreading=enable_multithreading,
            max_workers=max_workers,
            perimeter_path=Path(__file__).parent.parent.parent
            / "resources"
            / "sensocomune"
            / "tdm.sampled.json",
        )

    @override
    def _build_cache_dir(self, model_name: str, batch_class_name: str) -> Path:
        """Override to force a distinct cache directory name."""
        escaped_model_name: str = quote(model_name, safe="-_.")
        cache_dir: Path = (
            Path(__file__).parent.parent.parent
            / "resources"
            / "generated"
            / "DefinitionGeneratorV2Batch"
            / escaped_model_name
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @override
    def _get_cache_key(self, item: Lemma) -> str:
        if not isinstance(item, Lemma):
            raise TypeError("DefinitionGeneratorV2Batch cache key expects a Lemma item")
        payload: str = json.dumps(
            {"lemma_data": item.to_dict(), "min_senses": self.__count_senses(item)},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def __count_senses(item: Lemma) -> int:
        return sum(len(acc.senses) for acc in item.acceptations)

    @override
    def _get_perimeter(self) -> list[Lemma]:
        data: list[dict[str, Any]] = json.loads(self._get_perimeter_path().read_text())
        return SensoComune.from_json_list(data).lemmas

    @override
    def analyse(self, item: Lemma) -> dict[str, Any]:
        if not isinstance(item, Lemma):
            raise TypeError("DefinitionGeneratorV2Batch.analyse expects a Lemma item")

        min_senses: int = self.__count_senses(item)
        user_prompt: str = PromptType.DEFINITION_V2.prepare_user_prompt_template(
            lemma=item.lemma,
            min_senses=min_senses,
        )
        system_prompt: str = PromptType.DEFINITION_V2.get_system_prompt()
        response: ChatCompletion = self._call_openrouter(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        sense_extraction_content: str = (
            response.choices[0]
            .message.content.removeprefix("```json")
            .removesuffix("```")
        )
        try:
            definition_generation: DefinitionGeneration = (
                DefinitionGeneration.model_validate_json(sense_extraction_content)
            )
            return {
                "lemma": definition_generation.lemma,
                "min_senses": min_senses,
                "senses": [
                    sense.model_dump() for sense in definition_generation.senses
                ],
            }
        except ValueError:
            raise


class JudgementBatch(Batch):

    def __init__(
        self,
        openrouter_model_name: str,
        openrouter_api_key: str,
        enable_multithreading: bool = False,
        max_workers: int = 8,
        perimeter_path: Path | None = None,
    ) -> None:
        super().__init__(
            openrouter_model_name=openrouter_model_name,
            openrouter_api_key=openrouter_api_key,
            enable_multithreading=enable_multithreading,
            max_workers=max_workers,
            perimeter_path=perimeter_path,
        )

    @override
    def _get_perimeter(self) -> list[dict[str, Any]]:
        perimeter: list[dict[str, Any]] = []
        for file in self._get_perimeter_path().glob("*.json"):
            if not file.is_file():
                continue
            try:
                perimeter.append(json.loads(file.read_text()))
            except json.JSONDecodeError:
                self.__logger.warning(
                    "Skipping malformed perimeter file",
                    perimeter_file=str(file),
                )
        return perimeter

    @override
    def analyse(self, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise TypeError("JudgementBatch.analyse expects a definition result item")
        return self.__analyse_single(definition_result=item)

    @override
    def _get_cache_key(self, item: Any) -> str:
        if not isinstance(item, dict):
            raise TypeError("JudgementBatch cache key expects a definition result item")
        payload: str = json.dumps(item, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"judgement::{payload}".encode("utf-8")).hexdigest()

    def __analyse_single(self, definition_result: dict[str, Any]) -> dict[str, Any]:
        lemma: Lemma = self.__extract_lemma(definition_result=definition_result)
        sense_extraction: list[SenseExtraction] = self.__extract_sense(
            definition_result=definition_result
        )
        judgement: Judgement = self.__judge_extraction(
            lemma=lemma,
            sense_extraction=sense_extraction,
        )
        return {
            "lemma": lemma.lemma,
            "senses": [sense.model_dump() for sense in sense_extraction],
            "judgement": judgement.model_dump(),
            "tp": len(judgement.true_positive),
            "fp": len(judgement.false_positive),
            "fn": len(judgement.false_negative),
        }

    def __extract_lemma(self, definition_result: dict[str, Any]) -> Lemma:
        lemma_data: Any = definition_result.get("lemma_data")
        if not isinstance(lemma_data, dict):
            raise RuntimeError(
                "Missing or invalid lemma_data entry in definition result"
            )
        return Lemma.from_dict(lemma_data)

    def __extract_sense(
        self, definition_result: dict[str, Any]
    ) -> list[SenseExtraction]:
        sense_entries: Any = definition_result.get("senses", [])
        if not isinstance(sense_entries, list):
            raise RuntimeError("Invalid senses entry in definition result")
        return [SenseExtraction.model_validate(obj=entry) for entry in sense_entries]

    def __judge_extraction(
        self, lemma: Lemma, sense_extraction: list[SenseExtraction]
    ) -> Judgement:
        predicted_senses: str = "".join(
            [
                f"{i+1}. [{sense.pos}] {sense.definition}\n"
                for i, sense in enumerate(sense_extraction)
            ]
        )
        ground_truth_senses: str = "".join(
            [
                f"{i+1}. {sense.glossa}\n"
                for i, sense in enumerate(
                    sense
                    for acceptation in lemma.acceptations
                    for sense in acceptation.senses
                )
            ]
        )
        user_prompt: str = PromptType.JUDGEMENT_ALIGNMENT.prepare_user_prompt_template(
            ground_truth_senses=ground_truth_senses,
            predicted_senses=predicted_senses,
        )
        system_prompt: str = PromptType.JUDGEMENT_ALIGNMENT.get_system_prompt()
        response: ChatCompletion = self._call_openrouter(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        judgement_response: str = (
            response.choices[0]
            .message.content.removeprefix("```json")
            .removesuffix("```")
        )
        try:
            return Judgement.model_validate_json(json_data=judgement_response)
        except json.JSONDecodeError:
            raise


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SensoComune batches")
    parser.add_argument(
        "--openrouter-api-key",
        required=True,
        help="OpenRouter API key",
    )
    parser.add_argument(
        "--openrouter-model-name",
        required=True,
        help="OpenRouter model name",
    )
    parser.add_argument(
        "--enable-multithreading",
        action="store_true",
        help="Enable multithreading where supported",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of worker threads",
    )
    parser.add_argument(
        "--perimeter-path",
        type=Path,
        help="Optional path to override the default perimeter file for batches that support it",
        required=False,
    )

    subparsers = parser.add_subparsers(dest="batch", required=True)
    subparsers.add_parser("definition-generator", help="Run DefinitionGeneratorBatch")
    subparsers.add_parser("judgement", help="Run JudgementBatch")
    subparsers.add_parser(
        "definition-generator-v2", help="Run DefinitionGeneratorV2Batch"
    )
    subparsers.add_parser("judgement-on-v2", help="Run JudgementBatch (V2 perimeter)")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    common_kwargs = {
        "openrouter_model_name": args.openrouter_model_name,
        "openrouter_api_key": args.openrouter_api_key,
        "enable_multithreading": args.enable_multithreading,
        "max_workers": args.max_workers,
    }
    if args.perimeter_path:
        common_kwargs["perimeter_path"] = args.perimeter_path

    if args.batch == "definition-generator":
        result: Any = DefinitionGeneratorBatch(**common_kwargs).execute()
    elif args.batch == "definition-generator-v2":
        result: Any = DefinitionGeneratorV2Batch(**common_kwargs).execute()
    elif args.batch == "judgement":
        result = JudgementBatch(**common_kwargs).execute()
    elif args.batch == "judgement-on-v2":
        result = JudgementBatch(**common_kwargs).execute()
    else:
        raise ValueError(f"Unknown batch type: {args.batch}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
