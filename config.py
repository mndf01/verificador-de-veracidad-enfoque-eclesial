from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

class ModelsConfig(BaseModel):
    sb_light_model: str
    sb_medium_model: str
    sb_activo: str

class PathsConfig(BaseModel):
    data_dir: str
    # Zona Bronze: crudos (_v1.json) que generan los scrapers
    raw_corpus_dir: str
    # Zona Gold: procesados (_clasificado.json) listos para la IA
    corpus_dir: str
    chroma_dir: str


def resolver_ruta(relativa: str | Path) -> Path:
    """Resuelve una ruta de config.toml a absoluta contra la raíz del proyecto.

    Las rutas en config.toml son relativas (p. ej. "data/corp_extractor/raw").
    Al ejecutar los scripts desde distintos directorios, conviene anclarlas
    a la carpeta del proyecto (donde vive config.py) para que nunca se
    pierdan por el cwd.
    """
    ruta = Path(relativa)
    if ruta.is_absolute():
        return ruta
    return CONFIG_PATH.parent / ruta

class ChromaConfig(BaseModel):
    coleccion: str
    espacio_hnsw: str
    telemetria: bool
    batch_size: int

class CorpExtractorConfig(BaseModel):
    modo_ejecucion: str


# TODO -> para agregar otros config files tipo .env falta adaptar esta clase para que lea.
class Settings(BaseSettings):
    models: ModelsConfig
    paths: PathsConfig
    chroma: ChromaConfig
        
    #extractor del corpus vaticano
    corp_extractor: CorpExtractorConfig

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=CONFIG_PATH),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

@lru_cache
def obtener_configuraciones() -> Settings:
    return Settings()