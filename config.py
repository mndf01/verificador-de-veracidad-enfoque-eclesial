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
    corpus_dir: str
    chroma_dir: str

class ChromaConfig(BaseModel):
    coleccion: str
    espacio_hnsw: str
    telemetria: bool
    batch_size: int

# TODO -> para agregar otros config files tipo .env falta adaptar esta clase para que lea.
class Settings(BaseSettings):
    models: ModelsConfig
    paths: PathsConfig
    chroma: ChromaConfig

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