import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CLIAgentDispatch"
    app_version: str = "0.1.0"
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8032, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # OpenCode Configuration
    opencode_server_url: str = Field(
        default="http://localhost:4096", alias="OPENCODE_SERVER_URL"
    )
    opencode_cli_path: str = Field(
        default=os.path.expanduser("~/.nvm/versions/node/v22.17.0/bin/opencode"),
        alias="OPENCODE_PATH",
    )
    opencode_default_model: str = Field(
        default="qwen3.7-flash", alias="OPENCODE_MODEL_ID"
    )
    opencode_default_provider: str = Field(
        default="litellm", alias="OPENCODE_PROVIDER_ID"
    )

    # Antigravity (agy) Configuration
    agy_path: str = Field(
        default=os.path.expanduser("~/.local/bin/agy"), alias="AGY_PATH"
    )
    agy_default_model: str = Field(
        default="Gemini 3.8 Flash (Low)", alias="AGY_MODEL"
    )

    # Execution Defaults
    default_timeout: int = Field(default=180, alias="DEFAULT_TIMEOUT")
    default_workspace: str = Field(
        default="/containers", alias="DEFAULT_WORKSPACE"
    )
    cors_origins: list[str] = ["*"]


settings = Settings()
