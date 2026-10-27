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
    app_version: str = "0.3.3"
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8032, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # OpenCode Configuration
    opencode_server_url: str = Field(default="http://localhost:4096", alias="OPENCODE_SERVER_URL")
    opencode_cli_path: str = Field(
        default=os.path.expanduser("~/.nvm/versions/node/v22.17.0/bin/opencode"),
        alias="OPENCODE_PATH",
    )
    opencode_default_model: str = Field(default="opencode-default", alias="OPENCODE_MODEL_ID")
    opencode_default_provider: str = Field(default="litellm", alias="OPENCODE_PROVIDER_ID")

    # Antigravity (agy) Configuration
    agy_path: str = Field(default=os.path.expanduser("~/.local/bin/agy"), alias="AGY_PATH")
    agy_default_model: str = Field(default="Gemini 3.8 Flash (Low)", alias="AGY_MODEL")

    # T3 Code (t3) Configuration
    t3_server_url: str = Field(default="http://127.0.0.1:3773", alias="T3_SERVER_URL")
    t3_cli_path: str = Field(default=os.path.expanduser("~/.local/bin/t3"), alias="T3_CLI_PATH")
    t3_ui_url: str = Field(default="https://ttt.wileyriley.com", alias="T3_UI_URL")
    t3_default_provider: str = Field(default="antigravity", alias="T3_DEFAULT_PROVIDER")
    t3_default_model: str = Field(default="gemini-3.8-flash-high", alias="T3_DEFAULT_MODEL")
    t3_token_cache_path: str = Field(
        default=os.path.expanduser("~/.t3/userdata/secrets/cli_agent_dispatch_mcp_token.json"),
        alias="T3_TOKEN_CACHE_PATH",
    )

    # ntfy Push Notification Configuration
    ntfy_url: str = Field(default="http://10.0.0.10:9010/wileyriley-sso", alias="NTFY_URL")
    ntfy_user: str = Field(default="steve", alias="NTFY_USER")
    ntfy_pass: str = Field(default="topfire89", alias="NTFY_PASS")

    # Execution Defaults
    default_timeout: int = Field(default=180, alias="DEFAULT_TIMEOUT")
    default_workspace: str = Field(default="/containers", alias="DEFAULT_WORKSPACE")
    cors_origins: list[str] = ["*"]


settings = Settings()
