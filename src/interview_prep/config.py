from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    tavily_api_key: str = ""
    model: str = ""
    provider: str = "fireworks"
    base_url: str = "https://api.fireworks.ai/inference/v1"
    model_timeout: float = 75.0
    model_max_tokens: int = 5000
    input_max_chars: int = 40000
    upload_max_bytes: int = 1000000
    runtime_dir: Path = ROOT / "runtime"

    @classmethod
    def from_env(cls):
        # Read on each explicit engine creation so locally supplied keys take effect.
        load_dotenv(ROOT / ".env", override=True)
        provider = os.getenv("MODEL_PROVIDER", "fireworks").strip().lower()
        if provider not in ("fireworks", "nebius"):
            raise ValueError("MODEL_PROVIDER must be fireworks or nebius")
        prefix = "NEBIUS" if provider == "nebius" else "FIREWORKS"
        model = os.getenv(prefix + "_MODEL", "").strip()
        # Bounded calibration for this reasoning model after a 5,000-token
        # structured profile ended at the output limit. Other models keep 5,000.
        output_cap = 8192 if provider == "fireworks" and model == "accounts/fireworks/models/glm-5p3-flash" else 5000
        return cls(api_key=os.getenv(prefix + "_API_KEY", "").strip(),
                   tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
                   model=model, provider=provider, model_max_tokens=output_cap,
                   base_url="https://api.tokenfactory.nebius.com/v1" if provider == "nebius" else "https://api.fireworks.ai/inference/v1")

    def readiness(self):
        missing = []
        if not self.api_key:
            missing.append(f"Set {self.provider.upper()}_API_KEY in the local .env file.")
        if not self.model:
            missing.append(f"Set {self.provider.upper()}_MODEL to the selected tool-capable model in .env.")
        if not self.tavily_api_key:
            missing.append("Set TAVILY_API_KEY in the local .env file.")
        return missing
