import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class NCBIConfig:
    tool: str = os.getenv("NCBI_TOOL", "synthclinclit")
    email: str = os.getenv("NCBI_EMAIL", "")
    api_key: str | None = os.getenv("NCBI_API_KEY") or None

    def common_params(self) -> dict[str, str]:
        params: dict[str, str] = {"tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params


NCBI = NCBIConfig()

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
