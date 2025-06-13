import json
import logging
from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI
from ironhide.gemini import GeminiAgent, tool
from ironhide.settings import settings
from pydantic import BaseModel, Field

app = FastAPI()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)s  %(filename)s  %(funcName)s  %(message)s",
)


class Request(BaseModel):
    """User Message to Agent."""

    content: str


class Contrato(BaseModel):
    indice_de_reajuste: str = Field(
        description="Índice usado para reajuste do contrato",
    )
    data_reajuste_contrato: str = Field(
        description="Data de reajuste do contrato",
    )
    data_inicio_validade_contrato: date = Field(
        description="Data de início da validade do contrato",
    )
    data_fim_validade_contrato: date = Field(
        description="Data de término da validade do contrato",
    )
    prazo_da_locacao_em_meses: int = Field(
        description="Duração do contrato em meses",
    )
    dobra_no_mes_dezembro: bool = Field(
        description="Indica se há valor dobrado em dezembro",
    )


class Imovel(BaseModel):
    valor_aluguel_em_reais: str = Field(
        description="Valor do aluguel em reais",
    )


class Dados(BaseModel):
    contrato: Contrato = Field(
        description="Informações do contrato",
    )
    imovel: Imovel = Field(
        description="Informações do imóvel",
    )


class Extractor(GeminiAgent):
    instructions = "You are an expert at structured data extraction. You will be given unstructured text and should convert it into the given structure."

agent = Extractor()


@app.post("/")
async def agent_message(
    message: Request,
) -> Dados:
    """Get response from agent."""
    return await agent.structured_chat(message.content, response_format=Dados)
