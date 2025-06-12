import json
import logging
from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI, File, UploadFile
from ironhide import OpenaiAgent, tool
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


class Image(OpenaiAgent):
    instructions = """You are an expert at extracting text from images and converting it into structured data. 
    Your task is to:
    1. First, carefully read and extract all text visible in the provided image
    2. Then, analyze the extracted text to identify relevant information
    3. Finally, convert the information into the requested structured format
    Be thorough in your text extraction and precise in your data structuring."""


agent = Image()


@app.post("/")
async def agent_message(
    file: Annotated[UploadFile, File()],
) -> str:
    """Extract text from image and convert to structured data."""
    contents = await file.read()
    files = {"file": (file.filename, contents, file.content_type)}
    return await agent.chat("Extract the text from the image", files=files)
