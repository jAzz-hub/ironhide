"""Enumerations and data models for the Ironhide framework."""

from enum import Enum


class Provider(str, Enum):
    """Enumeration of supported AI service providers."""

    openai = "openai"
    gemini = "gemini"
