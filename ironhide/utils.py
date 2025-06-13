def get_prompt(file_path: str, prompt: str) -> str:
    """Read and return the content of a prompt file.

    Args:
        file_path: Path to the main file.
        prompt: Name of the prompt file (relative to the main file's directory).

    Returns:
        The content of the prompt file.

    """
    system_file_path = Path(file_path).parent / prompt
    with system_file_path.open(encoding="utf-8") as system_file:
        return system_file.read()

