"""
Input validation utilities with retry mechanism.

Provides robust input validation with automatic retry on failure.
"""

from typing import Optional, List, Callable, Any
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def prompt_int_with_retry(
    message: str,
    default: Optional[int] = None,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    validator: Optional[Callable[[int], bool]] = None,
    validator_error_msg: Optional[str] = None,
) -> int:
    """Prompt for integer input with validation and retry.

    Args:
        message: Prompt message
        default: Default value (optional)
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        validator: Custom validation function (optional)
        validator_error_msg: Error message for custom validator (optional)

    Returns:
        Validated integer value
    """
    while True:
        # Build prompt with default
        if default is not None:
            prompt_text = f"{message} [{default}]"
        else:
            prompt_text = message

        # Get input
        user_input = Prompt.ask(prompt_text, default=str(default) if default is not None else None)

        # Try to parse as integer
        try:
            value = int(user_input)
        except ValueError:
            console.print(f"[red]✗ Invalid input. Please enter a valid integer.[/red]")
            console.print()
            continue

        # Validate range
        if min_val is not None and value < min_val:
            console.print(f"[red]✗ Value must be at least {min_val}[/red]")
            console.print()
            continue

        if max_val is not None and value > max_val:
            console.print(f"[red]✗ Value must be at most {max_val}[/red]")
            console.print()
            continue

        # Custom validation
        if validator is not None:
            if not validator(value):
                error_msg = validator_error_msg or "Invalid value"
                console.print(f"[red]✗ {error_msg}[/red]")
                console.print()
                continue

        # All validations passed
        return value


def prompt_float_with_retry(
    message: str,
    default: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Prompt for float input with validation and retry.

    Args:
        message: Prompt message
        default: Default value (optional)
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        Validated float value
    """
    while True:
        # Build prompt with default
        if default is not None:
            prompt_text = f"{message} [{default}]"
        else:
            prompt_text = message

        # Get input
        user_input = Prompt.ask(prompt_text, default=str(default) if default is not None else None)

        # Try to parse as float
        try:
            value = float(user_input)
        except ValueError:
            console.print(f"[red]✗ Invalid input. Please enter a valid number.[/red]")
            console.print()
            continue

        # Validate range
        if min_val is not None and value < min_val:
            console.print(f"[red]✗ Value must be at least {min_val}[/red]")
            console.print()
            continue

        if max_val is not None and value > max_val:
            console.print(f"[red]✗ Value must be at most {max_val}[/red]")
            console.print()
            continue

        # All validations passed
        return value


def prompt_choice_with_retry(
    message: str,
    choices: List[str],
    default: Optional[str] = None,
) -> str:
    """Prompt for choice input with validation and retry.

    Args:
        message: Prompt message
        choices: List of valid choices
        default: Default choice (optional)

    Returns:
        Selected choice
    """
    while True:
        # Get input
        user_input = Prompt.ask(message, default=default)

        # Validate choice
        if user_input not in choices:
            console.print(f"[red]✗ Invalid choice. Please select from: {', '.join(choices)}[/red]")
            console.print()
            continue

        return user_input


def prompt_int_list_with_retry(
    message: str,
    default: Optional[str] = None,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    validator: Optional[Callable[[List[int]], tuple[bool, Optional[str]]]] = None,
) -> List[int]:
    """Prompt for comma-separated integer list with validation and retry.

    Args:
        message: Prompt message
        default: Default value as string (e.g., "0,1,2,3")
        min_val: Minimum allowed value for each integer (optional)
        max_val: Maximum allowed value for each integer (optional)
        validator: Custom validation function that returns (is_valid, error_message) (optional)

    Returns:
        List of validated integers
    """
    while True:
        # Get input
        user_input = Prompt.ask(message, default=default)

        # Clean input: support Chinese comma and spaces
        user_input_cleaned = user_input.replace("，", ",").replace(" ", "")

        # Try to parse as integers
        try:
            values = [int(x.strip()) for x in user_input_cleaned.split(",") if x.strip()]
        except ValueError:
            console.print(f"[red]✗ Invalid format. Please enter numbers separated by commas.[/red]")
            console.print()
            continue

        # Validate each value's range
        invalid_values = []
        for value in values:
            if min_val is not None and value < min_val:
                invalid_values.append(value)
            elif max_val is not None and value > max_val:
                invalid_values.append(value)

        if invalid_values:
            if min_val is not None and max_val is not None:
                console.print(f"[red]✗ Invalid value(s): {invalid_values}[/red]")
                console.print(f"[yellow]Valid range: {min_val}-{max_val}[/yellow]")
            elif min_val is not None:
                console.print(f"[red]✗ Value(s) must be at least {min_val}: {invalid_values}[/red]")
            elif max_val is not None:
                console.print(f"[red]✗ Value(s) must be at most {max_val}: {invalid_values}[/red]")
            console.print()
            continue

        # Custom validation
        if validator is not None:
            is_valid, error_msg = validator(values)
            if not is_valid:
                console.print(f"[red]✗ {error_msg}[/red]")
                console.print()
                continue

        # All validations passed
        return values
