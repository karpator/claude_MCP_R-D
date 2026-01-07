from typing import TypeVar, Generic, Optional, Any
from pydantic import BaseModel, ConfigDict


class CommonInput(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"  # Allow additional fields for flexibility
    )

    pb_given_data: Optional[Any] = None


class CommonOutput(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"  # Allow additional fields for flexibility
    )


class CommonContext(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"  # Allow additional fields for flexibility
    )


TInput = TypeVar('TInput', bound=CommonInput)
TOutput = TypeVar('TOutput', bound=CommonOutput)
TContext = TypeVar('TContext', bound=CommonContext)


class SharedObject(BaseModel, Generic[TInput, TOutput, TContext]):
    """
    A Pydantic model for sharing data between nodes in a flow.
    Can be typed with specific input, output, and context models.
    """
    input: TInput
    output: TOutput
    context: TContext = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"  # Allow additional fields for flexibility
    )
