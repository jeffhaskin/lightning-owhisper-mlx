"""Pydantic models representing Deepgram-compatible request and response payloads."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Description of the transcription model in Deepgram-compatible format."""

    name: str
    version: str
    arch: str


class Metadata(BaseModel):
    """Metadata returned in each Deepgram result message."""

    request_id: str = Field(..., alias="request_id")
    model_info: ModelInfo


class WordTiming(BaseModel):
    """Represents the start and end offsets of a recognised word."""

    word: str
    start: float
    end: float
    confidence: Optional[float] = None


class Alternative(BaseModel):
    """A transcription alternative for a given channel."""

    transcript: str
    confidence: Optional[float] = None
    words: List[WordTiming] = Field(default_factory=list)


class ChannelResult(BaseModel):
    """Container for the alternatives detected within a channel."""

    alternatives: List[Alternative] = Field(default_factory=list)


class ListenV1Results(BaseModel):
    """Primary result payload returned to clients following Deepgram's schema."""

    type: str = Field("results", const=True)
    channel_index: List[int] = Field(default_factory=lambda: [0])
    duration: float
    start: float
    is_final: bool
    speech_final: bool
    channel: ChannelResult
    metadata: Metadata


class ListenV1Metadata(BaseModel):
    """Initial metadata message sent when the connection is established."""

    type: str = Field("metadata", const=True)
    metadata: Metadata


class ListenV1UtteranceEnd(BaseModel):
    """Message emitted when the server detects the end of an utterance."""

    type: str = Field("utterance_end", const=True)
    channel_index: List[int] = Field(default_factory=lambda: [0])
    metadata: Metadata


class ListenV1SpeechStarted(BaseModel):
    """Notification triggered when incoming audio contains speech-like content."""

    type: str = Field("speech_started", const=True)
    channel_index: List[int] = Field(default_factory=lambda: [0])
    metadata: Metadata


class InboundConfiguration(BaseModel):
    """Client configuration message."""

    type: str
    sample_rate: Optional[int] = None
    encoding: Optional[str] = None
    channels: Optional[int] = None
    language: Optional[str] = None


class InboundFinalize(BaseModel):
    """Finalise request payload."""

    type: str


class InboundKeepAlive(BaseModel):
    """Keep alive message."""

    type: str


InboundMessage = InboundConfiguration | InboundFinalize | InboundKeepAlive


__all__ = [
    "Alternative",
    "ChannelResult",
    "InboundConfiguration",
    "InboundFinalize",
    "InboundKeepAlive",
    "InboundMessage",
    "ListenV1Metadata",
    "ListenV1Results",
    "ListenV1SpeechStarted",
    "ListenV1UtteranceEnd",
    "Metadata",
    "ModelInfo",
    "WordTiming",
]
