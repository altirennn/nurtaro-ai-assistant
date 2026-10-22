from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age_months: int = Field(ge=0, le=35)
    preferred_language: Literal["ru", "kk"] = "ru"
    interests: list[str] = Field(default_factory=list, max_length=20)
    available_materials: list[str] = Field(default_factory=list, max_length=30)
    parent_preferences: str = Field(default="", max_length=1000)

    @field_validator("interests", "available_materials")
    @classmethod
    def clean_list(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip()[:80] for value in values if value.strip()]
        return list(dict.fromkeys(cleaned))


class ProfileResponse(ProfileCreate):
    id: str
    created_at: str
    updated_at: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=4000)
    profile_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Вопрос не должен быть пустым")
        return value


class SourceReference(BaseModel):
    document_id: str
    source_id: str
    title: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    language: Literal["ru", "kk"]
    sources: list[SourceReference]
    abstained: bool
    generation_mode: str
    latency_ms: float
    personalization: dict[str, object] | None = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    source_type: str
    indexed_chunks: int
    is_shared: bool
    created_at: str


class IndexRequest(BaseModel):
    document_id: str | None = Field(default=None, max_length=64)
    force: bool = False


class EvaluationResponse(BaseModel):
    run_id: str
    summary: dict[str, object]
    results_json: str
    results_csv: str


class VideoDetection(BaseModel):
    class_name: str
    confidence: float
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4, description="Координаты x1,y1,x2,y2 в пикселях исходного кадра")


class VideoFrameResult(BaseModel):
    timestamp_seconds: float
    detections: list[VideoDetection]


class VideoAnalysisResponse(BaseModel):
    model: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    sampled_frames: int
    detection_counts_by_class: dict[str, int]
    frames: list[VideoFrameResult]
    inference_ms: float
    warnings: list[str] = Field(default_factory=list)
