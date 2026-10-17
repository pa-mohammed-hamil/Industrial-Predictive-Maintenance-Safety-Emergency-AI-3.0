# app/agents/tools/knowledge_tools.py

from __future__ import annotations

import hashlib
import inspect
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def finite(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if hasattr(value, "model_dump"):
        try:
            return json_safe(
                value.model_dump(mode="json")
            )
        except Exception:
            return json_safe(
                value.model_dump()
            )

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def normalize_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        value.strip(),
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# =========================================================
# Enums
# =========================================================

class KnowledgeSourceType(str, Enum):
    MANUAL = "manual"
    SOP = "sop"
    SAFETY_PROCEDURE = "safety_procedure"
    MAINTENANCE_GUIDE = "maintenance_guide"
    SERVICE_MANUAL = "service_manual"
    OEM_DOCUMENT = "oem_document"

    INSPECTION_GUIDE = "inspection_guide"

    FAILURE_GUIDE = "failure_guide"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"

    INCIDENT_REPORT = "incident_report"
    MAINTENANCE_RECORD = "maintenance_record"
    WORK_ORDER = "work_order"

    ENGINEERING_STANDARD = "engineering_standard"
    POLICY = "policy"
    REGULATION = "regulation"

    TECHNICAL_NOTE = "technical_note"

    OTHER = "other"


class KnowledgeScope(str, Enum):
    GLOBAL = "global"
    ORGANIZATION = "organization"
    SITE = "site"
    ZONE = "zone"
    ASSET_TYPE = "asset_type"
    ASSET = "asset"


class RetrievalMethod(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    GRAPH = "graph"
    PARENT_CHILD = "parent_child"
    EXACT = "exact"


class EvidenceStrength(str, Enum):
    UNKNOWN = "unknown"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    AUTHORITATIVE = "authoritative"


class KnowledgeStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACCESS_DENIED = "access_denied"
    ERROR = "error"


# =========================================================
# Query
# =========================================================

@dataclass(slots=True)
class KnowledgeQuery:
    query: str

    asset_id: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    source_types: list[KnowledgeSourceType] = field(
        default_factory=list
    )

    preferred_method: RetrievalMethod = (
        RetrievalMethod.HYBRID
    )

    top_k: int = 8

    minimum_score: float = 0.25

    require_approved: bool = True
    require_current: bool = True

    include_historical_records: bool = False

    user_id: str | None = None
    tenant_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.query = normalize_text(self.query)

        if not self.query:
            raise ValueError("query cannot be empty.")

        self.top_k = max(
            1,
            min(int(self.top_k), 50),
        )

        self.minimum_score = clamp(
            self.minimum_score
        )


# =========================================================
# Retrieved chunk
# =========================================================

@dataclass(slots=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str

    title: str

    text: str

    source_type: KnowledgeSourceType

    retrieval_method: RetrievalMethod

    score: float

    source_name: str | None = None
    source_uri: str | None = None

    page: int | None = None
    section: str | None = None

    asset_id: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    approved: bool = False

    effective_from: datetime | None = None
    effective_until: datetime | None = None

    version: str | None = None

    authority: float = 0.5

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.score = clamp(self.score)
        self.authority = clamp(self.authority)

        self.text = normalize_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Citation
# =========================================================

@dataclass(slots=True)
class KnowledgeCitation:
    citation_id: str

    document_id: str
    chunk_id: str

    title: str

    source_type: KnowledgeSourceType

    source_name: str | None

    source_uri: str | None

    page: int | None
    section: str | None

    quote: str | None

    retrieval_score: float
    authority_score: float

    evidence_strength: EvidenceStrength

    version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class KnowledgeEvidence:
    evidence_id: str

    claim: str

    supported: bool

    confidence: float

    citations: list[KnowledgeCitation]

    explanation: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Search result
# =========================================================

@dataclass(slots=True)
class KnowledgeSearchResult:
    status: KnowledgeStatus

    query: str

    chunks: list[KnowledgeChunk]

    citations: list[KnowledgeCitation]

    confidence: float

    retrieval_method: RetrievalMethod

    total_results: int

    warnings: list[str] = field(
        default_factory=list
    )

    generated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Grounded answer
# =========================================================

@dataclass(slots=True)
class GroundedKnowledgeAnswer:
    answer_id: str

    query: str

    answer: str

    grounded: bool

    confidence: float

    evidence: list[KnowledgeEvidence]

    citations: list[KnowledgeCitation]

    unsupported_claims: list[str]

    warnings: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# SOP result
# =========================================================

@dataclass(slots=True)
class SOPResult:
    found: bool

    procedure_name: str | None

    procedure_text: str | None

    document_id: str | None

    version: str | None

    approved: bool

    citation: KnowledgeCitation | None

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Interfaces
# =========================================================

class HybridRetriever(Protocol):

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Mapping[str, Any],
    ) -> Sequence[Any]:
        ...


class VectorRetriever(Protocol):

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Mapping[str, Any],
    ) -> Sequence[Any]:
        ...


class BM25Retriever(Protocol):

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Mapping[str, Any],
    ) -> Sequence[Any]:
        ...


class GraphRetriever(Protocol):

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Mapping[str, Any],
    ) -> Sequence[Any]:
        ...


class Reranker(Protocol):

    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Any],
        top_k: int,
    ) -> Sequence[Any]:
        ...


class PermissionService(Protocol):

    async def can_access_document(
        self,
        *,
        user_id: str | None,
        tenant_id: str | None,
        document_id: str,
    ) -> bool:
        ...


class KnowledgeRepository(Protocol):

    async def get_document(
        self,
        document_id: str,
    ) -> Any | None:
        ...


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class KnowledgeToolsConfig:
    default_top_k: int = 8

    maximum_top_k: int = 30

    minimum_retrieval_score: float = 0.25

    minimum_grounding_confidence: float = 0.60

    strong_evidence_threshold: float = 0.75
    authoritative_threshold: float = 0.90

    max_quote_characters: int = 300

    rerank_enabled: bool = True

    permission_filter_enabled: bool = True

    require_approved_safety_documents: bool = True

    exclude_expired_documents: bool = True


# =========================================================
# Knowledge tools
# =========================================================

class KnowledgeTools:

    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        graph_retriever: GraphRetriever | None = None,
        reranker: Reranker | None = None,
        permission_service: PermissionService | None = None,
        repository: KnowledgeRepository | None = None,
        config: KnowledgeToolsConfig | None = None,
    ) -> None:

        self.hybrid_retriever = hybrid_retriever
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever

        self.reranker = reranker

        self.permission_service = permission_service

        self.repository = repository

        self.config = (
            config
            or KnowledgeToolsConfig()
        )

    # =====================================================
    # Search
    # =====================================================

    async def search(
        self,
        request: KnowledgeQuery,
    ) -> KnowledgeSearchResult:

        warnings: list[str] = []

        filters = self._build_filters(request)

        try:
            raw = await self._retrieve(
                request=request,
                filters=filters,
            )

        except Exception as exc:
            return KnowledgeSearchResult(
                status=KnowledgeStatus.ERROR,

                query=request.query,

                chunks=[],
                citations=[],

                confidence=0.0,

                retrieval_method=(
                    request.preferred_method
                ),

                total_results=0,

                warnings=[
                    f"Retrieval failed: {exc}"
                ],
            )

        chunks = [
            self._normalize_chunk(
                item,
                request.preferred_method,
            )
            for item in raw
        ]

        chunks = [
            item
            for item in chunks
            if item.score
            >= request.minimum_score
        ]

        chunks = self._filter_current(
            chunks,
            request,
        )

        chunks = await self._filter_permissions(
            chunks,
            request,
        )

        chunks = self._deduplicate(chunks)

        if (
            self.config.rerank_enabled
            and self.reranker is not None
            and chunks
        ):
            chunks = await self._rerank(
                request,
                chunks,
            )

        chunks = chunks[:request.top_k]

        if not chunks:
            return KnowledgeSearchResult(
                status=KnowledgeStatus.NO_RESULTS,

                query=request.query,

                chunks=[],
                citations=[],

                confidence=0.0,

                retrieval_method=(
                    request.preferred_method
                ),

                total_results=0,

                warnings=[
                    "No sufficiently relevant "
                    "knowledge was retrieved."
                ],
            )

        citations = [
            self._citation_from_chunk(chunk)
            for chunk in chunks
        ]

        confidence = (
            self._retrieval_confidence(chunks)
        )

        if (
            confidence
            < self.config
            .minimum_grounding_confidence
        ):
            status = (
                KnowledgeStatus
                .INSUFFICIENT_EVIDENCE
            )

            warnings.append(
                "Retrieved evidence is below the "
                "configured grounding confidence."
            )

        else:
            status = KnowledgeStatus.SUCCESS

        result = KnowledgeSearchResult(
            status=status,

            query=request.query,

            chunks=chunks,

            citations=citations,

            confidence=confidence,

            retrieval_method=(
                request.preferred_method
            ),

            total_results=len(chunks),

            warnings=warnings,

            metadata={
                "asset_id": request.asset_id,
                "site_id": request.site_id,
                "zone_id": request.zone_id,
            },
        )

        self._record_metrics(result)

        return result

    # =====================================================
    # SOP
    # =====================================================

    async def get_sop(
        self,
        *,
        query: str,
        asset_id: str | None = None,
        asset_type: str | None = None,
        site_id: str | None = None,
        zone_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> SOPResult:

        result = await self.search(
            KnowledgeQuery(
                query=query,

                asset_id=asset_id,
                asset_type=asset_type,

                site_id=site_id,
                zone_id=zone_id,

                source_types=[
                    KnowledgeSourceType.SOP,
                    KnowledgeSourceType.SAFETY_PROCEDURE,
                    KnowledgeSourceType.MAINTENANCE_GUIDE,
                ],

                top_k=5,

                require_approved=True,
                require_current=True,

                user_id=user_id,
                tenant_id=tenant_id,
            )
        )

        if not result.chunks:
            return SOPResult(
                found=False,

                procedure_name=None,
                procedure_text=None,

                document_id=None,
                version=None,

                approved=False,

                citation=None,

                warnings=result.warnings,
            )

        chunk = result.chunks[0]

        return SOPResult(
            found=True,

            procedure_name=chunk.title,

            procedure_text=chunk.text,

            document_id=chunk.document_id,

            version=chunk.version,

            approved=chunk.approved,

            citation=(
                self._citation_from_chunk(
                    chunk
                )
            ),

            warnings=result.warnings,
        )

    # =====================================================
    # Maintenance knowledge
    # =====================================================

    async def maintenance_guidance(
        self,
        *,
        asset_id: str,
        asset_type: str | None,
        failure_mode: str | None,
        symptoms: Sequence[str],
        site_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> KnowledgeSearchResult:

        symptom_text = ", ".join(
            normalize_text(item)
            for item in symptoms
            if item.strip()
        )

        query = (
            f"Maintenance procedure for "
            f"{asset_type or 'industrial asset'}"
        )

        if failure_mode:
            query += (
                f" with failure mode "
                f"{failure_mode}"
            )

        if symptom_text:
            query += (
                f". Symptoms: {symptom_text}"
            )

        return await self.search(
            KnowledgeQuery(
                query=query,

                asset_id=asset_id,
                asset_type=asset_type,

                site_id=site_id,

                source_types=[
                    KnowledgeSourceType.MAINTENANCE_GUIDE,
                    KnowledgeSourceType.SERVICE_MANUAL,
                    KnowledgeSourceType.OEM_DOCUMENT,
                    KnowledgeSourceType.SOP,
                    KnowledgeSourceType.TROUBLESHOOTING_GUIDE,
                ],

                top_k=8,

                user_id=user_id,
                tenant_id=tenant_id,
            )
        )

    # =====================================================
    # Failure knowledge
    # =====================================================

    async def failure_guidance(
        self,
        *,
        asset_id: str | None,
        asset_type: str | None,
        failure_mode: str,
        symptoms: Sequence[str] = (),
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> KnowledgeSearchResult:

        symptom_text = ", ".join(
            normalize_text(item)
            for item in symptoms
            if item.strip()
        )

        query = (
            f"Root causes, diagnostic checks and "
            f"recommended actions for "
            f"{failure_mode}"
        )

        if asset_type:
            query += f" on {asset_type}"

        if symptom_text:
            query += (
                f". Observed symptoms: "
                f"{symptom_text}"
            )

        return await self.search(
            KnowledgeQuery(
                query=query,

                asset_id=asset_id,
                asset_type=asset_type,

                source_types=[
                    KnowledgeSourceType.FAILURE_GUIDE,
                    KnowledgeSourceType.TROUBLESHOOTING_GUIDE,
                    KnowledgeSourceType.SERVICE_MANUAL,
                    KnowledgeSourceType.OEM_DOCUMENT,
                    KnowledgeSourceType.MAINTENANCE_GUIDE,
                ],

                top_k=8,

                user_id=user_id,
                tenant_id=tenant_id,
            )
        )

    # =====================================================
    # Safety knowledge
    # =====================================================

    async def safety_procedure(
        self,
        *,
        hazard: str,
        asset_id: str | None = None,
        asset_type: str | None = None,
        site_id: str | None = None,
        zone_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> KnowledgeSearchResult:

        return await self.search(
            KnowledgeQuery(
                query=(
                    f"Approved emergency and safety "
                    f"procedure for {hazard}"
                ),

                asset_id=asset_id,
                asset_type=asset_type,

                site_id=site_id,
                zone_id=zone_id,

                source_types=[
                    KnowledgeSourceType.SAFETY_PROCEDURE,
                    KnowledgeSourceType.SOP,
                    KnowledgeSourceType.ENGINEERING_STANDARD,
                    KnowledgeSourceType.POLICY,
                ],

                top_k=8,

                require_approved=True,
                require_current=True,

                user_id=user_id,
                tenant_id=tenant_id,

                metadata={
                    "safety_critical": True,
                },
            )
        )

    # =====================================================
    # Inspection knowledge
    # =====================================================

    async def inspection_guidance(
        self,
        *,
        asset_id: str | None,
        asset_type: str | None,
        defect: str,
        severity: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> KnowledgeSearchResult:

        query = (
            f"Inspection criteria and recommended "
            f"action for {defect}"
        )

        if asset_type:
            query += f" on {asset_type}"

        if severity:
            query += f" severity {severity}"

        return await self.search(
            KnowledgeQuery(
                query=query,

                asset_id=asset_id,
                asset_type=asset_type,

                source_types=[
                    KnowledgeSourceType.INSPECTION_GUIDE,
                    KnowledgeSourceType.SERVICE_MANUAL,
                    KnowledgeSourceType.OEM_DOCUMENT,
                    KnowledgeSourceType.ENGINEERING_STANDARD,
                ],

                top_k=8,

                user_id=user_id,
                tenant_id=tenant_id,
            )
        )

    # =====================================================
    # Similar incidents
    # =====================================================

    async def similar_incidents(
        self,
        *,
        description: str,
        asset_type: str | None = None,
        site_id: str | None = None,
        top_k: int = 5,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> KnowledgeSearchResult:

        return await self.search(
            KnowledgeQuery(
                query=(
                    "Similar historical industrial "
                    f"incidents: {description}"
                ),

                asset_type=asset_type,
                site_id=site_id,

                source_types=[
                    KnowledgeSourceType.INCIDENT_REPORT,
                    KnowledgeSourceType.MAINTENANCE_RECORD,
                    KnowledgeSourceType.WORK_ORDER,
                ],

                top_k=top_k,

                require_approved=False,

                include_historical_records=True,

                user_id=user_id,
                tenant_id=tenant_id,
            )
        )

    # =====================================================
    # Evidence for a claim
    # =====================================================

    async def evidence_for_claim(
        self,
        *,
        claim: str,
        asset_id: str | None = None,
        asset_type: str | None = None,
        site_id: str | None = None,
        source_types: Sequence[
            KnowledgeSourceType
        ] = (),
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> KnowledgeEvidence:

        result = await self.search(
            KnowledgeQuery(
                query=claim,

                asset_id=asset_id,
                asset_type=asset_type,
                site_id=site_id,

                source_types=list(source_types),

                top_k=6,

                user_id=user_id,
                tenant_id=tenant_id,
            )
        )

        supported = bool(
            result.citations
            and result.confidence
            >= self.config
            .minimum_grounding_confidence
        )

        if supported:
            explanation = (
                "The claim has supporting "
                "retrieved evidence."
            )
        else:
            explanation = (
                "The available knowledge base does "
                "not provide sufficiently strong "
                "evidence for this claim."
            )

        return KnowledgeEvidence(
            evidence_id=new_id(),

            claim=claim,

            supported=supported,

            confidence=result.confidence,

            citations=result.citations,

            explanation=explanation,
        )

    # =====================================================
    # Ground multiple claims
    # =====================================================

    async def ground_claims(
        self,
        claims: Sequence[str],
        *,
        asset_id: str | None = None,
        asset_type: str | None = None,
        site_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[KnowledgeEvidence]:

        evidence: list[KnowledgeEvidence] = []

        for claim in claims:
            claim = normalize_text(claim)

            if not claim:
                continue

            item = await self.evidence_for_claim(
                claim=claim,

                asset_id=asset_id,
                asset_type=asset_type,
                site_id=site_id,

                user_id=user_id,
                tenant_id=tenant_id,
            )

            evidence.append(item)

        return evidence

    # =====================================================
    # Build grounded answer
    # =====================================================

    def build_grounded_answer(
        self,
        *,
        query: str,
        answer: str,
        evidence: Sequence[KnowledgeEvidence],
        metadata: Mapping[str, Any] | None = None,
    ) -> GroundedKnowledgeAnswer:

        evidence_list = list(evidence)

        unsupported = [
            item.claim
            for item in evidence_list
            if not item.supported
        ]

        citations: list[KnowledgeCitation] = []

        seen: set[tuple[str, str]] = set()

        for item in evidence_list:
            for citation in item.citations:
                key = (
                    citation.document_id,
                    citation.chunk_id,
                )

                if key in seen:
                    continue

                seen.add(key)
                citations.append(citation)

        if evidence_list:
            confidence = sum(
                item.confidence
                for item in evidence_list
            ) / len(evidence_list)
        else:
            confidence = 0.0

        grounded = bool(
            evidence_list
            and not unsupported
            and confidence
            >= self.config
            .minimum_grounding_confidence
        )

        warnings: list[str] = []

        if unsupported:
            warnings.append(
                f"{len(unsupported)} claim(s) "
                "lack sufficient supporting "
                "evidence."
            )

        if not citations:
            warnings.append(
                "No citations are attached to the "
                "answer."
            )

        return GroundedKnowledgeAnswer(
            answer_id=new_id(),

            query=query,
            answer=answer,

            grounded=grounded,

            confidence=round(
                clamp(confidence),
                4,
            ),

            evidence=evidence_list,

            citations=citations,

            unsupported_claims=unsupported,

            warnings=warnings,

            metadata=dict(
                metadata or {}
            ),
        )

    # =====================================================
    # Retrieval
    # =====================================================

    async def _retrieve(
        self,
        *,
        request: KnowledgeQuery,
        filters: Mapping[str, Any],
    ) -> Sequence[Any]:

        method = request.preferred_method

        retriever: Any = None

        if method == RetrievalMethod.HYBRID:
            retriever = self.hybrid_retriever

        elif method == RetrievalMethod.VECTOR:
            retriever = self.vector_retriever

        elif method == RetrievalMethod.BM25:
            retriever = self.bm25_retriever

        elif method == RetrievalMethod.GRAPH:
            retriever = self.graph_retriever

        if retriever is None:
            # Graceful fallback order.
            retriever = (
                self.hybrid_retriever
                or self.vector_retriever
                or self.bm25_retriever
                or self.graph_retriever
            )

        if retriever is None:
            raise RuntimeError(
                "No knowledge retriever is configured."
            )

        return await maybe_await(
            retriever.search(
                query=request.query,
                top_k=min(
                    request.top_k * 3,
                    self.config.maximum_top_k,
                ),
                filters=filters,
            )
        )

    # =====================================================
    # Reranking
    # =====================================================

    async def _rerank(
        self,
        request: KnowledgeQuery,
        chunks: Sequence[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:

        if self.reranker is None:
            return list(chunks)

        try:
            result = await maybe_await(
                self.reranker.rerank(
                    query=request.query,
                    documents=chunks,
                    top_k=request.top_k,
                )
            )

            normalized: list[KnowledgeChunk] = []

            for item in result:
                if isinstance(
                    item,
                    KnowledgeChunk,
                ):
                    normalized.append(item)
                    continue

                normalized.append(
                    self._normalize_chunk(
                        item,
                        request.preferred_method,
                    )
                )

            return normalized

        except Exception:
            # Retrieval remains usable if reranking
            # fails.
            return list(chunks)

    # =====================================================
    # Permissions
    # =====================================================

    async def _filter_permissions(
        self,
        chunks: Sequence[KnowledgeChunk],
        request: KnowledgeQuery,
    ) -> list[KnowledgeChunk]:

        if (
            not self.config
            .permission_filter_enabled
            or self.permission_service is None
        ):
            return list(chunks)

        permitted: list[KnowledgeChunk] = []

        permission_cache: dict[str, bool] = {}

        for chunk in chunks:
            allowed = permission_cache.get(
                chunk.document_id
            )

            if allowed is None:
                try:
                    allowed = bool(
                        await maybe_await(
                            self.permission_service
                            .can_access_document(
                                user_id=request.user_id,
                                tenant_id=request.tenant_id,
                                document_id=(
                                    chunk.document_id
                                ),
                            )
                        )
                    )
                except Exception:
                    # Fail closed.
                    allowed = False

                permission_cache[
                    chunk.document_id
                ] = allowed

            if allowed:
                permitted.append(chunk)

        return permitted

    # =====================================================
    # Current / approved filter
    # =====================================================

    def _filter_current(
        self,
        chunks: Sequence[KnowledgeChunk],
        request: KnowledgeQuery,
    ) -> list[KnowledgeChunk]:

        now = utc_now()

        output: list[KnowledgeChunk] = []

        safety_query = bool(
            request.metadata.get(
                "safety_critical"
            )
        )

        for chunk in chunks:
            if request.require_approved:
                if not chunk.approved:
                    continue

            if (
                safety_query
                and self.config
                .require_approved_safety_documents
                and not chunk.approved
            ):
                continue

            if (
                request.require_current
                and self.config
                .exclude_expired_documents
            ):
                if (
                    chunk.effective_from
                    and chunk.effective_from > now
                ):
                    continue

                if (
                    chunk.effective_until
                    and chunk.effective_until < now
                ):
                    continue

            output.append(chunk)

        return output

    # =====================================================
    # Filters
    # =====================================================

    @staticmethod
    def _build_filters(
        request: KnowledgeQuery,
    ) -> dict[str, Any]:

        filters: dict[str, Any] = {}

        if request.asset_id:
            filters["asset_id"] = request.asset_id

        if request.asset_type:
            filters["asset_type"] = (
                request.asset_type
            )

        if request.site_id:
            filters["site_id"] = request.site_id

        if request.zone_id:
            filters["zone_id"] = request.zone_id

        if request.source_types:
            filters["source_types"] = [
                item.value
                for item in request.source_types
            ]

        filters["include_historical_records"] = (
            request.include_historical_records
        )

        return filters

    # =====================================================
    # Normalize chunk
    # =====================================================

    def _normalize_chunk(
        self,
        raw: Any,
        method: RetrievalMethod,
    ) -> KnowledgeChunk:

        if isinstance(raw, KnowledgeChunk):
            return raw

        def get(
            key: str,
            default: Any = None,
        ) -> Any:
            if isinstance(raw, Mapping):
                return raw.get(key, default)

            return getattr(raw, key, default)

        source_type_raw = get(
            "source_type",
            "other",
        )

        if isinstance(
            source_type_raw,
            KnowledgeSourceType,
        ):
            source_type = source_type_raw
        else:
            try:
                source_type = (
                    KnowledgeSourceType(
                        str(
                            source_type_raw
                        ).lower()
                    )
                )
            except ValueError:
                source_type = (
                    KnowledgeSourceType.OTHER
                )

        retrieval_method_raw = get(
            "retrieval_method",
            method,
        )

        if isinstance(
            retrieval_method_raw,
            RetrievalMethod,
        ):
            retrieval_method = (
                retrieval_method_raw
            )
        else:
            try:
                retrieval_method = (
                    RetrievalMethod(
                        str(
                            retrieval_method_raw
                        ).lower()
                    )
                )
            except ValueError:
                retrieval_method = method

        text = str(
            get(
                "text",
                get(
                    "content",
                    "",
                ),
            )
            or ""
        )

        document_id = str(
            get(
                "document_id",
                get(
                    "doc_id",
                    "unknown",
                ),
            )
        )

        chunk_id = str(
            get(
                "chunk_id",
                get(
                    "id",
                    stable_hash(
                        document_id + text
                    )[:24],
                ),
            )
        )

        return KnowledgeChunk(
            chunk_id=chunk_id,

            document_id=document_id,

            title=str(
                get(
                    "title",
                    "Untitled",
                )
            ),

            text=text,

            source_type=source_type,

            retrieval_method=(
                retrieval_method
            ),

            score=clamp(
                finite(
                    get(
                        "score",
                        get(
                            "relevance_score",
                            0.0,
                        ),
                    )
                )
            ),

            source_name=get(
                "source_name"
            ),

            source_uri=get(
                "source_uri",
                get("url"),
            ),

            page=self._optional_int(
                get("page")
            ),

            section=get("section"),

            asset_id=get("asset_id"),
            asset_type=get("asset_type"),

            site_id=get("site_id"),
            zone_id=get("zone_id"),

            approved=bool(
                get(
                    "approved",
                    False,
                )
            ),

            effective_from=(
                self._datetime(
                    get("effective_from")
                )
            ),

            effective_until=(
                self._datetime(
                    get("effective_until")
                )
            ),

            version=(
                str(get("version"))
                if get("version") is not None
                else None
            ),

            authority=clamp(
                finite(
                    get(
                        "authority",
                        get(
                            "authority_score",
                            0.5,
                        ),
                    ),
                    default=0.5,
                )
            ),

            metadata=dict(
                get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    # =====================================================
    # Citation
    # =====================================================

    def _citation_from_chunk(
        self,
        chunk: KnowledgeChunk,
    ) -> KnowledgeCitation:

        combined = (
            0.65 * chunk.score
            + 0.35 * chunk.authority
        )

        if (
            combined
            >= self.config
            .authoritative_threshold
            and chunk.approved
        ):
            strength = (
                EvidenceStrength.AUTHORITATIVE
            )

        elif (
            combined
            >= self.config
            .strong_evidence_threshold
        ):
            strength = EvidenceStrength.STRONG

        elif combined >= 0.50:
            strength = EvidenceStrength.MODERATE

        elif combined > 0:
            strength = EvidenceStrength.WEAK

        else:
            strength = EvidenceStrength.UNKNOWN

        quote = chunk.text[
            : self.config.max_quote_characters
        ]

        if (
            len(chunk.text)
            > self.config.max_quote_characters
        ):
            quote = quote.rstrip() + "…"

        return KnowledgeCitation(
            citation_id=new_id(),

            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,

            title=chunk.title,

            source_type=chunk.source_type,

            source_name=chunk.source_name,
            source_uri=chunk.source_uri,

            page=chunk.page,
            section=chunk.section,

            quote=quote,

            retrieval_score=chunk.score,
            authority_score=chunk.authority,

            evidence_strength=strength,

            version=chunk.version,
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def _retrieval_confidence(
        chunks: Sequence[KnowledgeChunk],
    ) -> float:

        if not chunks:
            return 0.0

        top = chunks[:5]

        weighted_scores: list[float] = []

        for index, chunk in enumerate(top):
            rank_weight = 1.0 / (
                1.0 + index * 0.20
            )

            evidence_score = (
                0.70 * chunk.score
                + 0.30 * chunk.authority
            )

            if chunk.approved:
                evidence_score += 0.05

            weighted_scores.append(
                clamp(evidence_score)
                * rank_weight
            )

        denominator = sum(
            1.0 / (1.0 + i * 0.20)
            for i in range(len(top))
        )

        score = (
            sum(weighted_scores)
            / denominator
        )

        source_diversity = len(
            {
                chunk.document_id
                for chunk in top
            }
        )

        if source_diversity >= 2:
            score += 0.03

        if source_diversity >= 3:
            score += 0.02

        return round(
            clamp(score),
            4,
        )

    # =====================================================
    # Deduplication
    # =====================================================

    @staticmethod
    def _deduplicate(
        chunks: Sequence[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:

        best: dict[
            tuple[str, str],
            KnowledgeChunk,
        ] = {}

        for chunk in chunks:
            key = (
                chunk.document_id,
                chunk.chunk_id,
            )

            current = best.get(key)

            if (
                current is None
                or chunk.score > current.score
            ):
                best[key] = chunk

        return sorted(
            best.values(),
            key=lambda item: (
                item.score,
                item.authority,
            ),
            reverse=True,
        )

    # =====================================================
    # Date helpers
    # =====================================================

    @staticmethod
    def _datetime(
        value: Any,
    ) -> datetime | None:

        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(
                    tzinfo=timezone.utc
                )

            return value

        if isinstance(value, date):
            return datetime(
                value.year,
                value.month,
                value.day,
                tzinfo=timezone.utc,
            )

        if isinstance(value, str):
            text = value.strip()

            if text.endswith("Z"):
                text = (
                    text[:-1]
                    + "+00:00"
                )

            try:
                parsed = datetime.fromisoformat(
                    text
                )

                if parsed.tzinfo is None:
                    parsed = parsed.replace(
                        tzinfo=timezone.utc
                    )

                return parsed

            except ValueError:
                return None

        return None

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # =====================================================
    # Observability
    # =====================================================

    @staticmethod
    def _record_metrics(
        result: KnowledgeSearchResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "knowledge_search_total"
            ).inc(
                labels={
                    "status":
                        result.status.value,

                    "method":
                        result
                        .retrieval_method.value,
                }
            )

            registry.histogram(
                "knowledge_retrieval_confidence"
            ).observe(
                result.confidence,

                labels={
                    "method":
                        result
                        .retrieval_method.value,
                },
            )

            registry.histogram(
                "knowledge_retrieval_results"
            ).observe(
                result.total_results
            )

        except Exception:
            pass


# =========================================================
# Singleton
# =========================================================

_default_knowledge_tools: KnowledgeTools | None = None


def configure_knowledge_tools(
    tools: KnowledgeTools,
) -> None:
    global _default_knowledge_tools

    _default_knowledge_tools = tools


def get_knowledge_tools() -> KnowledgeTools:
    if _default_knowledge_tools is None:
        raise RuntimeError(
            "KnowledgeTools is not configured. "
            "Call configure_knowledge_tools() "
            "during application startup."
        )

    return _default_knowledge_tools


# =========================================================
# Agent-facing tool functions
# =========================================================

async def search_knowledge(
    query: str,
    *,
    asset_id: str | None = None,
    asset_type: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    top_k: int = 8,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = await get_knowledge_tools().search(
        KnowledgeQuery(
            query=query,

            asset_id=asset_id,
            asset_type=asset_type,

            site_id=site_id,
            zone_id=zone_id,

            top_k=top_k,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


async def get_maintenance_guidance(
    *,
    asset_id: str,
    asset_type: str | None,
    failure_mode: str | None,
    symptoms: Sequence[str],
    site_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = (
        await get_knowledge_tools()
        .maintenance_guidance(
            asset_id=asset_id,
            asset_type=asset_type,

            failure_mode=failure_mode,
            symptoms=symptoms,

            site_id=site_id,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


async def get_failure_guidance(
    *,
    failure_mode: str,
    asset_id: str | None = None,
    asset_type: str | None = None,
    symptoms: Sequence[str] = (),
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = (
        await get_knowledge_tools()
        .failure_guidance(
            asset_id=asset_id,
            asset_type=asset_type,

            failure_mode=failure_mode,
            symptoms=symptoms,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


async def get_safety_procedure(
    *,
    hazard: str,
    asset_id: str | None = None,
    asset_type: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = (
        await get_knowledge_tools()
        .safety_procedure(
            hazard=hazard,

            asset_id=asset_id,
            asset_type=asset_type,

            site_id=site_id,
            zone_id=zone_id,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


async def get_inspection_guidance(
    *,
    defect: str,
    asset_id: str | None = None,
    asset_type: str | None = None,
    severity: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = (
        await get_knowledge_tools()
        .inspection_guidance(
            asset_id=asset_id,
            asset_type=asset_type,

            defect=defect,
            severity=severity,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


async def find_similar_incidents(
    *,
    description: str,
    asset_type: str | None = None,
    site_id: str | None = None,
    top_k: int = 5,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = (
        await get_knowledge_tools()
        .similar_incidents(
            description=description,

            asset_type=asset_type,
            site_id=site_id,

            top_k=top_k,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


async def get_claim_evidence(
    claim: str,
    *,
    asset_id: str | None = None,
    asset_type: str | None = None,
    site_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:

    result = (
        await get_knowledge_tools()
        .evidence_for_claim(
            claim=claim,

            asset_id=asset_id,
            asset_type=asset_type,
            site_id=site_id,

            user_id=user_id,
            tenant_id=tenant_id,
        )
    )

    return result.to_dict()


# =========================================================
# Exports
# =========================================================

__all__ = [
    "KnowledgeSourceType",
    "KnowledgeScope",
    "RetrievalMethod",
    "EvidenceStrength",
    "KnowledgeStatus",

    "KnowledgeQuery",
    "KnowledgeChunk",
    "KnowledgeCitation",
    "KnowledgeEvidence",
    "KnowledgeSearchResult",
    "GroundedKnowledgeAnswer",
    "SOPResult",

    "HybridRetriever",
    "VectorRetriever",
    "BM25Retriever",
    "GraphRetriever",
    "Reranker",
    "PermissionService",
    "KnowledgeRepository",

    "KnowledgeToolsConfig",
    "KnowledgeTools",

    "configure_knowledge_tools",
    "get_knowledge_tools",

    "search_knowledge",
    "get_maintenance_guidance",
    "get_failure_guidance",
    "get_safety_procedure",
    "get_inspection_guidance",
    "find_similar_incidents",
    "get_claim_evidence",
]