"""Pydantic request/response models for the Signal Engine API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AccountType(str, Enum):
    CHEQUING = "chequing"
    CREDIT_CARD = "credit_card"
    INVESTMENT_RRSP = "investment_rrsp"
    INVESTMENT_TFSA = "investment_tfsa"
    INVESTMENT_RESP = "investment_resp"
    INVESTMENT_NON_REG = "investment_non_reg"


class Channel(str, Enum):
    POS = "pos"
    ONLINE = "online"
    ETRANSFER = "etransfer"
    ACH = "ach"
    INTERNAL_TRANSFER = "internal_transfer"


class TransactionIn(BaseModel):
    txn_id: str
    timestamp: datetime
    account_type: AccountType
    amount: float
    merchant: str
    mcc: int
    mcc_category: str
    balance_after: float
    channel: Channel


class PredictRequest(BaseModel):
    user_id: str
    rrsp_room: float = Field(default=25000.0, description="Unused RRSP contribution room")
    transactions: list[TransactionIn]


class SpendingBuffer(BaseModel):
    liquid_cash: float
    monthly_burn_rate: float
    months_of_runway: float


class TargetProduct(BaseModel):
    code: str
    name: str
    projected_yield: str | None = None
    suggested_amount: float | None = None


class AuditEntry(BaseModel):
    feature: str
    value: float
    importance: float


class Traceability(BaseModel):
    spending_buffer: SpendingBuffer
    target_product: TargetProduct
    audit_log: list[AuditEntry]


class GovernanceTier(BaseModel):
    tier: str
    label: str
    reason: str
    workflow: str


class MacroContext(BaseModel):
    boc_prime_rate: float
    vix: float
    tsx_volatility: float
    rates_high: bool
    market_volatile: bool


class PersonaTier(str, Enum):
    ASPIRING_AFFLUENT = "aspiring_affluent"
    STICKY_FAMILY_LEADER = "sticky_family_leader"
    GENERATION_NERD = "generation_nerd"
    NOT_ELIGIBLE = "not_eligible"


class ReviewStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class SignalHypothesis(BaseModel):
    user_id: str
    persona_tier: PersonaTier
    signal: str
    confidence: float
    traceability: Traceability
    governance: GovernanceTier | None = None
    macro_context: MacroContext | None = None
    macro_reasons: list[str] = []
    nudge: str = ""
    feedback_reason: str | None = None
    staged_at: str
    status: ReviewStatus = ReviewStatus.PENDING


class PredictResponse(BaseModel):
    user_id: str
    persona_tier: str
    hypothesis: SignalHypothesis | None = None
    message: str = ""


class FeedbackRequest(BaseModel):
    user_id: str
    persona_tier: str
    signal: str
    product_code: str
    confidence: float
    governance_tier: str = ""
    action: str = Field(description="approved | rejected | pending")
    reason: str = ""


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    total_feedback: int = 0


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: list[str] = []
    version: str = "0.2.0"
