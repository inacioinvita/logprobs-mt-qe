"""Shared Brazilian Portuguese -> English sample for offline QE demos."""

from __future__ import annotations

SOURCE = (
    "A equipe de atendimento abriu um tíquete, mas o pedido ficou parado porque "
    "o fornecedor ainda não enviou a inscrição estadual corrigida nem confirmou "
    "se a cobrança do boleto seria estornada."
)

MARKER = "English translation:"

CANDIDATE_A = (
    "The customer service team opened a support ticket, but the request stalled "
    "because the supplier had still not sent the corrected state registration "
    "details or confirmed whether the boleto charge would be reversed."
)

CANDIDATE_B = (
    "The support team opened a ticket, but the request was delayed because the "
    "supplier had not yet sent the corrected registration or confirmed whether "
    "the payment would be refunded."
)

CANDIDATE_C = (
    "The attendance team opened a ticket, but the order stayed stopped because "
    "the supplier still had not sent the corrected state inscription nor "
    "confirmed if the boleto collection would be reversed."
)

CANDIDATE_D = (
    "The customer service team opened a complaint, but the order was cancelled "
    "because the supplier had not renewed its federal tax number or confirmed "
    "whether the bank transfer had failed."
)

CANDIDATE_SCORES = [
    ("A", CANDIDATE_A, -0.91, -1.58, 2.48, "best"),
    ("B", CANDIDATE_B, -1.05, -1.74, 2.86, "fluent but generic"),
    ("D", CANDIDATE_D, -1.64, -2.73, 5.16, "fluent but wrong"),
    ("C", CANDIDATE_C, -2.31, -4.08, 10.07, "literal and awkward"),
]


def _position(token: str, logprob: float, alternatives: list[tuple[str, float]] | None = None) -> dict:
    entries: dict[str, dict] = {
        str(abs(hash((token, logprob, 1))) % 1000000): {
            "rank": 1,
            "logprob": logprob,
            "decoded_token": token,
        }
    }
    for rank, (alt_token, alt_logprob) in enumerate(alternatives or [], start=2):
        entries[str(abs(hash((alt_token, alt_logprob, rank))) % 1000000)] = {
            "rank": rank,
            "logprob": alt_logprob,
            "decoded_token": alt_token,
        }
    return entries


TARGET_TOKENS = [
    (" The", -0.18, [(" A", -2.10), (" Customer", -3.20)]),
    (" customer", -0.34, [(" support", -1.16), (" client", -1.90)]),
    (" service", -0.22, [(" support", -0.95), (" care", -1.80)]),
    (" team", -0.20, [(" department", -1.45), (" staff", -2.00)]),
    (" opened", -0.28, [(" created", -0.90), (" filed", -1.60)]),
    (" a", -0.05, [(" the", -2.40), (" one", -3.10)]),
    (" support", -0.52, [(" service", -0.95), (" help", -1.70)]),
    (" ticket", -0.43, [(" case", -0.86), (" request", -1.20)]),
    (",", -0.03, [(".", -3.80), (";", -4.00)]),
    (" but", -0.10, [(" however", -1.60), (" and", -2.20)]),
    (" the", -0.07, [(" this", -2.40), (" its", -2.80)]),
    (" request", -0.86, [(" order", -1.03), (" case", -1.18)]),
    (" stalled", -0.78, [(" was delayed", -0.95), (" remained pending", -1.10)]),
    (" because", -0.08, [(" since", -1.80), (" as", -2.10)]),
    (" the", -0.05, [(" a", -2.70), (" its", -3.10)]),
    (" supplier", -0.21, [(" vendor", -1.12), (" provider", -1.70)]),
    (" had", -0.09, [(" has", -1.95), (" still", -2.20)]),
    (" still", -0.16, [(" not", -1.80), (" yet", -2.00)]),
    (" not", -0.06, [(" never", -2.70), ("n't", -3.00)]),
    (" sent", -0.24, [(" provided", -1.30), (" submitted", -1.55)]),
    (" the", -0.04, [(" its", -2.40), (" a", -2.90)]),
    (" corrected", -0.62, [(" updated", -1.30), (" revised", -1.45)]),
    (" state", -1.18, [(" tax", -1.24), (" registration", -1.55)]),
    (" registration", -1.32, [(" tax registration", -1.38), (" inscription", -1.90)]),
    (" details", -0.72, [(" information", -1.20), (" number", -1.70)]),
    (" or", -0.10, [(" nor", -1.60), (" and", -2.20)]),
    (" confirmed", -0.27, [(" said", -1.40), (" verified", -1.70)]),
    (" whether", -0.35, [(" if", -0.90), (" that", -2.10)]),
    (" the", -0.06, [(" this", -2.20), (" a", -2.80)]),
    (" boleto", -1.58, [(" payment slip", -1.62), (" payment", -1.75)]),
    (" charge", -1.10, [(" payment", -1.14), (" fee", -1.65)]),
    (" would", -0.42, [(" should", -1.20), (" will", -1.80)]),
    (" be", -0.18, [(" get", -2.30), (" have been", -2.60)]),
    (" reversed", -0.93, [(" refunded", -1.06), (" cancelled", -1.30)]),
    (".", -0.02, [("!", -4.20), ("", -5.00)]),
]

PROMPT_LOGPROBS = [
    None,
    _position("English", -0.01),
    _position(" translation", -0.01),
    _position(":", -0.01),
    *[_position(token, logprob, alternatives) for token, logprob, alternatives in TARGET_TOKENS],
]

WEAK_SPANS = [
    ("state registration details", -1.32),
    ("boleto charge", -1.58),
    ("would be reversed", -0.93),
]

AGREEMENT_CONSISTENT = [-0.91, -1.05, -1.12]
AGREEMENT_INCONSISTENT = [-0.91, -1.64, -2.31]
