"""Necessary LLM prerequisite, deliberately not a claim of scientific validity."""


def llm_prerequisite(cert_manifest, probe):
    config = cert_manifest.get("config", {})
    reasons = []
    if config.get("semantic_mode") != "local_llm":
        reasons.append("CERT experiment did not use a real local LLM")
    runtime = config.get("semantic_runtime") or {}
    if not runtime.get("weight_sha256") or not runtime.get("receipt_sha256"):
        reasons.append("Dataset run lacks verified model/runtime provenance")
    if not probe.get("status", "").startswith("local_inference_observed") or not probe.get("samples"):
        reasons.append("Real local inference smoke test has not passed")
    for sample in probe.get("samples", []):
        meta = sample.get("metadata", {})
        if meta.get("mode") != "local_llm" or not meta.get("model_reported") or meta.get("model_reported") != meta.get("model_requested"):
            reasons.append("Probe contains missing/mixed model identity")
        provenance = meta.get("deployment") or {}
        if not provenance.get("weight_sha256") or provenance.get("weight_sha256") != runtime.get("weight_sha256"):
            reasons.append("Probe weights differ from dataset run or are unverified")
    return {"llm_prerequisite_met": not reasons, "reasons": sorted(set(reasons)),
            "scientific_validity_established": False,
            "note": "LLM presence alone does not validate marker quality, confidence or scientific conclusions"}
