"""Single chronological CERT pass; labels attached only after feature extraction."""
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import numpy as np
from benchmark.baseline_comparison import FeatureObservation
from core.models import MarkerVector
from extractors.graph_topology import DynamicCommunicationGraph
from extractors.llm_semantic import LLMSemanticExtractor
from extractors.spacy_statistical import SpacyStatisticalExtractor
from parsers.email_parser import EmailMessageParser
from parsers.role_directory import RoleDirectory
from profiling.sliding_window import SlidingTimeWindow
from profiling.workload_filter import TeamWorkloadContextTracker


def stratified_users(roster, per_role, seed):
    groups = defaultdict(list)
    for user, info in roster.items():
        groups[info.get("role", "unknown")].append(user)
    # Ordering depends only on identity, role and seed, never on incident labels.
    return {role: sorted(users, key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).digest())[:per_role]
            for role, users in sorted(groups.items())}


def validate_archive(archive):
    archive = Path(archive)
    required = [archive / "r4.2/email.csv", archive / "answers/insiders.csv", archive / "r4.2/LDAP"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("CERT r4.2 unavailable; required paths missing: " + ", ".join(missing))


def extract_cohort(archive, targets, semantic_mode="heuristic", seed=20261112,
                   include_references=True, message_sink=None):
    validate_archive(archive)
    archive = Path(archive)
    directory = RoleDirectory(archive / "r4.2/LDAP")
    reference_roster = directory.snapshots[0]
    pools = stratified_users(reference_roster, 5, seed)
    references = {user for group in pools.values() for user in group} if include_references else set()
    tracked = set(targets) | references
    windows = {u: SlidingTimeWindow(u) for u in tracked}
    graphs = {u: DynamicCommunicationGraph() for u in tracked}
    vectors, training_peers = defaultdict(list), defaultdict(list)
    activity = TeamWorkloadContextTracker()
    stats = SpacyStatisticalExtractor()
    start = previous = month = None
    roster, aliases = {}, {}
    count = 0
    with LLMSemanticExtractor(mode=semantic_mode) as semantic:
        if semantic_mode == "local_llm":
            semantic.verify_runtime()
        for msg in EmailMessageParser().stream_cert_emails(str(archive / "r4.2/email.csv")):
            ts = msg.timestamp
            if previous is not None and ts < previous:
                raise ValueError("CERT email rows are unordered; stable-sort a derived copy, never modify original CERT")
            previous = ts
            if start is None:
                start = ts
                train_end, calibration_end, test_start = (start + timedelta(days=d) for d in (30, 60, 74))
            current_month = ts.year, ts.month
            if current_month != month:
                roster = directory.at(ts)
                for user, info in roster.items():
                    address = info.get("email", "").lower()
                    if address:
                        aliases[address] = user
                        for graph in graphs.values():
                            graph.register_alias(user, address)
                month = current_month
            sender = (msg.sender_user_id or aliases.get(msg.sender_id, "")).lower()
            role = roster.get(sender, {}).get("role", "unknown")
            activity.register_role_activity(role, ts, employee_id=sender)
            involved = ({sender} | {aliases.get(address, address) for address in msg.receiver_ids}) & tracked
            if ts >= train_end:
                involved &= set(targets)
            for user in involved:
                window = windows[user]
                info = roster.get(user, {})
                window.role = info.get("role", "unknown")
                window.email = info.get("email", window.email).lower()
                window.add_message(msg)
                graphs[user].add_communication(msg)
            if sender not in involved:
                continue
            if message_sink is not None:
                message_sink(msg)
            context = graphs[sender].update_context(windows[sender].export_context(), include_betweenness=False)
            basic = stats.compute_markers(msg, context)
            sem = semantic.compute_markers(msg, context)
            vector = MarkerVector(**{m: getattr(sem if m.startswith("llm_") else basic, m) for m in MarkerVector.feature_names()},
                                  llm_uncertainty=sem.llm_uncertainty, unavailable=basic.unavailable | sem.unavailable,
                                  semantic_metadata=sem.semantic_metadata)
            if ts < train_end and sender in references:
                training_peers[(role, sender)].append(vector)
            if sender in targets:
                factor = activity.get_role_workload_factor(role, ts, exclude_user=sender)
                vectors[sender].append(FeatureObservation(ts, vector, -1, factor, msg.message_id or "", role=role))
            count += 1
            if count % 25000 == 0:
                print(f"CERT features extracted: {count}; date={ts.date()}", flush=True)
    if start is None:
        raise ValueError("No usable CERT messages")
    metadata = {"start": str(start), "train_end": str(train_end), "calibration_end": str(calibration_end),
                "test_start": str(test_start), "end": str(previous), "semantic_mode": semantic_mode, "seed": seed,
                "reference_selection": "five identity-hash-selected users per initial role; exclude target when fitting" if include_references else "none; experiment A without role priors only",
                "inference_scope": "within-person temporal monitoring, not held-out-user generalization",
                "timezone": "CERT naive clock tagged UTC for arithmetic; actual source zone unspecified",
                "workload": "completed calendar days; one-day lag"}
    metadata["semantic_runtime"] = semantic.deployment_provenance
    return vectors, training_peers, (train_end, calibration_end, test_start, previous), metadata


def independent_prior(training_peers, role, target):
    users = {u for (r, u), values in training_peers.items() if r == role and u != target and len(values) >= 2}
    if len(users) < 2:
        return {}, {"status": "unavailable", "reason": "fewer than two independent peers", "role": role}
    vectors = [v for user in sorted(users) for v in training_peers[(role, user)]]
    prior = {}
    for name in MarkerVector.feature_names():
        values = [getattr(v, name) for v in vectors if name not in v.unavailable]
        if len(values) >= 2:
            prior[name] = {"mean": float(np.mean(values)), "variance": max(float(np.var(values, ddof=1)), 1e-4)}
    return prior, {"status": "available", "role": role, "users": sorted(users), "observations": len(vectors)}


def prepare_user(user, features, peers, boundaries, incidents):
    from dataclasses import replace
    train_end, calibration_end, test_start, end = boundaries
    # Metadata used here for evaluation eligibility only, never passed to feature extraction or detector.
    all_obs = []
    for observation in features:
        positive = any(observation.observation_id in i.email_ids for i in incidents)
        inside = any(i.start <= observation.timestamp <= i.end for i in incidents)
        label = 1 if positive else -1 if inside else 0
        all_obs.append(replace(observation, label=label))
    training = [o for o in all_obs if o.timestamp < train_end]
    calibration = [o for o in all_obs if train_end <= o.timestamp < calibration_end]
    testing = [o for o in all_obs if o.timestamp >= test_start]
    if any(i.start < test_start for i in incidents):
        raise ValueError("not_evaluable: scenario overlaps fixed training/calibration; no certified pre-incident history")
    if len(training) < 2 or len(calibration) < 10 or not testing:
        raise ValueError("not_evaluable: insufficient independent train/calibration/test history")
    role = training[0].role
    prior, prior_info = independent_prior(peers, role, user)
    previous_role = role
    role_changes = []
    for o in testing:
        if o.role != previous_role:
            new_prior, info = independent_prior(peers, o.role, user)
            role_changes.append({"timestamp": str(o.timestamp), "from": previous_role, "to": o.role, **info})
            o.role_prior, o.role_confirmed = new_prior, bool(new_prior)
            previous_role = o.role
    return training, calibration, testing, prior, {"initial_prior": prior_info, "role_changes": role_changes}
