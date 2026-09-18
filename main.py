from abc import ABC, abstractmethod
from collections import deque, Counter
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Tuple, Any
import networkx as nx
import math
import spacy
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


# ============================================================================
# 1. УРОВЕНЬ ПЕРВИЧНЫХ ДАННЫХ (DATA LAYER)
# ============================================================================

@dataclass
class CommunicationMessage:
    """Сущность единичного сообщения корпоративной коммуникации."""
    sender_id: str
    receiver_ids: List[str]
    timestamp: datetime
    channel: str = "email"
    subject: Optional[str] = None
    body: Optional[str] = None
    cleaned_body: Optional[str] = None

    def is_valid(self) -> bool:
        """Проверка валидности сообщения (наличие обязательных атрибутов)."""
        return bool(
            self.sender_id 
            and self.receiver_ids 
            and self.timestamp 
            and (self.cleaned_body or self.body)
        )


class MessageParser:
    """Компонент парсинга, нормализации и очистки корпуса сообщений."""
    # Регулярное выражение для извлечения канонического email или ID сотрудника
    EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+|[A-Za-z]{2,3}\d{4}')
    # Шаблоны служебных цитат и заголовков пересылки ответов
    QUOTE_PATTERNS = [
        re.compile(r'-{3,}\s*Original Message\s*-{3,}', re.IGNORECASE),
        re.compile(r'-{3,}\s*Forwarded by\s*-{3,}', re.IGNORECASE),
        re.compile(r'From:\s*.*?\nSent:\s*.*?\nTo:\s*', re.IGNORECASE),
    ]
    # Черный список отправителей (системные боты, рассылки, демоны доставки)
    SYSTEM_ACCOUNTS = {
        'mailer-daemon', 'postmaster', 'no-reply', 'noreply', 
        'notifications', 'calendar-notification', 'root'
    }

    def __init__(self, min_body_length: int = 5):
        self.min_body_length = min_body_length

    def normalize_email(self, raw_address: str) -> Optional[str]:
        """Нормализация адреса (извлечение чистой почты/ID, приведение к нижнему регистру)."""
        if not raw_address or pd.isna(raw_address):
            return None
        match = self.EMAIL_REGEX.search(str(raw_address))
        if match:
            return match.group(0).lower().strip()
        cleaned = str(raw_address).strip().lower()
        return cleaned if cleaned else None

    def parse_recipient_list(self, raw_recipients: Any) -> List[str]:
        """Парсинг списка получателей (To, Cc, Bcc), разделённых ';' или ','."""
        if not raw_recipients or pd.isna(raw_recipients):
            return []
        raw_list = re.split(r'[;,]', str(raw_recipients))
        valid_recipients = []
        for r in raw_list:
            norm = self.normalize_email(r)
            if norm and norm not in valid_recipients:
                valid_recipients.append(norm)
        return valid_recipients

    def clean_quoted_text(self, body: str) -> str:
        """Очистка тела письма от цитируемых цепочек ответов (reply-stripping)."""
        if not body or pd.isna(body):
            return ""
        text = str(body)
        # 1. Отсечение по сигнатурам цитирования/пересылки
        for pattern in self.QUOTE_PATTERNS:
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
        # 2. Построчная фильтрация строк цитирования '>'
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith('>'):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def parse_timestamp(self, raw_ts: Any) -> Optional[datetime]:
        """Преобразование временной метки в datetime с фиксацией часового пояса UTC."""
        if pd.isna(raw_ts) or not raw_ts:
            return None
        try:
            dt = pd.to_datetime(raw_ts, errors='coerce')
            if pd.isna(dt):
                return None
            if dt.tzinfo is None:
                dt = dt.tz_localize(timezone.utc)
            else:
                dt = dt.tz_convert(timezone.utc)
            return dt.to_pydatetime()
        except Exception:
            return None

    def parse_email(self, raw_content: str) -> Optional[CommunicationMessage]:
        """Парсинг сырого MIME/RFC 822 сообщения в объект CommunicationMessage."""
        if not raw_content or pd.isna(raw_content):
            return None
        lines = str(raw_content).splitlines()
        headers: Dict[str, str] = {}
        current_header = None
        body_start = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "":
                body_start = i + 1
                break
            if line.startswith((" ", "\t")) and current_header:
                headers[current_header] += " " + line.strip()
            elif ":" in line:
                key, val = line.split(":", 1)
                current_header = key.strip().lower()
                headers[current_header] = val.strip()
        sender = self.normalize_email(headers.get("from", ""))
        receivers = self.parse_recipient_list(headers.get("to", ""))
        ts = self.parse_timestamp(headers.get("date"))
        subject = headers.get("subject", "")
        raw_body = "\n".join(lines[body_start:]).strip()
        cleaned_body = self.clean_quoted_text(raw_body)
        msg = CommunicationMessage(
            sender_id=sender or "",
            receiver_ids=receivers,
            timestamp=ts,
            channel="email",
            subject=subject,
            body=raw_body,
            cleaned_body=cleaned_body
        )
        return msg if msg.is_valid() else None

    def parse_cert_row(self, row: Dict[str, Any]) -> Optional[CommunicationMessage]:
        """Парсинг строки из датасета CERT r4.2 (email.csv)."""
        sender = self.normalize_email(row.get("from") or row.get("user"))
        if not sender or sender in self.SYSTEM_ACCOUNTS:
            return None
        to_list = self.parse_recipient_list(row.get("to"))
        cc_list = self.parse_recipient_list(row.get("cc"))
        bcc_list = self.parse_recipient_list(row.get("bcc"))
        all_receivers = list(dict.fromkeys(to_list + cc_list + bcc_list))
        if not all_receivers:
            return None
        ts = self.parse_timestamp(row.get("date"))
        if not ts:
            return None
        content = str(row.get("content", ""))
        cleaned = self.clean_quoted_text(content)
        if len(cleaned) < self.min_body_length:
            return None
        return CommunicationMessage(
            sender_id=sender,
            receiver_ids=all_receivers,
            timestamp=ts,
            channel="email",
            subject=None,
            body=content,
            cleaned_body=cleaned
        )

    def filter_valid(self, messages: List[CommunicationMessage]) -> List[CommunicationMessage]:
        """Фильтрация служебных, системных и повреждённых сообщений."""
        return [
            msg for msg in messages 
            if msg and msg.is_valid() 
            and msg.sender_id not in self.SYSTEM_ACCOUNTS
            and len(msg.cleaned_body or "") >= self.min_body_length
        ]

    def stream_cert_emails(self, file_path: str, chunk_size: int = 50000) -> Iterator[CommunicationMessage]:
        """Потоковый генератор для CERT r4.2 email.csv без переполнения памяти."""
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str):
            for _, row in chunk.iterrows():
                msg = self.parse_cert_row(row.to_dict())
                if msg and msg.is_valid():
                    yield msg


# ============================================================================
# 2. УРОВЕНЬ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (FEATURE EXTRACTION LAYER)
# ============================================================================

@dataclass 
class EmployeeContext:
    """Контекстная информация о сотруднике, передаваемая в анализатор."""
    employee_id: str
    role: str
    in_degree: float
    out_degree: float
    reciprocity: float
    betweenness: float
    mean_reply_time: float
    median_reply_time: float
    reply_time_std: float
    daily_activity_change: float
    weekly_activity_change: float
    monthly_activity_change: float
    quarterly_activity_change: float


@dataclass
class MarkerVector:
    """Вектор извлечённых психолингвистических и семантических маркеров."""
    pronoun_marker: float = 0.0
    syntactic_marker: float = 0.0
    activity_marker: float = 0.0
    response_marker: float = 0.0
    social_marker: float = 0.0
    # Семантические маркеры (контур LLM)
    llm_aggression_marker: float = 0.0
    llm_disloyalty_marker: float = 0.0
    llm_burnout_marker: float = 0.0


class IFeatureExtractor(ABC):
    """Интерфейс модуля извлечения психолингвистических признаков и маркеров."""

    @abstractmethod
    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        """Извлечение признаков и расчёт единого вектора маркеров риска."""
        pass


class SpacyStatisticalExtractor(IFeatureExtractor):
    """
    Статистико-лингвистический анализатор на базе библиотеки spaCy.
    Реализует математические формулы индикаторов (10)-(14) НИР2.
    """
    # Дискурсивные связки (маркеры логических разрывов и контраста)
    DISCOURSE_MARKERS = {
        "however", "therefore", "thus", "although", "because", 
        "meanwhile", "moreover", "nevertheless", "furthermore"
    }
    # Словари местоимений английского языка
    FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
    SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "yourself", "yourselves"}
    THIRD_PERSON_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", 
                             "they", "them", "their", "theirs", "themselves", "it", "its"}

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            # Если языковая модель не скачана в окружении
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)

    def _extract_pronoun_marker(self, doc) -> float:
        """
        Формула (10) НИР2:
        I_pro = - (p1*log2(p1) + p2*log2(p2) + p3*log2(p3))
        pk = (nk + alpha) / (N + alpha * K), где alpha = 1.0, K = 3.0
        """
        pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
        total_pronouns = len(pronouns)
        n1 = sum(p in self.FIRST_PERSON_PRONOUNS for p in pronouns)
        n2 = sum(p in self.SECOND_PERSON_PRONOUNS for p in pronouns)
        n3 = sum(p in self.THIRD_PERSON_PRONOUNS for p in pronouns)
        alpha = 1.0
        k = 3.0
        denom = float(total_pronouns + alpha * k)
        p1 = (n1 + alpha) / denom
        p2 = (n2 + alpha) / denom
        p3 = (n3 + alpha) / denom
        # Вычисление энтропии распределения местоимений
        entropy = - (p1 * math.log2(p1) + p2 * math.log2(p2) + p3 * math.log2(p3))
        return float(entropy)

    def _extract_syntactic_marker(self, doc, text: str) -> float:
        """
        Формула (11) НИР2:
        I_syn = sqrt( (ln(1 + l_sent))^2 + (ln(1 + d_clause))^2 + (p_disc)^2 + (p_punct)^2 )
        """
        # 1. Средняя длина предложений
        sentences = list(doc.sents)
        lengths = [len(sent) for sent in sentences]
        avg_sent_len = float(np.mean(lengths)) if lengths else 0.0
        # 2. Глубина синтаксического дерева зависимостей
        depths = [len(list(token.ancestors)) for token in doc]
        dep_depth = float(np.mean(depths)) if depths else 0.0
        # 3. Доля дискурсивных связок
        words = [token.text.lower() for token in doc if token.is_alpha]
        disc_count = sum(w in self.DISCOURSE_MARKERS for w in words)
        p_disc = float(disc_count / max(len(words), 1))
        # 4. Частота преобладающих триграмм PoS-тегов
        pos_tags = [token.pos_ for token in doc]
        trigrams = [(pos_tags[i], pos_tags[i+1], pos_tags[i+2]) for i in range(len(pos_tags) - 2)]
        if trigrams:
            p_punct = float(Counter(trigrams).most_common(1)[0][1] / len(trigrams))
        else:
            p_punct = 0.0
        # Свертка по формуле (11)
        val = math.sqrt(
            math.log1p(avg_sent_len)**2 + 
            math.log1p(dep_depth)**2 + 
            p_disc**2 + 
            p_punct**2
        )
        return float(val)

    def _extract_social_marker(self, ctx: EmployeeContext) -> float:
        """
        Формула (12) НИР2:
        I_soc = sqrt( (ln(1 + deg_in))^2 + (ln(1 + deg_out))^2 + (rec)^2 + (bet)^2 )
        """
        val = math.sqrt(
            math.log1p(max(ctx.in_degree, 0.0))**2 +
            math.log1p(max(ctx.out_degree, 0.0))**2 +
            (ctx.reciprocity)**2 +
            (ctx.betweenness)**2
        )
        return float(val)

    def _extract_activity_marker(self, ctx: EmployeeContext) -> float:
        """
        Формула (13) НИР2:
        I_act = sqrt( (tanh(dH_day)^2 + tanh(dH_week)^2 + tanh(dH_month)^2 + tanh(dH_quart)^2) / 4 )
        """
        d_day = math.tanh(ctx.daily_activity_change)
        d_week = math.tanh(ctx.weekly_activity_change)
        d_month = math.tanh(ctx.monthly_activity_change)
        d_quart = math.tanh(ctx.quarterly_activity_change)

        val = math.sqrt((d_day**2 + d_week**2 + d_month**2 + d_quart**2) / 4.0)
        return float(val)

    def _extract_response_marker(self, ctx: EmployeeContext) -> float:
        """
        Формула (14) НИР2:
        I_lat = sqrt( (ln(1 + mu))^2 + (ln(1 + med))^2 + (ln(1 + sigma))^2 )
        """
        val = math.sqrt(
            math.log1p(max(ctx.mean_reply_time, 0.0))**2 +
            math.log1p(max(ctx.median_reply_time, 0.0))**2 +
            math.log1p(max(ctx.reply_time_std, 0.0))**2
        )
        return float(val)

    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        """Комплексное вычисление формульного вектора маркеров риска."""
        text = message.cleaned_body or message.body or ""
        doc = self.nlp(text)
        pronoun_m = self._extract_pronoun_marker(doc)
        syntactic_m = self._extract_syntactic_marker(doc, text)
        social_m = self._extract_social_marker(context)
        activity_m = self._extract_activity_marker(context)
        response_m = self._extract_response_marker(context)
        return MarkerVector(
            pronoun_marker=pronoun_m,
            syntactic_marker=syntactic_m,
            activity_marker=activity_m,
            response_marker=response_m,
            social_marker=social_m,
            llm_aggression_marker=0.0,
            llm_disloyalty_marker=0.0,
            llm_burnout_marker=0.0
        )


class LLMSemanticExtractor(IFeatureExtractor):
    """
    Семантический анализатор скрытых признаков угрозы на основе LLM (задача Главы 1).
    Извлекает семантические паттерны: агрессия, нелояльность, выгорание.
    """
    SYSTEM_PROMPT = (
        "You are an expert psycholinguistic and insider threat detection AI system. "
        "Analyze the corporate communication message and rate the following risk factors "
        "on a continuous scale from 0.0 (completely absent/normal) to 1.0 (critically high):\n"
        "1. aggression: hostility, toxic tone, threats, passive aggression.\n"
        "2. disloyalty: sabotage intent, negative attitude towards company/management, data exfiltration signs.\n"
        "3. burnout: emotional exhaustion, apathy, severe stress, detachment.\n"
        "Respond ONLY with a valid JSON object: {\"aggression\": float, \"disloyalty\": float, \"burnout\": float}."
    )

    def __init__(self, model_name: str = "llama3-8b", api_endpoint: Optional[str] = None):
        self.model_name = model_name
        self.api_endpoint = api_endpoint

    def generate_prompt(self, message: CommunicationMessage) -> str:
        """Формирование типизированного промпта со структурированным выходом (JSON)."""
        content = message.cleaned_body or message.body or ""
        return f"Message Subject: {message.subject or 'None'}\nBody:\n{content}"

    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        """
        Отправка запроса в LLM и парсинг числовых оценок в вектор маркеров.
        При отсутствии активного подключения возвращает безопасные нули (fallback).
        """
        # TODO Заглушка/каркас сетевого вызова к инференс-сервису (vLLM / Ollama / OpenAI API)
        # В Главе 1 здесь настраивается реальный HTTP-коннектор
        aggression = 0.0
        disloyalty = 0.0
        burnout = 0.0
        if self.api_endpoint:
            try:
                # TODO: HTTP POST на vLLM/Ollama
                pass
            except Exception:
                pass
        return MarkerVector(
            pronoun_marker=0.0,
            syntactic_marker=0.0,
            activity_marker=0.0,
            response_marker=0.0,
            social_marker=0.0,
            llm_aggression_marker=float(aggression),
            llm_disloyalty_marker=float(disloyalty),
            llm_burnout_marker=float(burnout)
        )


# ============================================================================
# 3. УРОВЕНЬ ПРОФИЛИРОВАНИЯ И СОСТОЯНИЯ (STATE & PROFILING LAYER)
# ============================================================================

class SlidingTimeWindow:
    """
    Скользящее временное окно для поточно-корректного учёта активности.
    Инкапсулирует двустороннюю очередь временных меток и гарантирует
    отсутствие заглядывания в будущее за счёт строгого отсечения событий t < (current_time - duration).
    """

    def __init__(self, window_duration_days: int):
        self.window_duration = timedelta(days=window_duration_days)
        self.events_queue: deque[datetime] = deque()

    def push_event(self, timestamp: datetime) -> None:
        """
        Добавление временной метки нового события в конец очереди.
        Предполагается строгая монотонность времени в потоке.
        """
        if not isinstance(timestamp, datetime):
            return
        self.events_queue.append(timestamp)

    def evict_expired(self, current_time: datetime) -> int:
        """
        Удаление событий, вышедших за левую границу скользящего окна.
        Событие считается устаревшим, если: t < current_time - window_duration.
        
        Сложность: O(K), где K — число устаревших событий, так как удаление
        из начала deque (popleft) выполняется за O(1).
        
        Возвращает:
            Количество актуальных событий, оставшихся в активном интервале окна.
        """
        if not current_time or not isinstance(current_time, datetime):
            return len(self.events_queue)
        threshold = current_time - self.window_duration
        # Выталкиваем из головы очереди все события, произошедшие ранее порогового времени
        while self.events_queue and self.events_queue[0] < threshold:
            self.events_queue.popleft()
        return len(self.events_queue)

    def get_count(self) -> int:
        """
        Текущее число событий внутри активного интервала окна.
        """
        return len(self.events_queue)


class BaselineProfile:
    """
    Динамический базовый профиль сотрудника (алгоритм EMA).
    Реализует математическую модель динамического профилирования
    (формулы 18, 21-22 отчёта НИР2) и сохраняет историю для визуализации.
    """

    TRACKED_MARKERS = [
        "pronoun_marker",
        "syntactic_marker",
        "activity_marker",
        "response_marker",
        "social_marker",
        "llm_aggression_marker",
        "llm_disloyalty_marker",
        "llm_burnout_marker",
    ]

    def __init__(
        self,
        alpha_ema: float = 0.1,
        min_variance: float = 1e-4,
        clip_limit: float = 10.0,
        warmup_threshold: int = 10,
        role_priors: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.alpha_ema = alpha_ema
        self.min_variance = min_variance
        self.clip_limit = clip_limit
        self.warmup_threshold = warmup_threshold
        self.role_priors = role_priors or {}

        self.observations_count: int = 0
        self.means: Dict[str, float] = {}
        self.variances: Dict[str, float] = {}

        # История для графиков эволюции профиля и отклонений
        self.means_history: Dict[str, List[float]] = {m: [] for m in self.TRACKED_MARKERS}
        self.deviations_history: Dict[str, List[float]] = {m: [] for m in self.TRACKED_MARKERS}

        for marker in self.TRACKED_MARKERS:
            prior = self.role_priors.get(marker, {})
            self.means[marker] = float(prior.get("mean", 0.0))
            self.variances[marker] = float(prior.get("variance", 0.0))

    def _get_marker_dict(self, markers: MarkerVector) -> Dict[str, float]:
        data = asdict(markers) if hasattr(markers, "__dataclass_fields__") else dict(markers)
        clean_dict = {}
        for m in self.TRACKED_MARKERS:
            val = data.get(m, 0.0)
            clean_dict[m] = float(val) if val is not None and not math.isnan(val) else 0.0
        return clean_dict

    def compute_deviations(self, markers: MarkerVector) -> Dict[str, float]:
        """Расчёт z-score ДО обновления профиля и фиксация в истории."""
        marker_vals = self._get_marker_dict(markers)
        z_scores: Dict[str, float] = {}

        if self.observations_count == 0 and not self.role_priors:
            neutral = {m: 0.0 for m in self.TRACKED_MARKERS}
            for m in self.TRACKED_MARKERS:
                self.deviations_history[m].append(0.0)
            return neutral

        for m in self.TRACKED_MARKERS:
            val = marker_vals[m]
            mean = self.means[m]
            var = self.variances[m]

            safe_var = max(var, self.min_variance)
            std = math.sqrt(safe_var)

            raw_z = (val - mean) / std
            z_clipped = float(max(-self.clip_limit, min(self.clip_limit, raw_z)))
            z_scores[m] = z_clipped
            self.deviations_history[m].append(z_clipped)

        return z_scores

    def update(self, markers: MarkerVector) -> None:
        """Инкрементное обновление EMA и сохранение значений средних в историю."""
        marker_vals = self._get_marker_dict(markers)

        if self.observations_count == 0 and not self.role_priors:
            for m in self.TRACKED_MARKERS:
                self.means[m] = marker_vals[m]
                self.variances[m] = self.min_variance
                self.means_history[m].append(marker_vals[m])
            self.observations_count = 1
            return

        for m in self.TRACKED_MARKERS:
            val = marker_vals[m]
            prev_mean = self.means[m]
            prev_var = self.variances[m]

            diff = val - prev_mean
            new_mean = prev_mean + self.alpha_ema * diff
            new_var = (1.0 - self.alpha_ema) * (prev_var + self.alpha_ema * (diff ** 2))

            self.means[m] = float(new_mean)
            self.variances[m] = float(new_var)
            self.means_history[m].append(float(new_mean))

        self.observations_count += 1

    def get_mean(self, marker_name: str) -> float:
        return self.means.get(marker_name, 0.0)

    def get_std(self, marker_name: str) -> float:
        var = self.variances.get(marker_name, self.min_variance)
        return math.sqrt(max(var, self.min_variance))


class EmployeeState:
    """
    Агрегированное состояние сотрудника в модельном времени потока.
    Инкапсулирует иерархию скользящих временных окон (1, 7, 30, 90 дней),
    буфер задержек отклика и индивидуальный динамический базовый профиль.
    """

    def __init__(self, employee_id: str, role: str = "general"):
        self.employee_id = employee_id
        self.role = role
        self.first_seen: Optional[datetime] = None
        self.last_seen: Optional[datetime] = None
        self.total_messages_sent: int = 0
        # Иерархия скользящих временных окон (сутки, неделя, месяц, квартал)
        self.daily_window = SlidingTimeWindow(window_duration_days=1)
        self.weekly_window = SlidingTimeWindow(window_duration_days=7)
        self.monthly_window = SlidingTimeWindow(window_duration_days=30)
        self.quarterly_window = SlidingTimeWindow(window_duration_days=90)
        # Буфер задержек ответов (в секундах) на входящие сообщения
        self.reply_delays: deque[float] = deque(maxlen=100)
        # Динамический базовый профиль маркеров риска (EMA)
        self.baseline_profile = BaselineProfile()

    def update_temporal_state(self, timestamp: datetime) -> None:
        """
        Сдвиг временных окон и регистрация нового исходящего события.
        1. Фиксирует границы первого и последнего появления сотрудника.
        2. Очищает все 4 окна от событий, вышедших за левую границу t < timestamp - duration.
        3. Регистрирует новое событие в очередях всех окон.
        """
        if not isinstance(timestamp, datetime):
            return
        if self.first_seen is None:
            self.first_seen = timestamp
        self.last_seen = timestamp
        self.total_messages_sent += 1
        # Очистка устаревших событий по текущему модельному времени
        self.daily_window.evict_expired(timestamp)
        self.weekly_window.evict_expired(timestamp)
        self.monthly_window.evict_expired(timestamp)
        self.quarterly_window.evict_expired(timestamp)
        # Добавление текущего события во все скользящие окна
        self.daily_window.push_event(timestamp)
        self.weekly_window.push_event(timestamp)
        self.monthly_window.push_event(timestamp)
        self.quarterly_window.push_event(timestamp)

    def register_reply_delay(self, delay_seconds: float) -> None:
        """
        Фиксация времени ответа (в секундах) на входящее сообщение.
        Задержки меньше 0 или аномалии парсинга отсекаются.
        """
        if delay_seconds is not None and delay_seconds > 0:
            self.reply_delays.append(float(delay_seconds))

    def get_activity_dynamics(self) -> Tuple[float, float, float, float]:
        """
        Расчёт относительных изменений активности (день, неделя, месяц, квартал)
        для формулы маркера активности (13) НИР2:
            Delta H = (N_current - N_expected) / max(N_expected, 1)
        
        Ожидаемая плотность рассчитывается через более широкое окно без заглядывания в будущее:
        - daily_change: суточная активность относительно среднего за неделю (N_week / 7).
        - weekly_change: недельная активность относительно среднего за месяц (N_month / 4.28).
        - monthly_change: месячная активность относительно среднего за квартал (N_quarter / 3.0).
        - quarterly_change: квартальная активность относительно темпа за всю историю наблюдений.
        """
        n_day = float(self.daily_window.get_count())
        n_week = float(self.weekly_window.get_count())
        n_month = float(self.monthly_window.get_count())
        n_quarter = float(self.quarterly_window.get_count())
        # 1. Суточный темп относительно недельной нормы
        expected_daily = n_week / 7.0
        daily_change = (n_day - expected_daily) / max(expected_daily, 1.0)
        # 2. Недельный темп относительно месячной нормы (~30/7 = 4.285)
        expected_weekly = n_month / 4.2857
        weekly_change = (n_week - expected_weekly) / max(expected_weekly, 1.0)
        # 3. Месячный темп относительно квартальной нормы (90/30 = 3.0)
        expected_monthly = n_quarter / 3.0
        monthly_change = (n_month - expected_monthly) / max(expected_monthly, 1.0)
        # 4. Квартальный темп относительно общего стажа наблюдений
        if self.first_seen and self.last_seen:
            history_days = max((self.last_seen - self.first_seen).total_seconds() / 86400.0, 1.0)
            expected_quarterly = (self.total_messages_sent / history_days) * 90.0
        else:
            expected_quarterly = n_quarter
        quarterly_change = (n_quarter - expected_quarterly) / max(expected_quarterly, 1.0)
        return (
            float(daily_change),
            float(weekly_change),
            float(monthly_change),
            float(quarterly_change),
        )

    def get_reply_statistics(self) -> Tuple[float, float, float]:
        """
        Вычисление статистических характеристик скорости ответа (mean, median, std)
        для формулы маркера задержек отклика (14) НИР2.
        
        Возвращает:
            (mean_delay, median_delay, std_delay) в секундах.
            Если данных нет — возвращаются нули (нейтральный сигнал).
        """
        if not self.reply_delays:
            return (0.0, 0.0, 0.0)
        arr = np.array(self.reply_delays, dtype=np.float64)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
        return (mean_val, median_val, std_val)


# ============================================================================
# 4. УРОВЕНЬ ТОПОЛОГИИ КОММУНИКАЦИЙ (TOPOLOGY LAYER)
# ============================================================================

class DynamicCommunicationGraph:
    """
    Динамический граф корпоративного взаимодействия со скользящим окном.
    Поддерживает актуальную структуру топологии в окне window_days,
    автоматически выталкивая устаревшие ребра при продвижении времени потока.
    """

    def __init__(self, window_days: int = 30, max_betweenness_nodes: int = 500):
        """
        Параметры:
            window_days: протяжённость окна актуальности рёбер (по умолчанию 30 дней).
            max_betweenness_nodes: порог размера подграфа для полного расчёта посредничества.
        """
        self.window_duration = timedelta(days=window_days)
        # Очередь событий взаимодействия: (timestamp, sender, receiver)
        self.edge_history: deque[Tuple[datetime, str, str]] = deque()
        # Взвешенный ориентированный граф NetworkX
        self.graph: nx.DiGraph = nx.DiGraph()
        self.max_betweenness_nodes = max_betweenness_nodes

    def add_interaction(self, sender: str, receiver: str, timestamp: datetime) -> None:
        """
        Добавление направленного факта коммуникации (sender -> receiver) в момент timestamp.
        1. Регистрирует событие в хронологической очереди edge_history.
        2. Добавляет или инкрементирует вес ребра в графе.
        """
        if not sender or not receiver or sender == receiver:
            return
        self.edge_history.append((timestamp, sender, receiver))
        if self.graph.has_edge(sender, receiver):
            self.graph[sender][receiver]["weight"] += 1.0
        else:
            self.graph.add_edge(sender, receiver, weight=1.0)

    def evict_expired(self, current_time: datetime) -> None:
        """
        Удаление взаимодействий, произошедших ранее порога (current_time - window_duration).
        Уменьшает вес соответствующих рёбер; при падении веса до 0 ребро удаляется.
        """
        if not current_time or not isinstance(current_time, datetime):
            return
        threshold = current_time - self.window_duration
        while self.edge_history and self.edge_history[0][0] < threshold:
            _, u, v = self.edge_history.popleft()
            if self.graph.has_edge(u, v):
                self.graph[u][v]["weight"] -= 1.0
                if self.graph[u][v]["weight"] <= 0:
                    self.graph.remove_edge(u, v)
                    # Если узел стал изолированным — удаляем его для экономии памяти
                    if self.graph.in_degree(u) == 0 and self.graph.out_degree(u) == 0:
                        self.graph.remove_node(u)
                    if self.graph.has_node(v) and self.graph.in_degree(v) == 0 and self.graph.out_degree(v) == 0:
                        self.graph.remove_node(v)

    def get_node_metrics(self, node_id: str) -> Tuple[float, float, float, float]:
        """
        Вычисление топологических метрик для узла на текущий момент модельного времени.
        
        Возвращает:
            (out_degree, in_degree, reciprocity, betweenness)
            - out_degree: взвешенная/невзвешенная исходящая степень узла.
            - in_degree: входящая степень узла.
            - reciprocity: коэффициент взаимности общения (доля взаимных связей).
            - betweenness: посредническая центральность в активной компоненте связности.
        """
        if not self.graph.has_node(node_id):
            return (0.0, 0.0, 0.0, 0.0)
        # 1. Степени узла
        out_deg = float(self.graph.out_degree(node_id))
        in_deg = float(self.graph.in_degree(node_id))
        # 2. Локальная взаимность (reciprocity) узла
        # Доля партнеров, с которыми есть двустороннее общение
        successors = set(self.graph.successors(node_id))
        predecessors = set(self.graph.predecessors(node_id))
        mutual_contacts = len(successors.intersection(predecessors))
        total_unique_contacts = len(successors.union(predecessors))
        reciprocity = float(mutual_contacts / total_unique_contacts) if total_unique_contacts > 0 else 0.0
        # 3. Посредническая центральность (betweenness centrality)
        # Вычисляется по эго-сети (окрестности радиуса 2) для обеспечения производительности O(k) вместо O(V*E)
        betweenness = 0.0
        try:
            # Ограничиваем расчёт локальной окрестностью узла
            ego_nodes = set(nx.ego_graph(self.graph, node_id, radius=2).nodes())
            if len(ego_nodes) > 2:
                subgraph = self.graph.subgraph(ego_nodes)
                b_dict = nx.betweenness_centrality(subgraph, normalized=True)
                betweenness = float(b_dict.get(node_id, 0.0))
        except Exception:
            betweenness = 0.0
        return (out_deg, in_deg, reciprocity, betweenness)  


# ============================================================================
# 5. УРОВЕНЬ ПОТОКОВОЙ ДИСПЕТЧЕРИЗАЦИИ И РЕЗУЛЬТАТОВ (STREAMING CORE)
# ============================================================================

@dataclass
class RiskSignal:
    """Выходная результирующая структура скрининга для аналитика ИБ."""
    employee_id: str
    timestamp: datetime
    raw_markers: MarkerVector
    normalized_deviations: Dict[str, float]
    channel: str = "email"
    explanations: List[str] = field(default_factory=list)


class StreamProcessor:
    """Главный диспетчер хронологического потока событий информационной среды."""

    def __init__(self, feature_extractor: IFeatureExtractor, topology_window_days: int = 30):
        self.current_time: Optional[datetime] = None
        self.employees: Dict[str, EmployeeState] = {}
        self.graph = DynamicCommunicationGraph(window_days=topology_window_days)
        self.feature_extractor = feature_extractor
        # Хранилище последних исходящих меток для расчёта задержек ответов: (receiver, sender) -> timestamp
        self._last_direct_message: Dict[Tuple[str, str], datetime] = {}

    def get_or_create_employee(self, employee_id: str, role: str = "general") -> EmployeeState:
        """Получение существующего или инициализация нового состояния сотрудника."""
        if employee_id not in self.employees:
            self.employees[employee_id] = EmployeeState(employee_id=employee_id, role=role)
        return self.employees[employee_id]

    def _track_reply_delays(self, sender: str, receivers: List[str], timestamp: datetime, sender_state: EmployeeState) -> None:
        """
        Фиксация задержек ответов на входящие сообщения.
        Если кто-то из получателей ранее писал отправителю, фиксируется время отклика.
        """
        for receiver in receivers:
            reverse_pair = (receiver, sender)
            if reverse_pair in self._last_direct_message:
                prev_time = self._last_direct_message[reverse_pair]
                delta_sec = (timestamp - prev_time).total_seconds()
                if delta_sec > 0:
                    sender_state.register_reply_delay(delta_sec)
            # Обновляем отметку прямого сообщения
            self._last_direct_message[(sender, receiver)] = timestamp

    def process_message(self, message: CommunicationMessage) -> RiskSignal:
        """
        Обработка единичного входящего события в строгом модельном времени:
        1. Проверка монотонности времени (t >= current_time).
        2. Продвижение скользящих окон графа и отправителя.
        3. Формирование контекста и извлечение маркеров.
        4. Расчёт z-score отклонений по текущему базовому профилю (ДО обновления).
        5. Инкрементное обновление профиля (EMA).
        6. Генерация выходного сигнала риска с массивом объяснений.
        """
        ts = message.timestamp
        # 1. Проверка монотонности модельного времени
        if self.current_time is not None and ts < self.current_time:
            # Предупреждение о нарушении хронологии (out-of-order event)
            pass
        self.current_time = ts
        # 2. Получение и синхронизация состояния автора сообщения
        sender = message.sender_id
        sender_state = self.get_or_create_employee(sender)
        sender_state.update_temporal_state(ts)
        # Регистрация задержек отклика собеседникам
        self._track_reply_delays(sender, message.receiver_ids, ts, sender_state)
        # 3. Синхронизация динамического графа взаимодействий
        self.graph.evict_expired(ts)
        for receiver in message.receiver_ids:
            self.get_or_create_employee(receiver)  # Инициализируем профиль адресата, если новый
            self.graph.add_interaction(sender, receiver, ts)
        # 4. Сборка топологического и динамического контекста (EmployeeContext)
        out_deg, in_deg, recip, betw = self.graph.get_node_metrics(sender)
        d_day, d_week, d_month, d_quart = sender_state.get_activity_dynamics()
        mean_rep, med_rep, std_rep = sender_state.get_reply_statistics()
        context = EmployeeContext(
            employee_id=sender,
            role=sender_state.role,
            in_degree=in_deg,
            out_degree=out_deg,
            reciprocity=recip,
            betweenness=betw,
            mean_reply_time=mean_rep,
            median_reply_time=med_rep,
            reply_time_std=std_rep,
            daily_activity_change=d_day,
            weekly_activity_change=d_week,
            monthly_activity_change=d_month,
            quarterly_activity_change=d_quart
        )
        # 5. Извлечение признаков и вычисление вектора маркеров
        raw_markers = self.feature_extractor.compute_markers(message, context)
        # 6. Расчёт z-score стандартизированных отклонений ДО обновления профиля
        z_scores = sender_state.baseline_profile.compute_deviations(raw_markers)
        # 7. Инкрементное обновление базового профиля автора новым наблюдением (EMA)
        sender_state.baseline_profile.update(raw_markers)
        # 8. Формирование массива интерпретируемых объяснений ext (формула 53 НИР2)
        explanations = []
        for marker_name, z_val in z_scores.items():
            if abs(z_val) >= 3.0:  # Порог правила трех сигм (3-sigma)
                direction = "превышение нормы" if z_val > 0 else "падение ниже нормы"
                explanations.append(f"Маркер '{marker_name}': {direction} (z={z_val:+.2f})")
        return RiskSignal(
            employee_id=sender,
            timestamp=ts,
            raw_markers=raw_markers,
            normalized_deviations=z_scores,
            channel=message.channel,
            explanations=explanations
        )

    def process_stream(self, message_stream: Iterator[CommunicationMessage]) -> Iterator[RiskSignal]:
        """Потоковая пакетно-независимая обработка упорядоченного генератора сообщений."""
        for msg in message_stream:
            if msg.is_valid():
                yield self.process_message(msg)


def visualize_baseline_evolution(employees: Dict[str, EmployeeState], top_n: int = 3, output_dir: str = "output"):
    """
    Построение графиков сходимости математических ожиданий маркеров риска (EMA)
    для сотрудников с наибольшим числом сообщений.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Отбор top_n наиболее активных авторов
    sorted_senders = sorted(
        employees.keys(), 
        key=lambda s: employees[s].total_messages_sent, 
        reverse=True
    )[:top_n]

    if not sorted_senders or employees[sorted_senders[0]].total_messages_sent < 2:
        print("[ИНФО] Недостаточно сообщений для построения графиков конвергенции профилей.")
        return

    marker_cols = [
        "pronoun_marker", "syntactic_marker", 
        "activity_marker", "response_marker", "social_marker"
    ]
    marker_titles = [
        "Местоименный маркер (энтропия)", "Синтаксический маркер (сложность)", 
        "Маркер активности (темп)", "Маркер задержек ответа", "Социально-сетевой маркер"
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=False)
    fig.suptitle("Конвергенция математических ожиданий базового профиля (EMA)", fontsize=15, fontweight="bold", y=0.99)

    for i, (marker, title) in enumerate(zip(marker_cols, marker_titles)):
        ax = axes[i]
        for sender in sorted_senders:
            y_vals = employees[sender].baseline_profile.means_history[marker]
            x_vals = range(1, len(y_vals) + 1)
            ax.plot(x_vals, y_vals, label=f"{sender} ({len(y_vals)} сообщ.)", alpha=0.85, linewidth=2)
            
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel("Мат. ожидание (μ)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        if i == 4:
            ax.set_xlabel("Порядковый номер сообщения сотрудника", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filepath = os.path.join(output_dir, "baseline_evolution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ИНФО] График эволюции базового профиля сохранён в '{filepath}'.")


def visualize_deviations(employees: Dict[str, EmployeeState], top_n: int = 3, output_dir: str = "output"):
    """
    Построение динамики нормализованных z-score отклонений с пороговыми линиями ±3σ.
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_senders = sorted(
        employees.keys(), 
        key=lambda s: employees[s].total_messages_sent, 
        reverse=True
    )[:top_n]

    if not sorted_senders or employees[sorted_senders[0]].total_messages_sent < 2:
        return

    marker_cols = [
        "pronoun_marker", "syntactic_marker", 
        "activity_marker", "response_marker", "social_marker"
    ]
    marker_titles = [
        "Отклонения: Местоименный маркер", "Отклонения: Синтаксический маркер", 
        "Отклонения: Активность", "Отклонения: Задержки ответа", "Отклонения: Социальные связи"
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=False)
    fig.suptitle("Динамика z-score отклонений от индивидуальной нормы", fontsize=15, fontweight="bold", y=0.99)

    for i, (marker, title) in enumerate(zip(marker_cols, marker_titles)):
        ax = axes[i]
        for sender in sorted_senders:
            y_vals = employees[sender].baseline_profile.deviations_history[marker]
            x_vals = range(1, len(y_vals) + 1)
            ax.plot(x_vals, y_vals, label=sender, alpha=0.75, linewidth=1.5)

        # Отрисовка порогов правила трёх сигм (±3σ)
        ax.axhline(y=3.0, color="r", linestyle="--", alpha=0.6, label="Порог аномалии (+3σ)")
        ax.axhline(y=-3.0, color="r", linestyle="--", alpha=0.6, label="Порог аномалии (-3σ)")

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel("z-score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        
        # Удаление дубликатов в легенде для пороговых линий
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
        if i == 4:
            ax.set_xlabel("Порядковый номер сообщения сотрудника", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filepath = os.path.join(output_dir, "deviations_dynamics.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ИНФО] График динамики отклонений сохранён в '{filepath}'.")


def run_pipeline(
    dataset_path: Optional[str] = None, 
    max_messages: Optional[int] = 1000,
    topology_window_days: int = 30
):
    print("\n" + "="*70)
    print("  СИСТЕМА ДВУХЭТАПНОГО ВЫЯВЛЕНИЯ ИНСАЙДЕРСКИХ УГРОЗ: ЭТАП 1")
    print("  (Потоковый риск-скрининг и динамическое профилирование)")
    print("="*70)

    print("\n[ИНФО] Инициализация экстрактора признаков spaCy...")
    extractor = SpacyStatisticalExtractor(model_name="en_core_web_sm")
    
    print(f"[ИНФО] Запуск StreamProcessor (скользящее окно графа: {topology_window_days} дн.)...")
    processor = StreamProcessor(
        feature_extractor=extractor, 
        topology_window_days=topology_window_days
    )
    parser = MessageParser(min_body_length=5)

    if dataset_path and os.path.exists(dataset_path):
        print(f"[ИНФО] Подключение к датасету: '{dataset_path}'")
        stream = parser.stream_cert_emails(dataset_path)
    else:
        print("[ВНИМАНИЕ] Файл датасета не найден или не указан. Запуск тестового генератора...")
        base_time = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        demo_messages = [
            CommunicationMessage(
                sender_id="aab0573@dtaa.com",
                receiver_ids=["cgh0024@dtaa.com"],
                timestamp=base_time + timedelta(minutes=i*15),
                subject=f"Status update {i}",
                body=f"Regular communication update {i}. All tasks moving as expected." if i < 15 else "I hate this management! Everything is broken and nobody listens to me!"
            )
            for i in range(25)
        ]
        stream = iter(demo_messages)

    total_processed = 0
    anomalies_detected = 0

    print("\n[ИНФО] Обработка потока событий...")
    pbar = tqdm(total=max_messages, desc="Скрининг событий", unit="сообщ.")

    start_time = time.time()
    last_log_time = start_time

    for signal in processor.process_stream(stream):
        total_processed += 1
        pbar.update(1)

        # Регистрация аномалий
        if signal.explanations:
            anomalies_detected += 1
            tqdm.write(f"\n[ТРЕВОГА] {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Автор: {signal.employee_id}")
            for exp in signal.explanations:
                tqdm.write(f"  └── {exp}")

        # Логирование скорости каждые 100 сообщений (подобно старой версии)
        if total_processed % 100 == 0:
            current_t = time.time()
            elapsed_interval = current_t - last_log_time
            speed = 100.0 / elapsed_interval if elapsed_interval > 0 else 0.0
            if max_messages and max_messages > total_processed:
                remaining_sec = (max_messages - total_processed) / speed if speed > 0 else 0
                tqdm.write(f"[ПРОГРЕСС] Обработано {total_processed:,} | Скорость: {speed:.1f} сообщ./сек | Оставшееся время: {remaining_sec/60:.1f} мин.")
            last_log_time = current_t

        if max_messages and total_processed >= max_messages:
            break

    pbar.close()
    total_time = time.time() - start_time
    avg_speed = total_processed / total_time if total_time > 0 else 0.0

    print("\n" + "-"*70)
    print("ИТОГИ РИСК-СКРИНИНГА:")
    print(f"  • Всего обработано сообщений:       {total_processed:,}")
    print(f"  • Зафиксировано аномальных сигналов: {anomalies_detected:,}")
    print(f"  • Активных профилей сотрудников:    {len(processor.employees):,}")
    print(f"  • Узлов в графе коммуникаций:       {processor.graph.graph.number_of_nodes():,}")
    print(f"  • Рёбер в графе коммуникаций:       {processor.graph.graph.number_of_edges():,}")
    print(f"  • Общее время работы:               {total_time:.2f} сек.")
    print(f"  • Средняя скорость конвейера:       {avg_speed:.1f} сообщ./сек.")
    print("-"*70 + "\n")

    # Построение графиков для топ-3 авторов
    print("[ИНФО] Построение графических отчётов...")
    visualize_baseline_evolution(processor.employees, top_n=3, output_dir="output")
    visualize_deviations(processor.employees, top_n=3, output_dir="output")


if __name__ == "__main__":
    # Укажите путь к распакованному файлу email.csv из CERT r4.2
    # Например: CERT_EMAIL_PATH = "data/CERT_r4.2/email.csv"
    CERT_EMAIL_PATH = None
    run_pipeline(
        dataset_path=CERT_EMAIL_PATH,
        max_messages=1000,
        topology_window_days=30
    )