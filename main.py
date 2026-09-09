# Импорты
import os
import pandas as pd
import re
import time
import numpy as np
from collections import Counter
import scipy.stats as stats
import spacy
import networkx as nx
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Константы и настройки
DATASET_PATH = "datasets/emails.csv"

TIME_WINDOWS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90
}

EPSILON = 1e-8

# Загрузка данных
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Загружено сырых записей: {len(df)}")
        print(f"[INFO] Столбцы: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке датасета: {e}")
        return pd.DataFrame()


# Парсинг и предобработка сообщений
def parse_email(raw_message):
    result = {
        "sender": None,
        "receiver": None,
        "timestamp": None,
        "subject": None,
        "body": None,
        "channel": "email"
    }
    if pd.isna(raw_message):
        return result
    lines = raw_message.splitlines()
    body_start = None
    for i, line in enumerate(lines):
        if line.startswith("From:"):
            result["sender"] = line.replace("From:", "").strip()
        elif line.startswith("To:"):
            result["receiver"] = line.replace("To:", "").strip()
        elif line.startswith("Date:"):
            raw_date = line.replace("Date:", "").strip()
            try:
                parsed_date = pd.to_datetime(raw_date, errors='coerce')
                if pd.notna(parsed_date) and (1990 <= parsed_date.year <= 2030):
                    result["timestamp"] = parsed_date
                else:
                    result["timestamp"] = None
            except Exception:
                result["timestamp"] = None
        elif line.startswith("Subject:"):
            result["subject"] = line.replace("Subject:", "").strip()
        elif line.strip() == "":
            body_start = i + 1
            break
    if body_start is not None:
        body = "\n".join(lines[body_start:])
        result["body"] = body.strip()
    return result


def parse_timestamp(timestamp_str):
    try:
        return pd.to_datetime(timestamp_str, utc=True)
    except:
        return pd.NaT


def preprocess_messages(df):
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing emails"):
        parsed = parse_email(row["message"])
        records.append({
            "sender": parsed["sender"], 
            "receiver": parsed["receiver"], 
            "timestamp": parsed["timestamp"], 
            "subject": parsed["subject"], 
            "message": parsed["body"], 
            "channel": parsed["channel"]
        })
    processed_df = pd.DataFrame(records)
    initial_len = len(processed_df)
    processed_df = processed_df.dropna(subset=["timestamp", "sender", "message"])
    dropped_len = initial_len - len(processed_df)
    processed_df = processed_df.sort_values(by="timestamp").reset_index(drop=True)
    print(f"[INFO] После парсинга и фильтрации осталось {len(processed_df)} сообщений.")
    print(f"[INFO] Отброшено {dropped_len} дефектных сообщений (в т.ч. с заглушками дат).")
    return processed_df

# Извлечение психолингвистических признаков
def split_receivers(receiver_field):
    if pd.isna(receiver_field):
        return []
    receivers = re.split(r"[;,]", str(receiver_field))
    return [r.strip().lower() for r in receivers if r.strip()]


def build_communication_graph(df):
    graph = nx.DiGraph()
    print("\n[ИНФО] Построение графа коммуникаций...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Построение графа", unit="сообщ."):
        sender = str(row["sender"]).strip().lower()
        receivers = split_receivers(row["receiver"])
        for receiver in receivers:
            if graph.has_edge(sender, receiver):
                graph[sender][receiver]["weight"] += 1
            else:
                graph.add_edge(sender, receiver, weight=1)
    return graph


def calculate_graph_metrics(graph):
    print("\n[ИНФО] Расчёт посреднической центральности...")
    print("[ПРЕДУПРЕЖДЕНИЕ] Для крупных графов операция может выполняться длительное время.")
    start_time = time.time()
    try:
        betweenness = nx.betweenness_centrality(graph, normalized=True)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось вычислить центральность: {e}")
        betweenness = {}
    elapsed = time.time() - start_time
    print(f"[ИНФО] Расчёт центральности завершён за {elapsed:.2f} сек.")
    return betweenness


def prepare_activity_statistics(df):
    print("\n[ИНФО] Подготовка статистики активности...")
    temp_df = df.copy()
    temp_df["sender"] = temp_df["sender"].astype(str).str.strip().str.lower()
    temp_df["parsed_timestamp"] = temp_df["timestamp"]
    temp_df = temp_df.dropna(subset=["parsed_timestamp"])
    sent_counter = temp_df.groupby("sender").size().to_dict()
    received_counter = {}
    for receiver_field in temp_df["receiver"]:
        receivers = split_receivers(receiver_field)
        for receiver in receivers:
            received_counter[receiver] = received_counter.get(receiver, 0) + 1
    activity_history = {}
    for sender, group in temp_df.groupby("sender"):
        group = group.sort_values("parsed_timestamp")
        daily = group.set_index("parsed_timestamp").resample("D").size()
        weekly = group.set_index("parsed_timestamp").resample("W").size()
        monthly = group.set_index("parsed_timestamp").resample("ME").size()
        quarterly = group.set_index("parsed_timestamp").resample("QE").size()
        activity_history[sender] = {
            "daily": {str(k.date()): int(v) for k, v in daily.items()},
            "weekly": {str(k.tz_localize(None).to_period("W")): int(v) for k, v in weekly.items()},
            "monthly": {str(k.tz_localize(None).to_period("M")): int(v) for k, v in monthly.items()},
            "quarterly": {str(k.tz_localize(None).to_period("Q")): int(v) for k, v in quarterly.items()}
        }
    return (sent_counter, received_counter, activity_history)


def prepare_reply_statistics(df):
    print("\n[ИНФО] Подготовка характеристик времени ответа...")
    temp_df = df.copy()
    temp_df["parsed_timestamp"] = temp_df["timestamp"]
    temp_df = temp_df.dropna(subset=["sender", "receiver", "parsed_timestamp"])
    temp_df["sender"] = temp_df["sender"].astype(str).str.strip().str.lower()
    temp_df["receiver"] = temp_df["receiver"].astype(str).str.strip().str.lower()
    last_message = {}
    reply_times = {}
    for _, row in tqdm(temp_df.iterrows(), total=len(temp_df), desc="Анализ времени ответа", unit="сообщ."):
        sender = row["sender"]
        receivers = split_receivers(row["receiver"])
        current_time = row["parsed_timestamp"]
        for receiver in receivers:
            reverse_key = (receiver, sender)
            if reverse_key in last_message:
                previous_time = last_message[reverse_key]
                delta = (current_time - previous_time).total_seconds()
                if delta > 0:
                    reply_times.setdefault(sender, []).append(delta)
            last_message[(sender, receiver)] = current_time
    statistics = {}
    for employee, values in reply_times.items():
        if len(values) == 0:
            statistics[employee] = (np.nan, np.nan, np.nan)
            continue
        statistics[employee] = (float(np.mean(values)), float(np.median(values)), float(np.std(values)))
    print(f"[ИНФО] Подготовлены характеристики ответа для {len(statistics):,} сотрудников.")
    return statistics


def extract_pronoun_features(doc):
    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    pronoun_count = len(pronouns)
    first_person = sum(p in ["i", "me", "my", "mine", "myself"] for p in pronouns)
    second_person = sum(p in ["you", "your", "yours", "yourself"] for p in pronouns)
    third_person = sum(p in ["he", "him", "his", "she", "her", "hers", "they", "them", "their"] for p in pronouns)
    return (pronoun_count, first_person, second_person, third_person)


def extract_syntactic_features(doc, text):
    sentences = list(doc.sents)
    sentence_lengths = [len(sent) for sent in sentences]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    dependency_depth = np.mean([len(list(token.ancestors)) for token in doc]) if len(doc) > 0 else 0
    pos_tags = [token.pos_ for token in doc]
    trigrams = [(pos_tags[i], pos_tags[i + 1], pos_tags[i + 2]) for i in range(len(pos_tags) - 2)]
    if trigrams:
        syntax_pattern_frequency = Counter(trigrams).most_common(1)[0][1] / len(trigrams)
    else:
        syntax_pattern_frequency = 0
    discourse_markers = ["however", "therefore", "thus", "although", "because", "meanwhile", "moreover"]
    discourse_count = sum(text.lower().count(marker) for marker in discourse_markers)
    discourse_ratio = discourse_count / max(len(doc), 1)
    return (avg_sentence_length, dependency_depth, discourse_ratio, syntax_pattern_frequency)


def extract_emotional_features(text):
    exclamation_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
    return (exclamation_count, question_count, uppercase_ratio)


def extract_activity_features(sender, timestamp, sent_counter, received_counter, activity_history):
    total_sent = sent_counter.get(sender, 0)
    received_messages = received_counter.get(sender, 0)
    work_hours = 0
    off_hours = 0
    daily_change = np.nan
    weekly_change = np.nan
    monthly_change = np.nan
    quarterly_change = np.nan
    if pd.notna(timestamp):
        hour = timestamp.hour
        if 8 <= hour <= 18:
            work_hours = 1
        else:
            off_hours = 1
        history = activity_history.get(sender, {})
        ts_naive = timestamp.tz_localize(None)
        # ---------- day ----------
        day_period = ts_naive.to_period("D")
        day_key = str(day_period.start_time.date())
        current_day = history.get("daily", {}).get(day_key, 0)
        prev_day_key = str((day_period - 1).start_time.date())
        prev_day = history.get("daily", {}).get(prev_day_key, 0)
        if prev_day > 0:
            daily_change = (current_day - prev_day) / prev_day
        # ---------- week ----------
        week_period = ts_naive.to_period("W")
        week_key = str(week_period)
        current_week = history.get("weekly", {}).get(week_key, 0)
        prev_week_key = str(week_period - 1)
        prev_week = history.get("weekly", {}).get(prev_week_key, 0)
        if prev_week > 0:
            weekly_change = (current_week - prev_week) / prev_week
        # ---------- month ----------
        month_period = ts_naive.to_period("M")
        month_key = str(month_period)
        current_month = history.get("monthly", {}).get(month_key, 0)
        prev_month_key = str(month_period - 1)
        prev_month = history.get("monthly", {}).get(prev_month_key, 0)
        if prev_month > 0:
            monthly_change = (current_month - prev_month) / prev_month
        # ---------- quarter ----------
        quarter_period = ts_naive.to_period("Q")
        quarter_key = str(quarter_period)
        current_quarter = history.get("quarterly", {}).get(quarter_key, 0)
        prev_quarter_key = str(quarter_period - 1)
        prev_quarter = history.get("quarterly", {}).get(prev_quarter_key, 0)
        if prev_quarter > 0:
            quarterly_change = (current_quarter - prev_quarter) / prev_quarter
    return {
        "sent_messages": total_sent,
        "received_messages": received_messages,
        "daily_change": daily_change,
        "weekly_change": weekly_change,
        "monthly_change": monthly_change,
        "quarterly_change": quarterly_change,
        "worktime_messages": work_hours,
        "offtime_messages": off_hours
    }


def extract_social_features(sender, receiver_field, graph, betweenness):
    try:
        out_degree = float(graph.out_degree(sender))
    except (TypeError, nx.NetworkXError):
        out_degree = 0.0
    try:
        in_degree = float(graph.in_degree(sender))
    except (TypeError, nx.NetworkXError):
        in_degree = 0.0
    receivers = split_receivers(receiver_field)
    reciprocity = 0.0
    for receiver in receivers:
        if graph.has_edge(receiver, sender):
            reciprocity = 1.0
            break
    betweenness_value = float(betweenness.get(sender, 0.0))
    return (out_degree, in_degree, reciprocity, betweenness_value)


def extract_response_features(sender, reply_statistics):
    return reply_statistics.get(sender, (np.nan, np.nan, np.nan))


def extract_features(df, nlp):
    print("\n[ИНФО] Запуск извлечения признаков...")
    print(f"[ИНФО] Количество сообщений для обработки: {len(df):,}")
    total_start_time = time.time()
    feature_rows = []
    # Подготовительные расчёты
    graph = build_communication_graph(df)
    betweenness = calculate_graph_metrics(graph)
    sent_counter, received_counter, activity_history = prepare_activity_statistics(df)
    reply_statistics = prepare_reply_statistics(df)
    # Извлечение признаков
    print("\n[ИНФО] Извлечение психолингвистических и временных признаков...")
    processed_count = 0
    feature_start_time = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение признаков", unit="сообщ."):
        if pd.isna(row["message"]):
            continue
        text = str(row["message"])
        sender = str(row["sender"]).strip().lower()
        receiver = row["receiver"]
        timestamp = row["timestamp"]
        try:
            doc = nlp(text)
        except Exception:
            continue
        # Временные окна
        window_day = None
        window_week = None
        window_month = None
        window_quarter = None
        if pd.notna(timestamp):
            ts_naive = timestamp.tz_localize(None)
            window_day = ts_naive.floor("D")
            window_week = str(ts_naive.to_period("W"))
            window_month = str(ts_naive.to_period("M"))
            window_quarter = str(ts_naive.to_period("Q"))
        # 1. Местоименные признаки
        (pronoun_count, first_person, second_person, third_person) = extract_pronoun_features(doc)
        # 2. Синтаксические признаки
        (avg_sentence_length, dependency_depth, discourse_ratio, syntax_pattern_frequency) = extract_syntactic_features(doc, text)
        # 3. Эмоциональная нагрузка
        (exclamation_count, question_count, uppercase_ratio) = extract_emotional_features(text)
        # 4. Активность
        activity_features = (extract_activity_features(sender, timestamp, sent_counter, received_counter, activity_history))
        # 5. Социально-сетевые признаки
        (out_degree, in_degree, reciprocity, betweenness_value) = extract_social_features(sender, receiver, graph, betweenness)
        # 6. Характеристики ответа
        (mean_reply_time, median_reply_time, reply_time_std) = extract_response_features(sender, reply_statistics)
        # Формирование записи
        feature_rows.append({
            "sender": sender,
            "receiver": receiver,
            "timestamp": timestamp,
            "window_day": window_day,
            "window_week": window_week,
            "window_month": window_month,
            "window_quarter": window_quarter,
            # местоимения
            "pronoun_count": pronoun_count,
            "first_person": first_person,
            "second_person": second_person,
            "third_person": third_person,
            # синтаксис
            "avg_sentence_length": avg_sentence_length,
            "dependency_depth": dependency_depth,
            "discourse_ratio": discourse_ratio,
            "syntax_pattern_frequency": syntax_pattern_frequency,
            # эмоции
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "uppercase_ratio": uppercase_ratio,
            # активность
            "sent_messages": activity_features["sent_messages"],
            "received_messages": activity_features["received_messages"],
            "daily_activity_change": activity_features["daily_change"],
            "weekly_activity_change": activity_features["weekly_change"],
            "monthly_activity_change": activity_features["monthly_change"],
            "quarterly_activity_change": activity_features["quarterly_change"],
            "worktime_messages": activity_features["worktime_messages"],
            "offtime_messages": activity_features["offtime_messages"],
            # социальный граф
            "out_degree": out_degree,
            "in_degree": in_degree,
            "reciprocity": reciprocity,
            "betweenness": betweenness_value,
            # время ответа
            "mean_reply_time": mean_reply_time,
            "median_reply_time": median_reply_time,
            "reply_time_std": reply_time_std
        })
        processed_count += 1
        if processed_count % 10000 == 0:
            elapsed = time.time() - feature_start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            remaining = (len(df) - processed_count) / speed if speed > 0 else 0
            print(f"\n[ИНФО] Обработано {processed_count:,} из {len(df):,} сообщений")
            print(f"[ИНФО] Скорость обработки: {speed:.1f} сообщ./сек.")
            print(f"[ИНФО] Ориентировочное оставшееся время: {remaining / 60:.1f} мин.")
    total_time = time.time() - total_start_time
    print("\n[ИНФО] Извлечение признаков завершено.")
    print(f"[ИНФО] Успешно обработано сообщений: {processed_count:,}")
    print(f"[ИНФО] Получено записей признаков: {len(feature_rows):,}")
    print(f"[ИНФО] Общее время выполнения: {total_time:.2f} сек.")
    return pd.DataFrame(feature_rows)

# Извлечение психолингвистических маркеров
def extract_pronoun_markers(feature_df):
    result = feature_df[["sender", "timestamp"]].copy()
    alpha = 1.0
    k = 3.0
    total = feature_df["pronoun_count"] + (alpha * k)
    p1 = (feature_df["first_person"] + alpha) / total
    p2 = (feature_df["second_person"] + alpha) / total
    p3 = (feature_df["third_person"] + alpha) / total
    result["pronoun_marker"] = - (p1 * np.log2(p1) + p2 * np.log2(p2) + p3 * np.log2(p3))
    return result


def extract_syntactic_markers(feature_df):
    result = feature_df[["sender", "timestamp"]].copy()
    len_norm = np.log1p(feature_df["avg_sentence_length"].fillna(0))
    dep_norm = np.log1p(feature_df["dependency_depth"].fillna(0))
    disc_norm = feature_df["discourse_ratio"].fillna(0)
    freq_norm = feature_df["syntax_pattern_frequency"].fillna(0)
    result["syntactic_marker"] = np.sqrt(len_norm**2 + dep_norm**2 + disc_norm**2 + freq_norm**2)
    return result


def extract_activity_markers(feature_df):
    result = feature_df[["sender", "timestamp"]].copy()
    daily = np.tanh(feature_df["daily_activity_change"].fillna(0))
    weekly = np.tanh(feature_df["weekly_activity_change"].fillna(0))
    monthly = np.tanh(feature_df["monthly_activity_change"].fillna(0))
    quarterly = np.tanh(feature_df["quarterly_activity_change"].fillna(0))
    result["activity_marker"] = np.sqrt((daily**2 + weekly**2 + monthly**2 + quarterly**2) / 4.0)
    return result


def extract_social_markers(feature_df):
    result = feature_df[["sender", "timestamp"]].copy()
    out_deg = np.log1p(pd.to_numeric(feature_df["out_degree"], errors='coerce').fillna(0))
    in_deg = np.log1p(pd.to_numeric(feature_df["in_degree"], errors='coerce').fillna(0))
    recip = pd.to_numeric(feature_df["reciprocity"], errors='coerce').fillna(0)
    betw = pd.to_numeric(feature_df["betweenness"], errors='coerce').fillna(0)
    result["social_marker"] = np.sqrt(out_deg**2 + in_deg**2 + recip**2 + betw**2)
    return result


def extract_response_markers(feature_df):
    result = feature_df[["sender", "timestamp"]].copy()
    mean_rep = np.log1p(feature_df["mean_reply_time"].fillna(0))
    med_rep = np.log1p(feature_df["median_reply_time"].fillna(0))
    std_rep = np.log1p(feature_df["reply_time_std"].fillna(0))
    result["response_marker"] = np.sqrt(mean_rep**2 + med_rep**2 + std_rep**2)
    return result


# Построение базового профиля сотрудника
def build_employee_baseline(markers_df, alpha=0.1):
    print(f"\n[ИНФО] Динамическое обновление базовых профилей (алгоритм EMA, alpha={alpha})...")
    profiles = {}
    history = {}
    marker_cols = [
        'pronoun_marker', 'syntactic_marker', 
        'activity_marker', 'response_marker', 'social_marker'
    ]
    for _, row in tqdm(markers_df.iterrows(), total=len(markers_df), desc="Профилирование EMA", unit="сообщ."):
        sender = row['sender']
        if sender not in profiles:
            profiles[sender] = {
                'count': 0,
                'markers': {m: {'mean': 0.0, 'variance': 0.0} for m in marker_cols}
            }
            history[sender] = {m: [] for m in marker_cols}
            for m in marker_cols:
                val = row[m] if not pd.isna(row[m]) else 0.0
                profiles[sender]['markers'][m]['mean'] = val
                profiles[sender]['markers'][m]['variance'] = 0.0
                history[sender][m].append(val)  
            profiles[sender]['count'] += 1
            continue
        profiles[sender]['count'] += 1
        for m in marker_cols:
            val = row[m] if not pd.isna(row[m]) else 0.0  
            prev_mean = profiles[sender]['markers'][m]['mean']
            prev_var = profiles[sender]['markers'][m]['variance']
            diff = val - prev_mean
            new_mean = prev_mean + alpha * diff
            profiles[sender]['markers'][m]['mean'] = new_mean
            new_var = (1 - alpha) * (prev_var + alpha * (diff ** 2))
            profiles[sender]['markers'][m]['variance'] = new_var
            history[sender][m].append(new_mean)
    print(f"[ИНФО] Сформированы базовые профили (EMA) для {len(profiles)} сотрудников.")
    return profiles, history


def visualize_baseline_evolution(history_dict, top_n=2):
    print("\n[ИНФО] Визуализация эволюции базового профиля...")
    os.makedirs("output", exist_ok=True)
    senders = sorted(
        history_dict.keys(), 
        key=lambda s: len(history_dict[s]['pronoun_marker']), 
        reverse=True
    )[:top_n]
    marker_cols = [
        'pronoun_marker', 'syntactic_marker', 
        'activity_marker', 'response_marker', 'social_marker'
    ]
    marker_titles = [
        'Местоименный маркер', 'Синтаксический маркер', 
        'Маркер активности', 'Маркер времени ответа', 'Социально-сетевой маркер'
    ]
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=False)
    fig.suptitle('Конвергенция математических ожиданий маркеров риска', fontsize=16, fontweight='bold', y=0.98)
    for i, (marker, title) in enumerate(zip(marker_cols, marker_titles)):
        ax = axes[i]
        for sender in senders:
            y_values = history_dict[sender][marker]
            x_values = range(1, len(y_values) + 1)
            ax.plot(x_values, y_values, label=sender, alpha=0.8, linewidth=2) 
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel('Мат. ожидание', fontsize=12)
        if i == 4:
            ax.set_xlabel('Количество проанализированных сообщений', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = "output/baseline_random1.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ИНФО] График конвергенции профилей сохранён в '{filepath}'.")


# Нормализация признаков и обновление профиля
def normalize_features(markers_df, alpha=0.1):
    print(f"\n[ИНФО] Потоковая нормализация и обновление профилей (EMA alpha={alpha})...")
    profiles = {}
    deviations_history = {}
    normalized_records = []
    marker_cols = [
        'pronoun_marker', 'syntactic_marker', 
        'activity_marker', 'response_marker', 'social_marker'
    ]
    min_variance = 1e-4
    clip_limit = 10.0
    for _, row in tqdm(markers_df.iterrows(), total=len(markers_df), desc="Нормализация", unit="сообщ."):
        sender = row['sender']
        norm_row = row.to_dict()
        if sender not in profiles:
            profiles[sender] = {
                'count': 0,
                'markers': {m: {'mean': 0.0, 'variance': 0.0} for m in marker_cols}
            }
            deviations_history[sender] = {m: [] for m in marker_cols}
            for m in marker_cols:
                val = row[m] if not pd.isna(row[m]) else 0.0
                profiles[sender]['markers'][m]['mean'] = val
                profiles[sender]['markers'][m]['variance'] = 0.0
                norm_row[f"{m}_norm"] = 0.0
                deviations_history[sender][m].append(0.0)          
            profiles[sender]['count'] += 1
            normalized_records.append(norm_row)
            continue
        for m in marker_cols:
            val = row[m] if not pd.isna(row[m]) else 0.0
            mean = profiles[sender]['markers'][m]['mean']
            variance = profiles[sender]['markers'][m]['variance']
            safe_variance = max(variance, min_variance)
            std = np.sqrt(safe_variance)
            raw_z_score = (val - mean) / std
            z_score = float(np.clip(raw_z_score, -clip_limit, clip_limit))
            norm_row[f"{m}_norm"] = z_score
            deviations_history[sender][m].append(z_score)
            diff = val - mean
            new_mean = mean + alpha * diff
            new_var = (1 - alpha) * (variance + alpha * (diff ** 2))
            profiles[sender]['markers'][m]['mean'] = new_mean
            profiles[sender]['markers'][m]['variance'] = new_var
        profiles[sender]['count'] += 1
        normalized_records.append(norm_row)
    print(f"[ИНФО] Нормализация завершена для {len(profiles)} профилей.")
    return pd.DataFrame(normalized_records), profiles, deviations_history


def visualize_deviations(deviations_history, top_n=2):
    print("\n[ИНФО] Визуализация отклонений от базового профиля...")
    os.makedirs("output", exist_ok=True)
    senders = sorted(
        deviations_history.keys(), 
        key=lambda s: len(deviations_history[s]['pronoun_marker']), 
        reverse=True
    )[:top_n]

    marker_cols = [
        'pronoun_marker', 'syntactic_marker', 
        'activity_marker', 'response_marker', 'social_marker'
    ]
    marker_titles = [
        'Местоименный маркер', 'Синтаксический маркер', 
        'Активность', 'Время ответа', 'Граф социальных взаимодействий'
    ]
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=False)
    fig.suptitle('Динамика отклонений от базового профиля', fontsize=16, fontweight='bold', y=0.98)
    for i, (marker, title) in enumerate(zip(marker_cols, marker_titles)):
        ax = axes[i]
        for sender in senders:
            y_values = deviations_history[sender][marker]
            x_values = range(1, len(y_values) + 1)
            ax.plot(x_values, y_values, label=sender, alpha=0.7, linewidth=1.5)
        ax.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Порог аномалии (+3σ)')
        ax.axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='Порог аномалии (-3σ)')
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel('Отклонение', fontsize=12)
        if i == 4:
            ax.set_xlabel('Порядковый номер сообщения сотрудника', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = "output/deviations_random1.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ИНФО] График отклонений сохранён в '{filepath}'.")


def main():
    print("Hello!")

    nlp = spacy.load("en_core_web_sm")

    df = load_dataset(DATASET_PATH)
    messages = preprocess_messages(df)
    messages = messages.sample(n=1000, random_state=42).reset_index(drop=True)
    print("\n[ИНФО] Хронологическая сортировка корпуса...")
    messages["timestamp"] = messages["timestamp"].apply(parse_timestamp)
    messages = messages.dropna(subset=["timestamp"])
    messages = messages.sort_values("timestamp").reset_index(drop=True)
    print(f"[ИНФО] После фильтрации и сортировки осталось {len(messages)} сообщений")

    features = extract_features(messages, nlp)
    pronoun_markers = extract_pronoun_markers(features)
    syntactic_markers = extract_syntactic_markers(features)
    activity_markers = extract_activity_markers(features)
    response_markers = extract_response_markers(features)
    social_markers = extract_social_markers(features)

    markers_df = features[['sender', 'timestamp']].copy()
    markers_df['pronoun_marker'] = pronoun_markers['pronoun_marker']
    markers_df['syntactic_marker'] = syntactic_markers['syntactic_marker']
    markers_df['activity_marker'] = activity_markers['activity_marker']
    markers_df['response_marker'] = response_markers['response_marker']
    markers_df['social_marker'] = social_markers['social_marker']
    markers_df['timestamp'] = pd.to_datetime(markers_df['timestamp'], utc=True)
    markers_df = markers_df.sort_values('timestamp').reset_index(drop=True)
    profiles, baseline_history = build_employee_baseline(markers_df)
    visualize_baseline_evolution(baseline_history, top_n=3)
    normalized_markers_df, profiles, deviations_history = normalize_features(markers_df, alpha=0.1)
    visualize_deviations(deviations_history, top_n=3)

    print(messages.head())
    print(messages.columns)
    print(features.head())
    print(features.columns)
    print("\n=== PRONOUN MARKERS ===")
    print(pronoun_markers.head())
    print("\n=== SYNTACTIC MARKERS ===")
    print(syntactic_markers.head())
    print("\n=== ACTIVITY MARKERS ===")
    print(activity_markers.head())
    print("\n=== RESPONSE MARKERS ===")
    print(response_markers.head())
    print("\n=== SOCIAL MARKERS ===")
    print(social_markers.head())
    print("\n[ИНФО] Нормализованные данные:")
    print(normalized_markers_df.head())

if __name__ == "__main__":
    main()