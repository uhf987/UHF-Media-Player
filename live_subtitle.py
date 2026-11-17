# -*- coding: utf-8 -*-
"""
live_subtitle.py  -  Video Stüdyo & Oynatıcı için "Sesten altyazı" + "Canlı Türkçe çeviri"

İki bağımsız özellik (oynatıcıda yan yana iki düğme):

  1) TranscribeWorker  : Filmin sesini faster-whisper ile metne çevirir ve videonun
                         yanına  <film>.ai.<dil>.srt  olarak kaydeder (dil otomatik
                         algılanır). Dosya zaten varsa hiçbir şey yapmaz.
  2) TranslateWorker   : O .srt'yi Ollama + TranslateGemma (4B/12B/27B) ile Türkçeye çevirir.
                         Çeviri OYNATMA KONUMUNDAN başlar (ileri sararsanız oradan
                         devam eder), bitince  <film>.ai.tr.srt  olarak kaydedilir.

Gereksinimler:  pip install faster-whisper requests      (PyQt5 zaten kurulu)
                Ollama (https://ollama.com) açık olmalı. TranslateGemma'yı Ollama ile
                kendiniz yüklersiniz (uygulama indirmez):  ollama pull translategemma:12b
                Whisper modelleri oynatıcıdaki 📦 Model Ayarları penceresinden indirilir.

Bu modül Qt'ye bağımlı iş parçacığı sınıfları (…Worker) ile Qt'siz saf mantığı
(SRT, cümle birleştirme, bölme, çeviri istemi) bir arada tutar.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import requests
from PyQt5.QtCore import QThread, pyqtSignal

OLLAMA_HOST = "http://localhost:11434"
TRANSLATE_MODEL = "translategemma:4b"
AI_TAG = "ai"            # film.ai.en.srt / film.ai.tr.srt  (gerçek altyazılarla karışmasın)
TARGET_LANG = "tr"

LANG_NAMES = {
    "en": "English", "tr": "Turkish", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "hu": "Hungarian",
    "ro": "Romanian", "el": "Greek", "he": "Hebrew", "hi": "Hindi", "id": "Indonesian",
    "th": "Thai", "vi": "Vietnamese", "uk": "Ukrainian", "fa": "Persian", "bg": "Bulgarian",
    "sr": "Serbian", "hr": "Croatian", "sk": "Slovak", "ca": "Catalan",
    "az": "Azerbaijani", "bn": "Bengali", "ur": "Urdu", "ms": "Malay", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "af": "Afrikaans", "sl": "Slovenian", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "is": "Icelandic", "ka": "Georgian", "hy": "Armenian",
    "kk": "Kazakh", "ne": "Nepali", "fil": "Filipino", "my": "Burmese", "km": "Khmer",
}

# Arayüzde gösterilecek Türkçe adlar (yoksa İngilizce ad gösterilir)
TR_NAMES = {
    "en": "İngilizce", "tr": "Türkçe", "de": "Almanca", "fr": "Fransızca", "es": "İspanyolca",
    "it": "İtalyanca", "ru": "Rusça", "ja": "Japonca", "ko": "Korece", "zh": "Çince",
    "ar": "Arapça", "pt": "Portekizce", "nl": "Felemenkçe", "pl": "Lehçe", "sv": "İsveççe",
    "da": "Danca", "no": "Norveççe", "fi": "Fince", "cs": "Çekçe", "hu": "Macarca",
    "ro": "Romence", "el": "Yunanca", "he": "İbranice", "hi": "Hintçe", "id": "Endonezce",
    "th": "Tayca", "vi": "Vietnamca", "uk": "Ukraynaca", "fa": "Farsça", "bg": "Bulgarca",
    "sr": "Sırpça", "hr": "Hırvatça", "sk": "Slovakça", "ca": "Katalanca", "az": "Azerbaycanca",
    "bn": "Bengalce", "ur": "Urduca", "ms": "Malayca", "sw": "Svahili", "ta": "Tamilce",
    "te": "Telugu dili", "mr": "Marathi dili", "gu": "Guceratça", "kn": "Kannada dili",
    "ml": "Malayalam dili", "pa": "Pencapça", "af": "Afrikaanca", "sl": "Slovence",
    "lt": "Litvanca", "lv": "Letonca", "et": "Estonca", "is": "İzlandaca", "ka": "Gürcüce",
    "hy": "Ermenice", "kk": "Kazakça", "ne": "Nepalce", "fil": "Filipince", "my": "Birmanca",
    "km": "Kmerce",
}


def lang_label(code):
    """'tr' -> 'Türkçe'; bilinmeyen kod olduğu gibi döner."""
    code = (code or "").lower()
    return TR_NAMES.get(code) or LANG_NAMES.get(code) or code


def translate_langs():
    """[(kod, görünen ad)] - Türkçe ada göre sıralı, çeviri için seçilebilen tüm diller."""
    items = [(c, lang_label(c)) for c in LANG_NAMES]
    items.sort(key=lambda t: t[1].casefold())
    return items


def pick_source(video, dst):
    """Çeviri kaynağı: videonun yanındaki film.ai.<dil>.srt dosyalarından hedef dil
    OLMAYAN en eski olanı (Whisper çıktısı ilk üretilen dosyadır). -> (dil, yol) | None"""
    cands = [(l, p) for l, p in find_sources(video) if l != dst]
    if not cands:
        return None
    cands.sort(key=lambda lp: os.path.getmtime(lp[1]))
    return cands[0]


class OllamaDown(Exception):
    pass


class ModelMissing(Exception):
    pass


class Stopped(Exception):
    pass


class Degenerate(Exception):
    pass


# =============================================================== yollar
def _base(video):
    return os.path.splitext(video)[0]


def final_path(video, lang):
    return f"{_base(video)}.{AI_TAG}.{lang}.srt"


def find_sources(video):
    """Videonun yanındaki film.ai.<dil>.srt dosyaları -> [(dil, yol)]."""
    base = _base(video)
    skip = len(os.path.basename(base)) + len(AI_TAG) + 2
    out = []
    for p in glob.glob(glob.escape(base) + f".{AI_TAG}.*.srt"):
        lang = os.path.basename(p)[skip:-4]
        if re.fullmatch(r"[A-Za-z]{2,3}", lang):
            out.append((lang.lower(), p))
    return sorted(out)


def _replace(src, dst):
    """Windows'ta hedef dosya başka bir program tarafından açıksa kısa süre bekleyip dener."""
    for _ in range(15):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(src, dst)


# ===================================================== ayarlar ve model yönetimi
def _app_dir(kind):
    """kind: 'config' | 'data'. UHF_DATA_DIR ortam değişkeni (testler için) ikisini de geçersiz kılar."""
    root = os.environ.get("UHF_DATA_DIR")
    if not root:
        if os.name == "nt":
            base = os.environ.get("APPDATA" if kind == "config" else "LOCALAPPDATA") or os.path.expanduser("~")
            root = os.path.join(base, "UHF")
        else:
            root = os.path.join(os.path.expanduser("~/.config" if kind == "config" else "~/.local/share"), "uhf")
    os.makedirs(root, exist_ok=True)
    return root


DEFAULT_CONFIG = {"whisper_model": None, "translate_model": TRANSLATE_MODEL, "custom_whisper_dirs": [],
                  "xxl_exe": None, "favorite_langs": ["tr", "en"], "last_target": None}


def config_path():
    return os.path.join(_app_dir("config"), "live_subtitle.json")


def load_config():
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    try:
        with open(config_path(), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cfg.update({k: data[k] for k in DEFAULT_CONFIG if k in data})
    except Exception:
        pass
    return cfg


def save_config(cfg):
    tmp = config_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    _replace(tmp, config_path())


# Whisper (faster-whisper / CTranslate2) modelleri. 'mb' yaklaşık indirme boyutudur
# (gerçek boyut indirme sırasında Hugging Face'ten okunur).
WHISPER_MODELS = [
    {"id": "tiny", "repo": "Systran/faster-whisper-tiny", "mb": 75, "note": "çok hızlı, düşük doğruluk"},
    {"id": "base", "repo": "Systran/faster-whisper-base", "mb": 145, "note": "hızlı, düşük doğruluk"},
    {"id": "small", "repo": "Systran/faster-whisper-small", "mb": 484, "note": "işlemci için dengeli"},
    {"id": "medium", "repo": "Systran/faster-whisper-medium", "mb": 1530, "note": "iyi doğruluk, yavaş"},
    {"id": "large-v2", "repo": "Systran/faster-whisper-large-v2", "mb": 3090, "note": "yüksek doğruluk"},
    {"id": "large-v3", "repo": "Systran/faster-whisper-large-v3", "mb": 3090, "note": "en yüksek doğruluk"},
    {"id": "large-v3-turbo", "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "mb": 1620,
     "note": "large-v3'e yakın, çok daha hızlı"},
]
# TranslateGemma boyutları (Ollama'da yüklenir; GB değerleri yaklaşık)
TRANSLATE_MODELS = [
    ("translategemma:4b", "4B", 3.3, "hızlı, hafif"),
    ("translategemma:12b", "12B", 8.1, "daha doğru çeviri"),
    ("translategemma:27b", "27B", 17.0, "en iyi çeviri, çok VRAM ister"),
]


def whisper_dir():
    d = os.path.join(_app_dir("data"), "whisper")
    os.makedirs(d, exist_ok=True)
    return d


def whisper_model_dir(mid):
    """Klasör adı 'faster-whisper-<id>' (faster-whisper-xxl bu düzeni bekler).
    Eski sürümde 'whisper/<id>' olarak inmiş model varsa anında yeniden adlandırılır."""
    new = os.path.join(whisper_dir(), "faster-whisper-" + mid)
    old = os.path.join(whisper_dir(), mid)
    if not os.path.exists(new) and os.path.isdir(old):
        try:
            os.rename(old, new)
        except OSError:
            return old
    return new


def is_whisper_downloaded(mid):
    """Yalnızca indirme TAMAMEN bittiğinde yazılan '.complete' işaretine bakar;
    yarım inmiş klasör 'indirildi' sayılmaz."""
    return os.path.isfile(os.path.join(whisper_model_dir(mid), ".complete"))


def is_ct2_dir(path):
    return os.path.isfile(os.path.join(path, "model.bin")) and os.path.isfile(os.path.join(path, "config.json"))


def find_subtitle_edit_models(max_depth=6):
    """%APPDATA%\\Subtitle Edit\\Whisper altında CTranslate2 (model.bin) klasörü arar.
    (whisper.cpp 'ggml' .bin dosyaları faster-whisper ile UYUMLU DEĞİLDİR, bulunmaz.)"""
    found = []
    for root in _se_roots():          # yeni sürüm: SpeechToText, eski sürüm: Whisper
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for d, dirs, files in os.walk(root):
            if d.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            if "model.bin" in files and is_ct2_dir(d):
                found.append((os.path.basename(d), d))
                dirs[:] = []
    return found


def whisper_entries(cfg=None):
    """Listede gösterilecek tüm Whisper modelleri (katalog + Subtitle Edit + elle eklenen)."""
    cfg = cfg or load_config()
    out = []
    for m in WHISPER_MODELS:
        out.append({"id": m["id"], "label": m["id"], "mb": m["mb"], "note": m["note"], "kind": "catalog",
                    "downloaded": is_whisper_downloaded(m["id"]), "path": whisper_model_dir(m["id"])})
    seen = set()
    extra = [("subtitle_edit", n, p) for n, p in find_subtitle_edit_models()] + \
            [("custom", os.path.basename(p.rstrip("\\/")) or p, p) for p in cfg.get("custom_whisper_dirs", [])]
    for kind, name, path in extra:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen or not is_ct2_dir(path):
            continue
        seen.add(key)
        label = ("Subtitle Edit: " if kind == "subtitle_edit" else "Klasör: ") + name
        out.append({"id": "dir:" + path, "label": label, "mb": None, "note": path, "kind": kind,
                    "downloaded": True, "path": path})
    return out


def resolve_whisper(mid):
    """Model kimliğini yerel klasöre çevirir; yoksa/indirilmemişse None."""
    if not mid:
        return None
    if mid.startswith("dir:"):
        return mid[4:] if is_ct2_dir(mid[4:]) else None
    return whisper_model_dir(mid) if is_whisper_downloaded(mid) else None


def whisper_label(mid):
    if mid and mid.startswith("dir:"):
        return os.path.basename(mid[4:].rstrip("\\/")) or mid[4:]
    return mid or "?"


def selected_whisper_id(cfg=None):
    """Kayıtlı seçim; yoksa/silinmişse indirilmiş en iyi model; hiçbiri yoksa öneri
    (GPU varsa large-v3, yoksa small) - bu durumda model henüz indirilmemiş olabilir."""
    cfg = cfg or load_config()
    mid = cfg.get("whisper_model")
    if mid and resolve_whisper(mid):
        return mid
    if mid:
        return mid                     # seçilmiş ama indirilmemiş/silinmiş: kullanıcı uyarılsın
    for m in ("large-v3", "large-v3-turbo", "large-v2", "medium", "small", "base", "tiny"):
        if is_whisper_downloaded(m):
            return m
    for e in whisper_entries(cfg):
        if e["downloaded"]:
            return e["id"]
    return "large-v3" if cuda_available() else "small"


def selected_translate_model(cfg=None):
    m = (cfg or load_config()).get("translate_model")
    return m if m in [t[0] for t in TRANSLATE_MODELS] else TRANSLATE_MODEL


HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
_WANTED_FILES = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json")


class WhisperDownloadWorker(QThread):
    """Bir Whisper (CTranslate2) modelini Hugging Face'ten indirir. Kendi indiricimizdir:
    gerçek bayt ilerlemesi verir, iptal edilebilir ve kaldığı yerden devam eder."""
    progress = pyqtSignal(int, str)          # yüzde, "123 / 3090 MB"
    finished_ok = pyqtSignal(str)            # model kimliği
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, model_id, endpoint=None):
        super().__init__()
        self.model_id = model_id
        self.endpoint = (endpoint or HF_ENDPOINT).rstrip("/")
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def run(self):
        try:
            meta = next((m for m in WHISPER_MODELS if m["id"] == self.model_id), None)
            if meta is None:
                self.failed.emit("Bilinmeyen model: " + self.model_id)
                return
            dest = whisper_model_dir(self.model_id)
            os.makedirs(dest, exist_ok=True)
            marker = os.path.join(dest, ".complete")
            if os.path.exists(marker):
                self.finished_ok.emit(self.model_id)
                return
            r = requests.get(f"{self.endpoint}/api/models/{meta['repo']}/tree/main", timeout=20)
            r.raise_for_status()
            files = [(f["path"], int(f["size"])) for f in r.json()
                     if f.get("type") == "file" and (f["path"] in _WANTED_FILES or f["path"].startswith("vocabulary."))]
            if "model.bin" not in [p for p, _ in files]:
                self.failed.emit("Beklenmeyen dosya listesi (model.bin yok).")
                return
            total = sum(s for _, s in files) or 1
            done, last = 0, 0.0
            for path, size in files:
                final = os.path.join(dest, path)
                if os.path.isfile(final) and os.path.getsize(final) == size:
                    done += size
                    continue
                part = final + ".part"
                have = os.path.getsize(part) if os.path.isfile(part) else 0
                if have > size:
                    os.remove(part)
                    have = 0
                if have < size:
                    headers = {"Range": f"bytes={have}-"} if have else {}
                    with requests.get(f"{self.endpoint}/{meta['repo']}/resolve/main/{path}", headers=headers,
                                      stream=True, timeout=(10, 60)) as resp:
                        if resp.status_code not in (200, 206):
                            raise RuntimeError(f"{path}: HTTP {resp.status_code}")
                        append = resp.status_code == 206
                        got = have if append else 0
                        with open(part, "ab" if append else "wb") as f:
                            for chunk in resp.iter_content(256 * 1024):
                                if self._stop.is_set():
                                    self.cancelled.emit()
                                    return                     # .part kalır: sonra devam eder
                                f.write(chunk)
                                got += len(chunk)
                                if time.monotonic() - last >= 0.2:
                                    last = time.monotonic()
                                    d = done + got
                                    self.progress.emit(min(99, int(100 * d / total)),
                                                       f"{d // 1048576} / {total // 1048576} MB")
                if os.path.getsize(part) != size:
                    raise RuntimeError(f"{path}: eksik indi, tekrar deneyin.")
                os.replace(part, final)
                done += size
            with open(marker, "w") as f:
                f.write("ok")
            self.progress.emit(100, f"{total // 1048576} MB")
            self.finished_ok.emit(self.model_id)
        except requests.exceptions.RequestException as e:
            self.failed.emit(f"İndirilemedi (internet bağlantısı?): {e}")
        except Exception as e:
            self.failed.emit(f"İndirme hatası: {e}")


# =============================================================== SRT
TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _t(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def parse_srt(path):
    raw = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n").strip()
    cues = []
    for block in re.split(r"\n\s*\n", raw):
        lines = block.split("\n")
        if len(lines) < 3:
            continue
        m = TIME_RE.match(lines[1].strip())
        if not m:
            continue
        g = m.groups()
        cues.append({"n": len(cues) + 1, "start": _t(*g[:4]), "end": _t(*g[4:]),
                     "text": "\n".join(lines[2:]).strip()})
    return cues


def fmt_ts(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, cues, text_of=None):
    """cues: start/end/text sözlükleri. text_of(cue) verilirse metni o belirler,
    None dönerse o satır yazılmaz."""
    i = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in cues:
            txt = text_of(c) if text_of else c["text"]
            if txt is None:
                continue
            i += 1
            f.write(f"{i}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{txt}\n\n")


# ========================================================= Whisper -> altyazı satırları
MAX_LINE = 42
MAX_CHARS = 84


def _wrap(text):
    if len(text) <= MAX_LINE:
        return text
    mid = len(text) // 2
    best = None
    for m in re.finditer(" ", text):
        d = abs(m.start() - mid)
        if best is None or d < best[0]:
            best = (d, m.start())
    return text if best is None else text[:best[1]] + "\n" + text[best[1] + 1:]


def _split_text(text, level=0):
    """Uzun metni <=MAX_CHARS parçalara böler: önce cümle sonu, sonra virgül, sonra kelime."""
    if len(text) <= MAX_CHARS:
        return [text]
    pats = (r"(?<=[.!?…])\s+", r"(?<=[,;:])\s+")
    if level < len(pats):
        pieces = [p for p in re.split(pats[level], text) if p]
        if len(pieces) > 1:
            out, cur = [], ""
            for p in pieces:
                for q in _split_text(p, level + 1):
                    if cur and len(cur) + 1 + len(q) <= MAX_CHARS:
                        cur += " " + q
                    else:
                        if cur:
                            out.append(cur)
                        cur = q
            if cur:
                out.append(cur)
            return out
        return _split_text(text, level + 1)
    out, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > MAX_CHARS:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


def filter_segments(segs):
    """Sessizlik/müzikte Whisper'ın uydurduğu satırları ve aynı cümlenin art arda
    3'ten fazla tekrarını eler ("No. No. No." kalır, döngü atılır)."""
    out, last, rep = [], None, 0
    for s in segs:
        t = " ".join(s["text"].split())
        if not t:
            continue
        if s.get("no_speech_prob", 0.0) > 0.6 and s.get("avg_logprob", 0.0) < -1.0:
            continue
        norm = re.sub(r"\W+", "", t.lower())
        if norm == last:
            rep += 1
        else:
            last, rep = norm, 0
        if rep >= 3:
            continue
        out.append({**s, "text": t})
    return out


def segments_to_cues(segs):
    """Whisper bölümleri -> okunabilir altyazı satırları (<=84 karakter, <=2 satır)."""
    cues = []
    for s in segs:
        parts = _split_text(" ".join(s["text"].split()))
        total = sum(len(p) for p in parts) or 1
        dur = max(0.4, s["end"] - s["start"])
        acc = 0
        for p in parts:
            a = s["start"] + dur * acc / total
            acc += len(p)
            b = s["start"] + dur * acc / total
            cues.append({"start": a, "end": b, "text": _wrap(p)})
    prev_end = 0.0
    for c in cues:                           # çakışmayı ve sıfır süreyi düzelt
        c["start"] = max(c["start"], prev_end)
        c["end"] = max(c["end"], c["start"] + 0.4)
        prev_end = c["end"]
    return cues


# ============================================================ cümle birleştirme
SENT_END = (".", "?", "!", "…", '"', "”", "»", ")", "]", "♪", "。")
DASHES = ("-", "–", "—")


def build_units(cues, max_lines=4, max_chars=260, max_gap=1.0):
    """Cümle bitmeden kesilen ardışık satırları tek birim yapar (diyalog çizgili
    satırlar, cümle sonu noktalaması ve uzun sessizlik birleştirmeyi keser)."""
    groups, cur = [], []
    for i, c in enumerate(cues):
        if cur:
            prev = cues[cur[-1]]
            chars = sum(len(cues[j]["text"]) for j in cur) + len(c["text"])
            dialog = any(l.lstrip().startswith(DASHES) for l in prev["text"].split("\n")) \
                or c["text"].lstrip().startswith(DASHES)
            ok = (not prev["text"].rstrip().endswith(SENT_END) and c["start"] - prev["end"] <= max_gap
                  and len(cur) < max_lines and chars <= max_chars and not dialog)
            if not ok:
                groups.append(cur)
                cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    units = []
    for g in groups:
        text = cues[g[0]]["text"] if len(g) == 1 else \
            " ".join(cues[j]["text"].replace("\n", " ") for j in g)
        units.append({"id": cues[g[0]]["n"], "idx": g, "text": text,
                      "lens": [len(cues[j]["text"].replace("\n", " ")) for j in g]})
    return units


def split_text(tr, lens):
    """Çevrilmiş tek cümleyi orijinal satırların uzunluk oranına göre k parçaya böler
    (kelime sınırında; virgül sonrası tercih edilir). Her parça dolu, kelimeler korunur."""
    k = len(lens)
    words = tr.split()
    if k == 1:
        return [tr]
    if len(words) < k:
        return words + ["…"] * (k - len(words))
    ends, pos = [], 0
    for w in words:
        pos += len(w) + 1
        ends.append(pos)
    total = sum(lens) or k
    bounds, prev, acc = [0], 0, 0
    for part in range(k - 1):
        acc += lens[part]
        target = ends[-1] * acc / total
        lo, hi = prev + 1, len(words) - (k - 1 - part)
        best = None
        for j in range(lo, hi + 1):
            d = abs(ends[j - 1] - target)
            if words[j - 1].endswith((",", ";", ":", "—")):
                d -= 4
            if best is None or d < best[0]:
                best = (d, j)
        prev = best[1]
        bounds.append(prev)
    bounds.append(len(words))
    return [" ".join(words[bounds[i]:bounds[i + 1]]) for i in range(k)]


# ================================================================ Ollama
def ollama_models(host=OLLAMA_HOST, timeout=2.5):
    """-> ("down", []) | ("ok", [yüklü model adları])"""
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        return "ok", [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return "down", []


def model_installed(name, installed):
    return any(n == name or n.split("/")[-1] == name for n in installed)


def ollama_status(host=OLLAMA_HOST, model=TRANSLATE_MODEL, timeout=2.5):
    """-> "down" | "no_model" | "ok" """
    state, names = ollama_models(host, timeout)
    if state == "down":
        return "down"
    return "ok" if model_installed(model, names) else "no_model"


# =========================================================== çeviri mantığı
def build_prompt(src, dst, before, items, after):
    S, D = LANG_NAMES.get(src, src), LANG_NAMES.get(dst, dst)
    fmt = lambda t: t.replace("\n", " / ")
    body = "\n".join(f"[{i}] {fmt(t)}" for i, t in items)
    ctx = ""
    if before:
        ctx += "(Context, previous lines, DO NOT output: " + " | ".join(fmt(t) for t in before) + ")\n"
    if after:
        ctx += "(Following lines, context only, DO NOT output: " + " | ".join(fmt(t) for t in after) + ")\n"
    return (f"You are a professional {S} ({src}) to {D} ({dst}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {S} text "
            f"while adhering to {D} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {D} translation, without any additional explanations or commentary. "
            f"Keep every [number] marker exactly as is, one line per marker. "
            f"Please translate the following {S} text into {D}:\n\n\n{ctx}{body}")


def parse_reply(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = {}
    for m in re.finditer(r"^\s*\[(\d+)\]\s*(.*?)\s*$", text, re.M):
        out[int(m.group(1))] = m.group(2).replace(" / ", "\n")
    return out


class Translator:
    """Qt'siz çeviri çekirdeği: birim aralığını çevirir; tutmazsa yeniden dener,
    ikiye böler, en son tek tek çevirir. Sonuçlar self.result[cue_no] içinde toplanır."""

    def __init__(self, cues, src, dst=TARGET_LANG, model=TRANSLATE_MODEL, host=OLLAMA_HOST,
                 ctx=3, retries=3, temp=0.2, num_ctx=4096, log=None, stop=None):
        self.cues, self.src, self.dst = cues, src, dst
        self.model, self.host = model, host.rstrip("/")
        self.ctx, self.retries, self.temp, self.num_ctx = ctx, retries, temp, num_ctx
        self.units = build_units(cues)
        self.result = {}
        self.log = log or (lambda m: None)
        self.stop = stop or threading.Event()

    # -- Ollama
    def _ask(self, prompt, want, limit, attempt):
        try:
            r = requests.post(self.host + "/api/generate", stream=True, timeout=(10, 600), json={
                "model": self.model, "prompt": prompt, "stream": True,
                "options": {"temperature": min(1.0, self.temp + 0.25 * attempt),
                            "num_ctx": self.num_ctx, "num_predict": limit,
                            "repeat_penalty": 1.0 + 0.1 * min(attempt, 2)}})
        except requests.exceptions.ConnectionError:
            raise OllamaDown("Ollama'ya ulaşılamıyor. Ollama'yı açıp tekrar deneyin.")
        if r.status_code == 404:
            r.close()
            raise ModelMissing(f"'{self.model}' Ollama'da yüklü değil (ollama pull {self.model}).")
        if r.status_code != 200:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:200]}")
        buf = ""
        try:
            for line in r.iter_lines():
                if self.stop.is_set():
                    raise Stopped()
                if not line:
                    continue
                d = json.loads(line)
                if "error" in d:
                    raise RuntimeError(d["error"])
                piece = d.get("response", "")
                buf += piece
                if "\n" in piece or "]" in piece:
                    nums = [int(x) for x in re.findall(r"^\s*\[(\d+)\]", buf, re.M)]
                    if len(nums) != len(set(nums)):
                        raise Degenerate("model aynı satırı tekrarlamaya başladı")
                    if any(n not in want for n in nums):
                        break
                if d.get("done"):
                    break
        except requests.exceptions.ConnectionError:
            raise OllamaDown("Ollama bağlantısı koptu. Ollama'yı açıp tekrar deneyin.")
        finally:
            r.close()
        return buf

    # -- yayınlama
    def _publish(self, blk, got):
        for u in blk:
            tr = got[u["id"]]
            if len(u["idx"]) == 1:
                self.result[self.cues[u["idx"][0]]["n"]] = tr
            else:
                parts = split_text(tr.replace("\n", " "), u["lens"])
                for j, p in zip(u["idx"], parts):
                    self.result[self.cues[j]["n"]] = p

    # -- çeviri
    def translate_range(self, us, ue):
        units = self.units
        size = ue - us
        blk = units[us:ue]
        items = [(u["id"], u["text"]) for u in blk]
        src = dict(items)
        want = set(src)
        before = [u["text"] for u in units[max(0, us - self.ctx):us]]
        after = [u["text"] for u in units[ue:ue + self.ctx]]
        limit = int(sum(len(t) for t in src.values()) / 2) + 8 * size + 128
        attempts = 2 if size > 4 else self.retries
        label = f"{blk[0]['id']}-{blk[-1]['id']}"

        for attempt in range(attempts):
            try:
                got = parse_reply(self._ask(build_prompt(self.src, self.dst, before, items, after),
                                            want, limit, attempt))
                bad = [n for n in want if n in got and len(got[n]) > 3 * len(src[n]) + 60]
                if want <= set(got) and not bad:
                    self._publish(blk, got)
                    return
                self.log(f"Satır {label}: eksik/anormal çıktı, tekrar deneniyor ({attempt + 1}/{attempts})")
            except (Stopped, OllamaDown, ModelMissing):
                raise
            except Degenerate as e:
                self.log(f"Satır {label}: {e}, tekrar deneniyor ({attempt + 1}/{attempts})")
            except Exception as e:
                self.log(f"Hata: {e}")
                time.sleep(1.5)

        if size > 4:
            mid = (us + ue) // 2
            self.translate_range(us, mid)
            self.translate_range(mid, ue)
            return

        for k, u in enumerate(blk):                       # küçük aralık: birim birim
            res = None
            i = us + k
            for attempt in range(2):
                b = [x["text"] for x in units[max(0, i - self.ctx):i]]
                a = [x["text"] for x in units[i + 1:i + 1 + self.ctx]]
                try:
                    g = parse_reply(self._ask(
                        build_prompt(self.src, self.dst, b, [(u["id"], u["text"])], a),
                        {u["id"]}, len(u["text"]) + 128, attempt))
                    if u["id"] in g and len(g[u["id"]]) <= 3 * len(u["text"]) + 60:
                        res = g[u["id"]]
                        break
                except (Stopped, OllamaDown, ModelMissing):
                    raise
                except Exception:
                    pass
            if res is None:
                self.log(f"Satır {u['id']} çevrilemedi, orijinali bırakıldı")
                res = u["text"]
            self._publish([u], {u["id"]: res})


# ============================================================ Qt işçileri
class TranslateWorker(QThread):
    """Kaynak .srt'yi Türkçeye çevirir. Sıra: oynatma konumundan başlayıp sona,
    sonra başa dönüp kalanlar. Her blok sonrası 'canlı' dosya güncellenir."""
    progress = pyqtSignal(int, int)          # çevrilen satır, toplam satır
    live_updated = pyqtSignal(str)           # canlı altyazı dosyası güncellendi
    finished_ok = pyqtSignal(str)            # nihai film.ai.tr.srt
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, source_path, src_lang, out_path, get_pos, model=TRANSLATE_MODEL,
                 host=OLLAMA_HOST, block=10, ctx=3, dst_lang=TARGET_LANG, live=True):
        super().__init__()
        self.source_path, self.src_lang, self.out_path = source_path, src_lang, out_path
        self.get_pos, self.model, self.host = get_pos, model, host
        self.block, self.ctx = block, ctx
        self.dst_lang, self.live = dst_lang, live      # live=False: tüm dosya baştan sona çevrilir
        self._stop = threading.Event()
        key = hashlib.md5(out_path.encode("utf-8", "replace")).hexdigest()[:10]
        self.live_path = os.path.join(tempfile.gettempdir(), f"uhf_live_{key}.srt")

    def cancel(self):
        self._stop.set()

    def _pos(self):
        try:
            return float(self.get_pos() or 0.0)
        except Exception:
            return 0.0

    def _write_live(self, cues, result):
        tmp = self.live_path + ".tmp"
        write_srt(tmp, cues, lambda c: result.get(c["n"]))
        _replace(tmp, self.live_path)

    def run(self):
        try:
            cues = parse_srt(self.source_path)
            if not cues:
                self.failed.emit("Kaynak altyazı boş.")
                return
            tr = Translator(cues, self.src_lang, self.dst_lang, self.model, self.host,
                            ctx=self.ctx, stop=self._stop)
            units, done = tr.units, set()
            last_emit = 0.0
            while True:
                if self._stop.is_set():
                    self.cancelled.emit()
                    return
                remaining = [i for i in range(len(units)) if i not in done]
                if not remaining:
                    break
                pos = self._pos() if self.live else 0.0
                start = next((i for i in remaining
                              if cues[units[i]["idx"][-1]]["end"] >= pos - 1.0), remaining[0])
                rs, end = set(remaining), start + 1
                while end < len(units) and end - start < self.block and end in rs:
                    end += 1
                tr.translate_range(start, end)
                done.update(range(start, end))
                self.progress.emit(len(tr.result), len(cues))
                if self.live and time.monotonic() - last_emit >= 3.0:
                    self._write_live(cues, tr.result)
                    self.live_updated.emit(self.live_path)
                    last_emit = time.monotonic()

            part = self.out_path + ".part"
            write_srt(part, cues, lambda c: tr.result.get(c["n"], c["text"]))
            _replace(part, self.out_path)
            if self.live:
                self._write_live(cues, tr.result)
                self.live_updated.emit(self.live_path)
            self.finished_ok.emit(self.out_path)
        except Stopped:
            self.cancelled.emit()
        except (OllamaDown, ModelMissing) as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"Beklenmeyen hata: {e}")


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def fw_import_error():
    """faster-whisper'ı GERÇEKTEN import etmeyi dener. Başarılıysa None, değilse
    gerçek hata metni (son zincir halkaları dahil) döner."""
    import traceback
    try:
        add_cuda_dll_dirs()
        import faster_whisper  # noqa: F401
        from faster_whisper import WhisperModel  # noqa: F401
        return None
    except BaseException as e:                      # ImportError, OSError (DLL), ...
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        lines = traceback.format_exception_only(type(e), e)
        cause = e.__cause__ or e.__context__
        while cause is not None and len(lines) < 4:
            lines += traceback.format_exception_only(type(cause), cause)
            cause = cause.__cause__ or cause.__context__
        return "".join(lines).strip()


def fw_missing_message(err=None):
    """faster-whisper yüklenemediğinde gösterilecek metin. Gerçek hata (err) varsa
    onu da gösterir; bileşen gerçekten yoksa kurulum/derleme yolunu söyler."""
    real_missing = (err is None) or ("No module named 'faster_whisper'" in err) \
        or ("import of faster_whisper halted" in err)
    if real_missing:
        if is_frozen():
            return ("Bu sürümde ses tanıma bileşeni (faster-whisper) pakete dahil edilmemiş.\n\n"
                    "Programı derlerken şu bayrakları ekleyip yeniden derleyin:\n"
                    "    --collect-data faster_whisper --collect-binaries ctranslate2")
        return "faster-whisper kurulu değil.\nKomut: pip install faster-whisper"
    return ("Ses tanıma bileşeni (faster-whisper) pakette var ama yüklenemedi.\n"
            "Asıl hata:\n\n" + err)


def have_faster_whisper():
    import importlib.util
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ValueError, ImportError):
        return sys.modules.get("faster_whisper") is not None


def add_cuda_dll_dirs():
    """Windows: CUDA DLL'leri (cublas/cudnn) exe'nin yanına/'cuda' klasörüne ya da
    nvidia pip paketlerine konulmuşsa ctranslate2'nin bulabilmesi için arama yoluna ekle.
    Hiçbiri yoksa zararsızdır; GPU yoksa zaten işlemciye düşülür."""
    if sys.platform != "win32":
        return
    bases = []
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        bases.append(mp)
    bases.append(os.path.dirname(os.path.abspath(sys.executable if is_frozen() else __file__)))
    dirs = []
    for b in bases:
        dirs += [b, os.path.join(b, "cuda"), os.path.join(b, "_internal"),
                 os.path.join(b, "_internal", "cuda")]
        nv = os.path.join(b, "nvidia")
        if os.path.isdir(nv):                       # nvidia-cublas-cu12 / cudnn paketleri
            for n in os.listdir(nv):
                dirs.append(os.path.join(nv, n, "bin"))
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


XXL_HINT = ("\n\nAlternatif: Subtitle Edit'te kurulu 'Purfview Faster Whisper XXL' bulunursa "
            "program onu otomatik kullanır (Subtitle Edit > Video > Sesten metne > motor indir).")


def _se_roots():
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    base = os.path.join(appdata, "Subtitle Edit")
    return [os.path.join(base, "SpeechToText"), os.path.join(base, "Whisper")]


def find_xxl_exe(cfg=None):
    """Subtitle Edit'in indirdiği faster-whisper-xxl programını bulur (yoksa None)."""
    import shutil
    names = ["faster-whisper-xxl.exe", "faster-whisper-xxl"]
    cfg = cfg if cfg is not None else load_config()
    p = cfg.get("xxl_exe")
    if p and os.path.isfile(p):
        return p
    for root in _se_roots():
        for n in names:
            c = os.path.join(root, "Purfview-Faster-Whisper-XXL", n)
            if os.path.isfile(c):
                return c
    return shutil.which("faster-whisper-xxl")


def xxl_model_args(exe, model_path):
    """(model_dir, model_adı): faster-whisper-xxl 'faster-whisper-<ad>' klasörünü ister."""
    mp = model_path.rstrip("\\/")
    base = os.path.basename(mp)
    if base.startswith("faster-whisper-"):
        return os.path.dirname(mp), base[len("faster-whisper-"):]
    alt = os.path.join(os.path.dirname(exe), "_models", "faster-whisper-" + base)
    if os.path.isfile(os.path.join(alt, "model.bin")):
        return os.path.dirname(alt), base
    raise RuntimeError(
        f"'{base}' modeli faster-whisper-xxl'in beklediği klasör düzeninde değil.\n"
        "📦 Model Ayarları'ndan Subtitle Edit'ten gelen bir model "
        "(ör. 'Subtitle Edit: faster-whisper-large-v3') seçin.")


def cuda_available():
    add_cuda_dll_dirs()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class TranscribeWorker(QThread):
    """Filmin sesini faster-whisper ile altyazıya çevirir -> film.ai.<dil>.srt
    (model_path: yerel CTranslate2 model klasörü; indirme 📦 penceresinde yapılır)."""
    progress = pyqtSignal(int, str)          # yüzde (-1 = belirsiz), durum metni
    finished_ok = pyqtSignal(str, str)       # srt yolu, dil
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, video, ff_index=None, model_path=None, model_label="", ffmpeg="ffmpeg",
                 no_window=0, language=None):
        super().__init__()
        self.video, self.ff_index = video, ff_index
        self.model_path, self.model_label = model_path, model_label or "model"
        self.ffmpeg, self.no_window, self.language = ffmpeg, no_window, language
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    # -- ffmpeg: seçili ses izi -> 16 kHz mono wav
    def _extract(self, wav):
        cmd = [self.ffmpeg, "-y", "-v", "error", "-i", self.video, "-map",
               f"0:{self.ff_index}" if self.ff_index is not None else "0:a:0",
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             creationflags=self.no_window)
        while p.poll() is None:
            if self._stop.is_set():
                p.terminate()
                p.wait()
                raise Stopped()
            time.sleep(0.2)
        if p.returncode != 0:
            err = p.stderr.read().decode(errors="replace")[:300]
            raise RuntimeError("Ses çıkarılamadı: " + err)

    def run(self):
        wav = None
        part = None
        try:
            existing = find_sources(self.video)
            if existing:                                   # zaten var: tekrar üretme
                lang, path = existing[0]
                self.finished_ok.emit(path, lang)
                return
            if not self.model_path or not os.path.isdir(self.model_path):
                self.failed.emit("Whisper modeli bulunamadı. 📦 Model Ayarları'ndan indirin.")
                return
            # 1) Python içi faster-whisper; 2) olmazsa Subtitle Edit'in faster-whisper-xxl programı
            WhisperModel, xxl = None, None
            fw_err = fw_import_error()
            if fw_err is None:
                from faster_whisper import WhisperModel
            else:
                xxl = find_xxl_exe()
                if not xxl:
                    self.failed.emit(fw_missing_message(fw_err) + XXL_HINT)
                    return

            self.progress.emit(-1, "Ses çıkarılıyor…")
            fd, wav = tempfile.mkstemp(suffix=".wav", prefix="uhf_asr_")
            os.close(fd)
            self._extract(wav)

            segs, lang, last_err = None, None, None
            if xxl:
                try:
                    segs, lang = self._transcribe_xxl(xxl, wav)
                except Stopped:
                    raise
                except Exception as e:
                    last_err = e
            else:
                # Tanıma (CUDA yoksa/başarısızsa işlemciye düşer)
                devices = ([("cuda", "float16")] if cuda_available() else []) + [("cpu", "int8")]
                for dev, ctype in devices:
                    try:
                        segs, lang = self._transcribe(WhisperModel, wav, dev, ctype)
                        break
                    except Stopped:
                        raise
                    except Exception as e:
                        last_err = e
                        if dev == "cuda":
                            self.progress.emit(-1, "GPU kullanılamadı, işlemci ile devam ediliyor…")
            if segs is None:
                self.failed.emit(f"Ses tanıma başarısız: {last_err}")
                return

            cues = segments_to_cues(filter_segments(segs))
            if not cues:
                self.failed.emit("Seste konuşma bulunamadı.")
                return
            out = final_path(self.video, lang)
            part = out + ".part"
            write_srt(part, cues)
            _replace(part, out)
            part = None
            self.finished_ok.emit(out, lang)
        except Stopped:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(f"Beklenmeyen hata: {e}")
        finally:
            for p in (wav, part):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def _transcribe_xxl(self, exe, wav):
        """Subtitle Edit'in kullandığı bağımsız faster-whisper-xxl programı ile tanıma.
        Python/faster-whisper kurulumu gerekmez. -> (segmentler, dil)"""
        import queue
        import wave
        parent, name = xxl_model_args(exe, self.model_path)
        try:
            with wave.open(wav, "rb") as w:
                total = w.getnframes() / float(w.getframerate() or 16000)
        except Exception:
            total = 0.0
        outdir = tempfile.mkdtemp(prefix="uhf_xxl_")
        cmd = [exe, "--model", name, "--model_dir", parent, "--output_format", "srt",
               "--output_dir", outdir, "--beep_off"]
        if self.language:
            cmd += ["--language", self.language]
        cmd.append(wav)
        lang, tail = self.language, []
        proc = None
        try:
            proc = subprocess.Popen(cmd, cwd=os.path.dirname(exe), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, creationflags=self.no_window)
            q = queue.Queue()

            def reader():
                for raw in iter(proc.stdout.readline, b""):
                    q.put(raw.decode("utf-8", "replace").rstrip())
                q.put(None)
            threading.Thread(target=reader, daemon=True).start()

            seg_re = re.compile(r"^\[(?:(\d\d):)?(\d\d):(\d\d)[.,]\d+ --> (?:(\d\d):)?(\d\d):(\d\d)[.,](\d+)\]")
            pct_re = re.compile(r"^\s*(\d{1,3})%\s*\|")
            det_re = re.compile(r"[Dd]etected language '?([A-Za-z]{2,3})'?")
            label = f"Altyazı üretiliyor ({name}, faster-whisper-xxl)…"
            while True:
                if self._stop.is_set():
                    raise Stopped()
                try:
                    line = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if line is None:
                    break
                tail = (tail + [line])[-6:]
                m = det_re.search(line)
                if m and not self.language:
                    lang = m.group(1).lower()
                m = seg_re.match(line)
                if m and total:
                    end = (int(m.group(4) or 0) * 3600 + int(m.group(5)) * 60 + int(m.group(6)))
                    self.progress.emit(int(min(99, 100 * end / total)), label)
                    continue
                m = pct_re.match(line)
                if m:
                    self.progress.emit(min(99, int(m.group(1))), label)
            proc.wait()
            srts = glob.glob(os.path.join(outdir, "*.srt"))
            if not srts:
                raise RuntimeError("faster-whisper-xxl çıktı üretmedi:\n" + "\n".join(tail[-4:]))
            cues = parse_srt(srts[0])
            segs = [{"start": c["start"], "end": c["end"], "text": c["text"],
                     "no_speech_prob": 0.0, "avg_logprob": 0.0} for c in cues]
            return segs, (lang or "en")
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except Exception:
                    proc.kill()
            import shutil
            shutil.rmtree(outdir, ignore_errors=True)

    def _transcribe(self, WhisperModel, wav, device, ctype):
        model = WhisperModel(self.model_path, device=device, compute_type=ctype)
        try:
            seg_iter, info = model.transcribe(
                wav, language=self.language, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False, beam_size=5 if device == "cuda" else 1)
            lang = info.language
            total = info.duration or 0.0
            segs = []
            for s in seg_iter:
                if self._stop.is_set():
                    raise Stopped()
                segs.append({"start": s.start, "end": s.end, "text": s.text,
                             "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
                             "avg_logprob": getattr(s, "avg_logprob", 0.0)})
                pct = int(min(99, 100 * s.end / total)) if total else -1
                self.progress.emit(pct, f"Altyazı üretiliyor ({self.model_label}, dil: {lang}, {device})…")
            return segs, lang
        finally:
            del model                      # GPU belleği çeviri modeli için boşalsın
            import gc
            gc.collect()
