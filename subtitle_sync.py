"""Altyazı senkronlama: elindeki (zamanlaması bozuk) altyazıyı, filmin sesinden üretilmiş
referans altyazıya (Whisper) göre hizalar.

Fikir: İki altyazıda da aynı konuşmalar vardır, yalnızca zamanları farklıdır. Referansın
zamanları filmle uyumludur; kullanıcının altyazısının metni (çevirisi) değerlidir. Zamanları
referanstan alıp metni kullanıcıdan koruruz.

Aşamalar
  1. Kaba hizalama: iki altyazının "konuşma var/yok" sinyalleri çapraz ilinti (FFT) ile
     karşılaştırılır. Birden çok ölçek (fps farkı: 25/23.976 vb.) denenir -> ölçek + kayma.
     Dilden bağımsızdır (İngilizce referans, Türkçe altyazı sorun olmaz).
  2. İnce eşleştirme: kaba dönüşümle her satır için aday referans satırları bulunur; zaman
     örtüşmesi + (varsa) metin benzerliği ile puanlanır; sıra bozulmayan en iyi eşleşme zinciri
     seçilir (ağırlıklı en uzun artan altdizi).
  3. Sağlam doğrusal uyum: eşleşmelerden t' = a*t + b bulunur (aykırılar atılır). Filmde
     kesinti/ek sahne varsa ilk geçişte eşleşmeyen satırlar için kaba hizalama yeniden
     yapılır -> parça parça (piecewise) dönüşüm.

Bu modül Qt'ye bağımlı DEĞİLDİR (yalnızca numpy). Alttaki SyncWorker, Qt varsa kullanılır.

Komut satırı:  python subtitle_sync.py kullanici.srt referans.srt cikti.srt
"""
import bisect
import math
import os
import re
import sys
import unicodedata

import numpy as np

DT = 0.1                      # konuşma sinyalinin zaman çözünürlüğü (sn)
Z_MIN = 5.0                   # kaba hizalama güveni (tepe noktasının z-skoru) alt sınırı
MAX_PASSES = 1                # genel hizalama geçişi; geri kalanı 'lego' parçalama ile bulunur
LEGO_ROUNDS = 3               # eşleşmeyen boşluklar için parça-parça arama turu
PIECE = 24                    # 'lego' parçalama: tek seferde hizalanan satır sayısı
FILL_MIN_JUMP = 8.0           # eksik sahne sayılması için kayma sıçraması (sn)
FILL_MIN_CUES = 2             # referanstan doldurulacak 'eksik sahne' en az kaç ardışık satır
INLIER_TOL = 1.0              # eşleşme "iyi" sayılması için kabul edilen hata (sn)

# Sık görülen kare hızı oranları (ref_zamanı = oran * kullanıcı_zamanı) ve tersleri
_FPS = [23.976, 24.0, 25.0, 29.97, 30.0]
COMMON_RATIOS = sorted({round(a / b, 6) for a in _FPS for b in _FPS if a != b} | {1.0})


class SyncFailed(Exception):
    """Altyazılar yeterince örtüşmüyor (yanlış film, çok farklı kesim vb.)."""


class Stopped(Exception):
    pass


# ============================================================== SRT okuma / yazma
TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _t(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def read_text(path):
    """İndirilen altyazılar sık sık Windows-1254 (Türkçe) / Latin-1 olur; sırayla dener."""
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8-sig", "utf-16", "cp1254", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def parse_srt(path):
    raw = read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    cues, cur = [], None
    for line in raw.split("\n"):
        m = TIME_RE.search(line)
        if m:
            if cur:
                # blok arasında boş satır yoksa bir sonraki bloğun sıra numarası metne sızar
                if len(cur["lines"]) > 1 and cur["lines"][-1].strip().isdigit():
                    cur["lines"].pop()
                cues.append(cur)
            g = m.groups()
            cur = {"start": _t(*g[:4]), "end": _t(*g[4:]), "lines": []}
        elif cur is not None:
            if line.strip() == "":
                cues.append(cur)
                cur = None
            else:
                cur["lines"].append(line)
    if cur:
        cues.append(cur)
    return [{"start": c["start"], "end": max(c["end"], c["start"]),
             "text": "\n".join(c["lines"]).strip()} for c in cues]


def fmt_ts(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, cues):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{c['text']}\n\n")


# ============================================================== 1) kaba hizalama
def _activity(starts, ends, n, dt=DT):
    """Aralıklardan 0/1 'konuşma var' sinyali (uzunluk n)."""
    i0 = np.clip((starts / dt).astype(np.int64), 0, n - 1)
    i1 = np.clip(np.ceil(ends / dt).astype(np.int64), 0, n)
    i1 = np.maximum(i1, i0 + 1)
    d = np.zeros(n + 1, dtype=np.float32)
    np.add.at(d, i0, 1.0)
    np.add.at(d, np.minimum(i1, n), -1.0)
    return (np.cumsum(d[:n]) > 0).astype(np.float32)


def coarse_align(ref, user, should_stop=None, dt=DT, scales=None):
    """ref_zamanı ≈ s * kullanıcı_zamanı + o olacak şekilde (s, o, z) bulur.
    z: tepe noktasının ilinti dağılımına göre z-skoru (güven)."""
    if len(ref) < 3 or len(user) < 3:
        raise SyncFailed("Hizalamak için yeterli altyazı satırı yok.")
    rs = np.array([c["start"] for c in ref]); re_ = np.array([c["end"] for c in ref])
    us = np.array([c["start"] for c in user]); ue = np.array([c["end"] for c in user])

    na = int(re_.max() / dt) + 2
    A = _activity(rs, re_, na, dt)
    A = A - A.mean()
    norm_a = float(np.sqrt((A * A).sum())) or 1.0
    max_s = 1.30
    nb_max = int(ue.max() * max_s / dt) + 2
    n = 1 << int(math.ceil(math.log2(na + nb_max + 2)))
    FA = np.fft.rfft(A, n)

    def score(s):
        nb = int(ue.max() * s / dt) + 2
        B = _activity(us * s, ue * s, nb, dt)
        B = B - B.mean()
        norm_b = float(np.sqrt((B * B).sum())) or 1.0
        c = np.fft.irfft(FA * np.conj(np.fft.rfft(B, n)), n)
        vals = np.concatenate([c[n - (nb - 1):], c[:na]]) / (norm_a * norm_b)
        k = int(np.argmax(vals))
        lag = k - (nb - 1)
        z = (float(vals[k]) - float(vals.mean())) / (float(vals.std()) or 1.0)
        return float(vals[k]), lag * dt, z

    def best_of(scales):
        best = None
        for s in scales:
            if should_stop and should_stop():
                raise Stopped()
            p, o, z = score(float(s))
            if best is None or p > best[0]:
                best = (p, float(s), o, z)
        return best

    if scales is not None:                       # ölçek bilinen: yalnızca kayma aranır
        _, s, o, z = best_of(list(scales))
        return s, o, z
    grid = list(np.arange(0.975, 1.0251, 0.0005)) + COMMON_RATIOS
    p, s, o, z = best_of(grid)
    p, s, o, z = best_of(list(np.arange(s - 0.0006, s + 0.00061, 0.00005)) + [s])
    return s, o, z


# ============================================================== metin benzerliği
def _norm_text(t):
    t = re.sub(r"<[^>]+>|\{\\[^}]*\}", " ", t)
    t = t.replace("İ", "i").replace("I", "ı").lower().replace("ı", "i")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[\W_]+", " ", t).strip()
    return t


def _grams(t):
    t = _norm_text(t)
    s = " " + t + " "
    return {s[k:k + 3] for k in range(len(s) - 2)} if len(t) >= 3 else set(), set(re.findall(r"\d{2,}", t))


class TextScorer:
    """İki satırın metin benzerliği (0..1). Aynı dilde: 3'lü karakter benzerliği.
    Farklı dilde ve `embedder` verilmişse: çok dilli cümle gömmeleri (kosinüs)."""

    def __init__(self, user, ref, embedder=None):
        self.user, self.ref, self.embedder = user, ref, embedder
        self._ug = [None] * len(user)
        self._rg = [None] * len(ref)
        self.mode = "trigram"
        self._ue = self._re = None

    def _g(self, cache, cues, k):
        if cache[k] is None:
            cache[k] = _grams(cues[k]["text"])
        return cache[k]

    def trigram(self, i, j):
        (ga, na), (gb, nb) = self._g(self._ug, self.user, i), self._g(self._rg, self.ref, j)
        sim = len(ga & gb) / len(ga | gb) if ga and gb else 0.0
        if na & nb:
            sim = min(1.0, sim + 0.3)
        return sim

    def calibrate(self, pairs):
        """Olası eşleşmeler (zamanca örtüşenler) arasında metin benzerliği çok düşükse diller
        farklıdır -> gömme modeline geç (varsa)."""
        if self.embedder is None:
            return
        sims = sorted(self.trigram(i, j) for i, j, ov in pairs if ov >= 0.4)[:]
        if len(sims) < 20:
            return
        if sims[len(sims) // 2] < 0.12:
            ui = sorted({i for i, _, _ in pairs}); rj = sorted({j for _, j, _ in pairs})
            ue = self.embedder([self.user[i]["text"] for i in ui])
            re_ = self.embedder([self.ref[j]["text"] for j in rj])
            self._ue = {i: ue[k] for k, i in enumerate(ui)}
            self._re = {j: re_[k] for k, j in enumerate(rj)}
            self.mode = "embed"

    def __call__(self, i, j):
        if self.mode == "embed":
            a, b = self._ue.get(i), self._re.get(j)
            if a is None or b is None:
                return 0.0
            cos = float(np.dot(a, b))
            return min(1.0, max(0.0, (cos - 0.3) / 0.5))
        return self.trigram(i, j)


# ============================================================== 2) ince eşleştirme
def _candidates(user, ref, ui_list, rj_list, a, b, win, rmid_sorted, rmid_order):
    """Her kullanıcı satırı için dönüşüm sonrası orta noktası `win` sn içindeki referans
    satırları -> [(i, j, zaman_örtüşmesi_IoU)]"""
    out = []
    for i in ui_list:
        c = user[i]
        ps, pe = a * c["start"] + b, a * c["end"] + b
        pm = (ps + pe) / 2
        lo = bisect.bisect_left(rmid_sorted, pm - win)
        hi = bisect.bisect_right(rmid_sorted, pm + win)
        for k in range(lo, hi):
            j = rmid_order[k]
            r = ref[j]
            inter = min(pe, r["end"]) - max(ps, r["start"])
            union = max(pe, r["end"]) - min(ps, r["start"])
            iou = max(0.0, inter) / union if union > 0 else 0.0
            out.append((i, j, iou))
    return out


def _prune_unsupported(cur, n, half=8, strong=0.4):
    """Tesadüfi eşleşmeleri ayıklar. Gerçek bölgelerde komşu satırların çoğunun güçlü bir
    eşi vardır (zaman örtüşmesi yüksek); başka kesimin bölgesinde ise yalnızca ara sıra
    rastlantısal örtüşme olur. Satır dizisi boyunca 'güçlü eşleşme' oranı (eşleşmeyen = 0)
    pencere ortalamasıyla düşük olan eşleşmeler atılır (satır havuza geri döner).
    Eşik, en iyi bölgelerin yoğunluğuna göre uyarlanır (gerçek altyazılar daha gürültülüdür)."""
    wv = np.zeros(n)
    for i, (_, w, _) in cur.items():
        wv[i] = 1.0 if w >= strong else 0.0
    sm = np.convolve(wv, np.ones(2 * half + 1) / (2 * half + 1), mode="same")
    idx = np.array(sorted(cur))
    q80 = float(np.percentile(sm[idx], 80))
    thr = min(0.55, max(0.2, 0.6 * q80))
    return {i: v for i, v in cur.items() if sm[i] >= thr}


def _drop_sparse(matched, user, rmid, a, half=6, tol=0.8, need=3):
    """Komşu desteği olmayan eşleşmeleri atar. Gerçek bölgede çevredeki satırların çoğu aynı
    kaymayla eşleşmiştir; başka bölgelere düşen tesadüfi eşleşmeler ise seyrek ve birbirinden
    bağımsızdır. (±half satır içinde, kayması tol sn içinde uyuşan en az `need` komşu aranır.)"""
    ids = sorted(matched)
    d = {i: rmid[matched[i][0]] - a * (user[i]["start"] + user[i]["end"]) / 2 for i in ids}   # ölçek (fps) etkisi çıkarılır
    keep = {}
    for i in ids:
        ok = sum(1 for k in ids if k != i and abs(k - i) <= half and abs(d[k] - d[i]) <= tol)
        if ok >= need:
            keep[i] = matched[i]
    return keep


def _min_strong(n, lags, p=0.3, alpha=0.02):
    """n satırlık parçada, tesadüfen (her satırın güçlü eşleşme olasılığı p) en iyi kaymayı
    seçmiş olma ihtimalini `alpha` altında tutan en az güçlü eşleşme sayısı. Aranan kayma
    sayısı (lags) kadar şans verildiği için eşik buna göre yükselir (çoklu karşılaştırma)."""
    for k in range(1, n + 1):
        tail = sum(math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(k, n + 1))
        if tail * max(1, lags) <= alpha:
            return k
    return n + 1


def _best_chain(pairs, nj):
    """pairs: [(i, j, w)] -> hem i hem j'de kesin artan, toplam ağırlığı en büyük zincir
    (ağırlıklı en uzun artan altdizi; Fenwick ağacı ile O(P log N)). -> pairs indeksleri"""
    if not pairs:
        return []
    order = sorted(range(len(pairs)), key=lambda k: (pairs[k][0], pairs[k][1]))
    size = nj + 2
    tv = [0.0] * (size + 1)
    tp = [-1] * (size + 1)

    def query(pos):
        v, p = 0.0, -1
        while pos > 0:
            if tv[pos] > v:
                v, p = tv[pos], tp[pos]
            pos -= pos & -pos
        return v, p

    def update(pos, v, p):
        while pos <= size:
            if v > tv[pos]:
                tv[pos], tp[pos] = v, p
            pos += pos & -pos

    best = [0.0] * len(pairs)
    prev = [-1] * len(pairs)
    idx = 0
    while idx < len(order):
        i = pairs[order[idx]][0]
        group = []
        while idx < len(order) and pairs[order[idx]][0] == i:
            group.append(order[idx])
            idx += 1
        for k in group:
            v, p = query(pairs[k][1])          # j' < j  (konum = j'+1 <= j)
            best[k], prev[k] = v + pairs[k][2], p
        for k in group:
            update(pairs[k][1] + 1, best[k], k)
    k = max(range(len(pairs)), key=lambda q: best[q])
    chain = []
    while k != -1:
        chain.append(k)
        k = prev[k]
    return chain[::-1]


# ============================================================== 3) sağlam doğrusal uyum
def _robust_line(x, y, a0, b0, free_scale):
    """y ≈ a*x + b. Aykırıları atan yinelemeli uyum. free_scale=False -> a = a0 sabit."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    a, b = a0, b0
    keep = np.ones(len(x), bool)
    for _ in range(8):
        r = y - (a * x + b)
        med = np.median(r[keep]) if keep.any() else 0.0
        mad = 1.4826 * np.median(np.abs(r[keep] - med)) if keep.any() else 1.0
        tol = max(1.0, 2.5 * mad)
        nk = np.abs(r - med) <= tol
        if nk.sum() < 3:
            break
        keep = nk
        xs, ys = x[keep], y[keep]
        if free_scale and keep.sum() >= 12 and (xs.max() - xs.min()) >= 120:
            a_new, b_new = np.polyfit(xs, ys, 1)
            if abs(a_new - a0) <= 0.01:
                a, b = float(a_new), float(b_new)
                continue
        a = a0
        b = float(np.median(ys - a * xs))
    return a, b, keep


def sync_cues(user, ref, progress=None, should_stop=None, embedder=None, fill_missing=True):
    """user, ref: [{'start','end','text'}]. -> (yeni_satırlar, rapor)
    fill_missing: kullanıcı altyazısında karşılığı olmayan referans satırları (filmdeki ek
    sahneler) italik olarak referanstan eklenir."""
    def say(p, msg):
        if progress:
            progress(p, msg)

    user = sorted(user, key=lambda c: c["start"])
    ref = sorted(ref, key=lambda c: c["start"])
    n, m = len(user), len(ref)
    if n < 3 or m < 3:
        raise SyncFailed("Altyazılardan biri çok kısa (en az 3 satır gerekir).")

    rmid = np.array([(c["start"] + c["end"]) / 2 for c in ref])
    rmid_order = list(np.argsort(rmid, kind="stable"))
    rmid_sorted = [float(rmid[k]) for k in rmid_order]
    scorer = TextScorer(user, ref, embedder)

    matched = {}               # i -> (j, w, hid)
    hyps = []                  # hid -> (a, b, z)
    first_z = None

    for pass_no in range(MAX_PASSES):
        used_j = {v[0] for v in matched.values()}
        ru = [i for i in range(n) if i not in matched]
        rr = [j for j in range(m) if j not in used_j]
        if len(ru) < 6 or len(rr) < 6:
            break
        say(10 + pass_no * 15, "Konuşma desenleri karşılaştırılıyor…" if pass_no == 0
            else "Filmdeki kesinti/ek sahne için ek hizalama aranıyor…")
        try:
            s, o, z = coarse_align([ref[j] for j in rr], [user[i] for i in ru], should_stop)
        except SyncFailed:
            if pass_no == 0:
                raise
            break
        if pass_no == 0:
            first_z = z
        if z < Z_MIN:
            if pass_no == 0:
                raise SyncFailed("Altyazılar arasında yeterli benzerlik bulunamadı "
                                 "(farklı bir film/sürüm olabilir).")
            break

        a, b = s, o
        new_pairs = []
        for it in range(2):
            say(30 + pass_no * 15 + it * 5, "Satırlar eşleştiriliyor…")
            cands = _candidates(user, ref, ru, rr, a, b, 2.5 if it == 0 else 1.5,
                                rmid_sorted, rmid_order)
            if pass_no == 0 and it == 0:
                scorer.calibrate(cands)
            pairs = []
            for i, j, iou in cands:
                w = iou + 0.5 * scorer(i, j)
                if w >= 0.2:
                    pairs.append((i, j, w))
            if it == 1:                       # önceki geçişlerin eşleşmelerini koru
                for i, (j, w, hid) in matched.items():
                    pairs.append((i, j, w + 10.0))
            chain = _best_chain(pairs, m)
            sel = [pairs[k] for k in chain]
            new_pairs = [(i, j, w) for i, j, w in sel if i not in matched]
            if len(new_pairs) < 6:
                break
            if it == 0:
                x = [(user[i]["start"] + user[i]["end"]) / 2 for i, _, _ in new_pairs]
                y = [rmid[j] for _, j, _ in new_pairs]
                a, b, _ = _robust_line(x, y, a, b, free_scale=True)
        if len(new_pairs) < max(6, int(0.02 * n)):
            if pass_no == 0:
                raise SyncFailed("Güvenilir satır eşleşmesi bulunamadı.")
            break
        hid = len(hyps)
        # Önceki geçişlerin eşleşmeleri + yeniler; gerçek ağırlıklarıyla
        cur = {i: matched[i] if i in matched else (j, w, hid) for i, j, w in sel}
        pruned = _prune_unsupported(cur, n)
        if len(pruned) - len([i for i in pruned if i in matched]) < 6:
            if pass_no == 0 and len(pruned) < 6:
                pruned = cur                  # ilk geçişte aşırı budama: olduğu gibi bırak
            elif pass_no > 0:
                break
        hyps.append((a, b, z))
        matched = pruned
        if len(matched) >= 0.9 * n:
            break

    if len(matched) < 6:
        raise SyncFailed("Güvenilir satır eşleşmesi bulunamadı.")

    # ---- 'lego': eşleşmeyen boşlukları, iki komşu eşleşmenin arasındaki referans aralığında
    # parça parça (PIECE satır) yeniden ara. Filmdeki her ek sahne/kesinti kendi kaymasını bulur;
    # referanstaki aralık komşu eşleşmelerle sınırlı olduğundan sıra bozulmaz, sahte eşleşme azalır.
    a_glob = hyps[0][0]
    dense = _drop_sparse(matched, user, rmid, a_glob)
    if len(dense) >= 6:                      # (çok gürültülü veride her şeyi atmamak için)
        matched = dense
    for rd in range(LEGO_ROUNDS):
        say(50 + rd * 8, "Eksik parçalar yerine oturtuluyor…")
        added = 0
        mi = sorted(matched)
        bounds = [-1] + mi + [n]
        for gl, gr in zip(bounds, bounds[1:]):
            lo_i, hi_i = gl + 1, gr - 1
            if hi_i - lo_i + 1 < 5:
                continue
            jL = matched[gl][0] if gl >= 0 else -1
            jR = matched[gr][0] if gr < n else m
            p0 = lo_i
            first = PIECE if rd == 0 else 8 * rd         # her turda parça sınırları kayar
            while p0 <= hi_i:
                p1 = min(hi_i, p0 + (first if p0 == lo_i else PIECE) - 1)
                if p1 - p0 + 1 < 5:
                    p0 = p1 + 1
                    continue
                if should_stop and should_stop():
                    raise Stopped()
                piece = list(range(p0, p1 + 1))
                rr = list(range(jL + 1, jR))
                p0 = p1 + 1
                if len(rr) < 3:
                    continue
                try:
                    _, o, z = coarse_align([ref[j] for j in rr], [user[i] for i in piece],
                                           should_stop, scales=[a_glob])
                except SyncFailed:
                    continue
                b = o
                sel = []
                for it in range(2):
                    cands = _candidates(user, ref, piece, rr, a_glob, b, 2.5 if it == 0 else 1.5,
                                        rmid_sorted, rmid_order)
                    pairs = [(i, j, iou + 0.5 * scorer(i, j)) for i, j, iou in cands
                             if iou + 0.5 * scorer(i, j) >= 0.2]
                    sel = [pairs[k] for k in _best_chain(pairs, m)]
                    if len(sel) < 4:
                        break
                    xx = [(user[i]["start"] + user[i]["end"]) / 2 for i, _, _ in sel]
                    yy = [rmid[j] for _, j, _ in sel]
                    b = float(np.median(np.array(yy) - a_glob * np.array(xx)))
                strong = sum(1 for _, _, w in sel if w >= 0.4)
                lags = (ref[rr[-1]]["end"] - ref[rr[0]]["start"]) / DT
                if len(sel) < 4 or strong < max(5, _min_strong(len(piece), lags)):
                    continue                              # bu parça filmde yok / belirsiz
                # aynı kaymaya sahip komşu hipotezi yeniden kullan (parçalar tek parça olsun)
                hid = None
                for nb_i in (gl, gr):
                    if 0 <= nb_i < n and nb_i in matched:
                        ha, hb, _ = hyps[matched[nb_i][2]]
                        if abs(ha - a_glob) < 1e-9 and abs(hb - b) < 0.75:
                            hid = matched[nb_i][2]
                if hid is None:
                    hid = len(hyps)
                    hyps.append((a_glob, b, z))
                for i, j, w in sel:
                    matched[i] = (j, w, hid)
                jL = max(jL, max(j for _, j, _ in sel))
                added += len(sel)
        if added < 4:
            break

    say(75, "Zaman dönüşümü hesaplanıyor…")
    # ---- parçalar: ardışık aynı hipotezli eşleşmeler
    items = sorted(matched.items())                    # i'ye göre
    xs_all = {i: (user[i]["start"] + user[i]["end"]) / 2 for i, _ in items}

    def runs_of(its):
        runs = []
        for i, (j, w, hid) in its:
            if runs and runs[-1]["hid"] == hid:
                runs[-1]["items"].append((i, j))
            else:
                runs.append({"hid": hid, "items": [(i, j)]})
        return runs

    runs = runs_of(items)
    # Tesadüfi parçaları ele: gerçek parçada satırların çoğunun güçlü eşi vardır (yoğunluk
    # yüksek); başka kesimin hipoteziyle oluşan sahte parçalarda yoğunluk düşüktür.
    for _ in range(4):
        good = []
        for r in runs:
            idx = [i for i, _ in r["items"]]
            strong = sum(1 for i in idx if matched[i][1] >= 0.4)
            if len(idx) >= 8 and strong / (idx[-1] - idx[0] + 1) >= 0.45:
                good.append(r)
        if not good or len(good) == len(runs):
            break                              # hiçbiri geçmezse (zayıf eşleşme) olduğu gibi bırak
        keep_ids = {i for r in good for i, _ in r["items"]}
        runs = runs_of([(i, v) for i, v in items if i in keep_ids])
        items = [(i, v) for i, v in items if i in keep_ids]

    models = []
    for r in runs:
        x = [xs_all[i] for i, _ in r["items"]]
        y = [rmid[j] for _, j in r["items"]]
        a0, b0, _ = hyps[r["hid"]]
        a, b, keep = _robust_line(x, y, a0, b0, free_scale=True)
        models.append({"a": a, "b": b, "lo": min(x), "hi": max(x), "pairs": len(x),
                       "inl": int(keep.sum())})

    # ---- satırları dönüştür (eşleşmeyenler en yakın parçanın modelini kullanır)
    def fit_score(i, mdl):
        """Satırın bu modelle yerleştiği yerde referanstan ne kadar destek gördüğü."""
        c = user[i]
        ps, pe = mdl["a"] * c["start"] + mdl["b"], mdl["a"] * c["end"] + mdl["b"]
        pm = (ps + pe) / 2
        best = 0.0
        for k in range(bisect.bisect_left(rmid_sorted, pm - 2.0), bisect.bisect_right(rmid_sorted, pm + 2.0)):
            j = rmid_order[k]
            r = ref[j]
            union = max(pe, r["end"]) - min(ps, r["start"])
            iou = max(0.0, min(pe, r["end"]) - max(ps, r["start"])) / union if union > 0 else 0.0
            best = max(best, iou + 0.5 * scorer(i, j))
        return best

    cuts = []
    for k in range(len(models) - 1):
        A, B = models[k], models[k + 1]
        gap = [i for i in range(n) if A["hi"] < (user[i]["start"] + user[i]["end"]) / 2 < B["lo"]]
        if not gap:
            cuts.append((A["hi"] + B["lo"]) / 2)
            continue
        sa = [fit_score(i, A) for i in gap]
        sb = [fit_score(i, B) for i in gap]
        # i'ye kadar A, sonrası B: toplam destek en büyük olan sınır (eşitlikte ortaya yakın)
        best_c, best_v = -1, -1e9
        for c in range(-1, len(gap)):
            v = sum(sa[:c + 1]) + sum(sb[c + 1:]) - 1e-3 * abs(c + 1 - len(gap) / 2)
            if v > best_v:
                best_c, best_v = c, v
        mids = [(user[i]["start"] + user[i]["end"]) / 2 for i in gap]
        left = A["hi"] if best_c < 0 else mids[best_c]
        right = B["lo"] if best_c + 1 >= len(gap) else mids[best_c + 1]
        cuts.append((left + right) / 2)
    out = []
    for c in user:
        mid = (c["start"] + c["end"]) / 2
        mdl = models[bisect.bisect_right(cuts, mid)]
        out.append({"start": mdl["a"] * c["start"] + mdl["b"],
                    "end": mdl["a"] * c["end"] + mdl["b"], "text": c["text"]})
    final = []
    for c in out:
        if c["end"] <= 0:
            continue
        c["start"] = max(0.0, c["start"])
        c["end"] = max(c["end"], c["start"] + 0.05)
        final.append(c)
    final.sort(key=lambda c: c["start"])

    # ---- filmde olup kullanıcı altyazısında karşılığı olmayan sahneleri referanstan doldur
    # Yalnızca GÜVENİLİR iki ardışık eşleşme arasında kayma sıçradıysa: altyazı kesintisiz
    # devam ederken referansta ek bir sahne (>= FILL_MIN_JUMP sn) geçmiştir. Kullanıcının tek tek
    # atladığı satırlar (kayma değişmez) ve belirsiz bölgeler doldurulmaz.
    filled_cues, filled_blocks = 0, 0
    if fill_missing:
        fs = np.array([c["start"] for c in final]); fe = np.array([c["end"] for c in final])

        def covered(r):
            return bool(((fs < r["end"] - 0.25) & (fe > r["start"] + 0.25)).any())
        used_j = {j for _, (j, _, _) in items}
        extra = []
        for (i1, (j1, w1, _)), (i2, (j2, w2, _)) in zip(items, items[1:]):
            if w1 < 0.4 or w2 < 0.4:
                continue
            jump = (rmid[j2] - rmid[j1]) - a_glob * (xs_all[i2] - xs_all[i1])
            if jump < FILL_MIN_JUMP:
                continue
            blk = [j for j in range(j1 + 1, j2) if j not in used_j and not covered(ref[j])]
            if len(blk) >= FILL_MIN_CUES:
                extra.extend(blk)
                filled_blocks += 1
        for j in extra:
            r = ref[j]
            txt = "\n".join(f"<i>{ln}</i>" for ln in r["text"].split("\n") if ln.strip())
            if txt:
                final.append({"start": r["start"], "end": r["end"], "text": txt})
                filled_cues += 1
        final.sort(key=lambda c: c["start"])

    # ---- rapor
    errs = []
    for r, mdl in zip(runs, models):
        for i, j in r["items"]:
            errs.append(abs(rmid[j] - (mdl["a"] * xs_all[i] + mdl["b"])))
    errs = np.array(errs)
    inl = float((errs <= INLIER_TOL).mean()) if len(errs) else 0.0
    segs = []
    for k, mdl in enumerate(models):
        segs.append({"scale": mdl["a"], "shift": mdl["a"] * mdl["lo"] + mdl["b"] - mdl["lo"],
                     "from": mdl["lo"], "to": mdl["hi"], "pairs": mdl["pairs"],
                     "start": 0.0 if k == 0 else cuts[k - 1]})
    report = {
        "total": n, "matched": len(items), "ref_total": m,
        "coverage": len(items) / n, "inlier_ratio": inl,
        "median_err": float(np.median(errs)) if len(errs) else None,
        "z": first_z, "segments": segs, "text_mode": scorer.mode,
        "filled_cues": filled_cues, "filled_blocks": filled_blocks,
        "quality": "good" if (len(items) / n >= 0.5 and inl >= 0.85) else "weak",
    }
    say(95, "Bitti")
    return final, report


def describe(rep):
    """Raporu kullanıcıya gösterilecek kısa Türkçe metne çevirir."""
    # Aynı kaymaya sahip ardışık parçaları tek parça say
    segs = []
    for sg in rep["segments"]:
        if segs and abs(sg["shift"] - segs[-1]["shift"]) < 1.5 and abs(sg["scale"] - segs[-1]["scale"]) < 0.002:
            continue
        segs.append(sg)
    if len(segs) == 1:
        s = segs[0]
        part = f"{s['shift']:+.1f} sn kaydırıldı"
        if abs(s["scale"] - 1.0) > 0.0003:
            part += f", hız oranı ×{s['scale']:.4f} (kare hızı farkı) düzeltildi"
    else:
        part = "Kayma bölümlere göre değişiyor (ek sahne/kesinti): " + ", ".join(
            f"{sg['shift']:+.0f} sn" + ("" if k == 0 else f" ({sg.get('start', sg['from']) / 60:.0f}. dk'dan sonra)")
            for k, sg in enumerate(segs))
    txt = f"{part}. {rep['matched']}/{rep['total']} satır doğrudan doğrulandı, kalanı komşu satırların kaymasıyla yerleştirildi"
    if rep["median_err"] is not None:
        txt += f" (ortanca sapma {rep['median_err']:.2f} sn)"
    if rep.get("filled_cues"):
        txt += (f". Filmde olup altyazıda bulunmayan {rep['filled_blocks']} bölümün "
                f"{rep['filled_cues']} satırı referanstan (italik) eklendi")
    if rep["quality"] == "weak":
        txt += ". ⚠ Eşleşme zayıf: sonucu kontrol edin"
    return txt


def sync_files(user_path, ref_path, out_path, progress=None, should_stop=None, embedder=None,
               fill_missing=True):
    user = parse_srt(user_path)
    ref = parse_srt(ref_path)
    cues, rep = sync_cues(user, ref, progress, should_stop, embedder, fill_missing)
    tmp = out_path + ".part"
    write_srt(tmp, cues)
    os.replace(tmp, out_path)
    return rep


# ============================================================== isteğe bağlı: çok dilli gömme
def load_embedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """`pip install sentence-transformers` kuruluysa çok dilli cümle gömmesi döndürür, yoksa None.
    İlk kullanımda model (~470 MB) indirilir. Farklı dildeki altyazı eşleştirmesini güçlendirir;
    olmadan da sistem zaman desenleriyle çalışır."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except Exception:
        return None

    def embed(texts):
        return model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False)
    return embed


# ============================================================== Qt işçisi (Qt yoksa tanımlanmaz)
try:
    from PyQt5.QtCore import QThread, pyqtSignal

    class SyncWorker(QThread):
        """user_srt'yi ref_srt'ye göre hizalayıp out_srt olarak yazar."""
        progress = pyqtSignal(int, str)
        finished_ok = pyqtSignal(str, dict)      # çıktı yolu, rapor
        failed = pyqtSignal(str)
        cancelled = pyqtSignal()

        def __init__(self, user_srt, ref_srt, out_srt, use_embeddings=True, fill_missing=True):
            super().__init__()
            self.user_srt, self.ref_srt, self.out_srt = user_srt, ref_srt, out_srt
            self.use_embeddings = use_embeddings
            self.fill_missing = fill_missing
            self._stop = False

        def cancel(self):
            self._stop = True

        def run(self):
            try:
                emb = None
                if self.use_embeddings:
                    try:
                        import sentence_transformers  # noqa: F401
                        self.progress.emit(5, "Dil modeli yükleniyor…")
                        emb = load_embedder()
                    except ImportError:
                        emb = None
                rep = sync_files(self.user_srt, self.ref_srt, self.out_srt,
                                 progress=lambda p, t: self.progress.emit(p, t),
                                 should_stop=lambda: self._stop, embedder=emb,
                                 fill_missing=self.fill_missing)
                self.finished_ok.emit(self.out_srt, rep)
            except Stopped:
                self.cancelled.emit()
            except SyncFailed as e:
                self.failed.emit(str(e))
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"Beklenmeyen hata: {e}")
except Exception:  # pragma: no cover
    pass


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    r = sync_files(sys.argv[1], sys.argv[2], sys.argv[3], progress=lambda p, t: print(f"%{p} {t}"))
    print(describe(r))
