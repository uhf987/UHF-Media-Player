"""Senkron motoru testleri (sentetik veri).
Çalıştır (ana klasörden):  python test_subtitle_sync.py   ya da   python -m unittest -v
(dosya tests/ klasöründeyse: python -m unittest discover -s tests -v)"""
import os
import random
import sys
import tempfile
import unittest

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                      # test dosyası ana klasördeyse
sys.path.insert(0, os.path.join(_here, ".."))  # tests/ klasöründeyse
import subtitle_sync as S  # noqa: E402


def make_ref(seed=1, minutes=110):
    """Gerçekçi konuşma deseni: kısa/uzun sessizlikler, 1-5 sn'lik satırlar."""
    rnd = random.Random(seed)
    t, cues = 20.0, []
    while t < minutes * 60:
        d = rnd.uniform(1.0, 5.0)
        cues.append({"start": t, "end": t + d, "text": f"line {len(cues)} " + "x" * rnd.randint(3, 40)})
        t += d + (rnd.uniform(0.1, 2.0) if rnd.random() < 0.8 else rnd.uniform(4, 40))
    return cues


def make_user(ref, fwd, seed=2, drop=0.12, merge=0.1, jitter=0.25, same_text=True, human_bias=0.0):
    """ref zamanından kullanıcı altyazısı üretir. fwd(t_user)->t_ref'in TERSİ uygulanır:
    kullanıcı_zamanı = inv(ref_zamanı). Her kullanıcı satırı hangi ref satırından geldiğini tutar."""
    rnd = random.Random(seed)
    out, k = [], 0
    while k < len(ref):
        if rnd.random() < drop:
            k += 1
            continue
        c = ref[k]
        end_k = k
        if rnd.random() < merge and k + 1 < len(ref):
            end_k = k + 1
        s = c["start"] + human_bias + rnd.gauss(0, jitter)
        e = ref[end_k]["end"] + human_bias + rnd.gauss(0, jitter)
        e = max(e, s + 0.4)
        text = c["text"] if same_text else "".join(rnd.choice("abcdefgh ") for _ in range(20))
        out.append({"start": s, "end": e, "text": text, "_k0": k, "_k1": end_k, "_s": s, "_e": e})
        k = end_k + 1
    for c in out:
        c["start"], c["end"] = fwd(c["start"]), fwd(c["end"])
    return out


def truth_errors(result, user):
    """Çıktı satırının başlangıcı, gerçek (ref) zamandan ne kadar sapıyor."""
    by = {id(u): u for u in user}
    errs = []
    srt_sorted = sorted(user, key=lambda c: c["start"])
    # sync_cues kullanıcı satırlarını başlangıca göre sıralar; çıktı da sıralıdır ama
    # parça sınırında yer değişebilir: metne göre değil sırayla eşleşemeyiz -> _s ile ara
    res = sorted(result, key=lambda c: c["text"])
    exp = sorted(srt_sorted, key=lambda c: c["text"])
    for r, u in zip(res, exp):
        if r["text"] != u["text"]:
            return None
        errs.append(abs(r["start"] - u["_s"]))
    return np.array(errs)


class SyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = make_ref()

    def check(self, user, med=0.3, p95=1.0, report_fn=None):
        res, rep = S.sync_cues([{k: v for k, v in c.items()} for c in user], self.ref, fill_missing=False)
        errs = truth_errors(res, user)
        self.assertIsNotNone(errs, "metinler eşleşmedi (test altyapısı)")
        self.assertLess(float(np.median(errs)), med, rep)
        self.assertLess(float(np.percentile(errs, 95)), p95, rep)
        return rep

    def test_small_shift(self):
        # kullanıcı altyazısı 1.5 sn GEÇ
        user = make_user(self.ref, lambda t: t + 1.5)
        rep = self.check(user)
        self.assertEqual(len(rep["segments"]), 1)
        self.assertAlmostEqual(rep["segments"][0]["shift"], -1.5, delta=0.3)

    def test_large_negative_shift(self):
        self.check(make_user(self.ref, lambda t: t - 37.2))

    def test_huge_positive_shift(self):
        self.check(make_user(self.ref, lambda t: t + 412.0))

    def test_fps_mismatch(self):
        # altyazı 25 fps için, film 23.976 fps: kullanıcı = ref / 1.0427
        r = 25 / 23.976
        rep = self.check(make_user(self.ref, lambda t: t / r + 3.0))
        self.assertAlmostEqual(rep["segments"][0]["scale"], r, delta=0.0006)

    def test_other_fps_direction(self):
        r = 23.976 / 25
        self.check(make_user(self.ref, lambda t: t / r - 5.0))

    def test_cut_in_middle(self):
        # filmde 4.5 dk'lık sahne var/yok farkı: ikinci yarı 270 sn farklı
        mid = 55 * 60
        user = make_user(self.ref, lambda t: t + 2.0 if t < mid else t + 2.0 + 270.0)
        rep = self.check(user, med=0.4, p95=1.5)
        self.assertGreaterEqual(len(rep["segments"]), 2)

    def test_cross_language_no_text_overlap(self):
        # metinler tamamen farklı (farklı dil gibi): yalnızca zaman deseni
        user = make_user(self.ref, lambda t: t + 6.0, same_text=False)
        # metin farklı olduğundan _s'e göre doğrulama için metni benzersizleştir
        for i, c in enumerate(user):
            c["text"] = f"u{i:05d}"
        res, rep = S.sync_cues([dict(c) for c in user], self.ref, fill_missing=False)
        errs = truth_errors(res, user)
        self.assertLess(float(np.median(errs)), 0.35, rep)

    def test_unrelated_subtitle_fails_or_weak(self):
        other = make_ref(seed=99, minutes=100)
        try:
            _, rep = S.sync_cues(other, self.ref)
        except S.SyncFailed:
            return
        self.assertEqual(rep["quality"], "weak", rep)

    def test_srt_roundtrip_and_cp1254(self):
        cues = [{"start": 1.5, "end": 3.0, "text": "Merhaba şöyle ğüişçö"},
                {"start": 4.0, "end": 5.25, "text": "İkinci\nsatır"}]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.srt")
            S.write_srt(p, cues)
            back = S.parse_srt(p)
            self.assertEqual([c["text"] for c in back], [c["text"] for c in cues])
            q = os.path.join(d, "b.srt")        # Windows-1254 kodlu eski altyazı
            with open(q, "wb") as f:
                f.write("1\n00:00:01,500 --> 00:00:03,000\nşğüıöçİ\n\n".encode("cp1254"))
            self.assertEqual(S.parse_srt(q)[0]["text"], "şğüıöçİ")

    def test_missing_blank_line_between_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.srt")
            open(p, "w").write("1\n00:00:01,000 --> 00:00:02,000\nA\n2\n00:00:03,000 --> 00:00:04,000\nB\n")
            self.assertEqual([c["text"] for c in S.parse_srt(p)], ["A", "B"])

    # ---- filmde olup kullanıcı altyazısında olmayan ek sahneler (yönetmen kurgusu vb.)
    def _extra_scene_case(self, blocks, off=3.0, seed=2, fill=True):
        kept = [c for c in self.ref if not any(a <= c["start"] < b for a, b in blocks)]
        missing = [c for c in self.ref if any(a <= c["start"] < b for a, b in blocks)]

        def fwd(t):                         # ref zamanı -> kullanıcı zamanı (ek sahneler çıkarılmış)
            return t - sum(b - a for a, b in blocks if b <= t) + off
        user = make_user(kept, fwd, seed=seed)
        for i, c in enumerate(user):
            c["text"] = f"u{i:05d}"
        res, rep = S.sync_cues([dict(c) for c in user], self.ref, fill_missing=fill)
        filled = [c for c in res if c["text"].startswith("<i>")]
        real = [c for c in res if not c["text"].startswith("<i>")]
        errs = truth_errors(real, user)
        return user, missing, filled, errs, rep

    def test_extra_scenes_aligned_and_filled(self):
        blocks = [(1000, 1200), (2500, 2560), (4000, 4400), (5000, 5030)]
        user, missing, filled, errs, rep = self._extra_scene_case(blocks)
        self.assertIsNotNone(errs)
        self.assertLess(float(np.median(errs)), 0.3, rep)
        self.assertLess(float(np.percentile(errs, 95)), 1.0, rep)
        # eksik sahnelerin satırlarının çoğu referanstan eklenmiş olmalı
        got = sum(1 for m_ in missing if any(abs(f["start"] - m_["start"]) < 0.5 for f in filled))
        self.assertGreaterEqual(got, 0.8 * len(missing), (got, len(missing), rep))
        # ... ve sahne dışında (normal bölgelerde) gereksiz ekleme olmamalı
        stray = [f for f in filled
                 if not any(a - 12 <= f["start"] <= b + 12 for a, b in blocks)]
        self.assertLessEqual(len(stray), 3, len(stray))

    def test_fill_can_be_disabled(self):
        _, _, filled, errs, rep = self._extra_scene_case([(1000, 1200)], fill=False)
        self.assertEqual(filled, [])
        self.assertLess(float(np.median(errs)), 0.3)

    def test_many_short_extra_scenes(self):
        blocks = [(600 + 500 * k, 640 + 500 * k) for k in range(10)]   # 10 adet 40 sn'lik sahne
        user, missing, filled, errs, rep = self._extra_scene_case(blocks, off=-12.0, seed=4)
        self.assertIsNotNone(errs)
        self.assertLess(float(np.median(errs)), 0.35, rep)
        self.assertLess(float(np.percentile(errs, 95)), 1.5, rep)

    def test_embedder_hook_used_when_languages_differ(self):
        # sahte "çeviri": aynı ref satırından gelen kullanıcı satırı ile ref satırı aynı vektör
        def code(k, base):                 # rakamsız; iki dilin harfleri ayrık (ortak 3'lü yok)
            return "".join(chr(base + int(d)) for d in str(k))
        user = make_user(self.ref, lambda t: t + 2.0, same_text=False)
        for i, c in enumerate(user):
            c["text"] = code(c["_k0"], 97)
        ref = [dict(c, text=code(i, 107)) for i, c in enumerate(self.ref)]
        rng = np.random.RandomState(0)
        vecs = {i: rng.randn(32) for i in range(len(ref))}
        for v in vecs:
            vecs[v] /= np.linalg.norm(vecs[v])

        def emb(texts):
            return np.array([vecs[int("".join(str((ord(ch) - 97) % 10) for ch in t))] for t in texts])
        res, rep = S.sync_cues([dict(c) for c in user], ref, embedder=emb)
        self.assertEqual(rep["text_mode"], "embed")


if __name__ == "__main__":
    unittest.main()
