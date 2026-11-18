# UHF Video Player

PyQt5 + libmpv tabanlı, hafif bir video oynatıcı. Filmin sesinden **yapay zekâ ile altyazı üretir** ve bunu **bağlamı dikkate alarak Türkçeye çevirir**. Her şey yerel çalışır: ses, altyazı ve çeviri internete gönderilmez.

> Altyazı özellikleri üç dosyadan oluşur: `video_studio_player.py` (oynatıcı ve arayüz), `live_subtitle.py` (Whisper, çeviri, model yönetimi) ve `subtitle_sync.py` (altyazı senkronlama).

## Özellikler

- **🎙 Altyazı** — Filmin sesini Whisper ile metne çevirir ve videonun yanına `film.ai.<dil>.srt` olarak kaydeder. Dosya zaten varsa yeniden üretmez, doğrudan yükler. İlerleme gösterilir, ikinci basışta işlem iptal edilir.
- **🌐 Translate** — Üretilen altyazıyı TranslateGemma (Ollama) ile istediğin dile çevirir. Kaynak dil, Whisper'ın algıladığı dildir (kendiliğinden bulunur). Basınca hedef dil sorulur: üstte favori diller (varsayılan Türkçe ve İngilizce), altta listedeki tüm diller. Tüm altyazı baştan sona çevrilir, bitince `film.ai.<dil>.srt` olarak kaydedilir ve otomatik yüklenir. O dile çevrilmiş dosya zaten varsa yeniden çevrilmez, doğrudan yüklenir. İkinci basışta işlem iptal edilir.
- **🔄 Senkronla** — Sürükleyip bıraktığın (ya da düğmeye basınca seçtiğin) altyazı filmle uyumsuzsa onu filme göre hizalar. Filmin sesinden çıkarılan Whisper altyazısı *referans* alınır; senin altyazının **metni korunur**, yalnızca zamanları düzeltilir. Sabit kaymayı (1–2 sn ya da dakikalarca), kare hızı farkını (25↔23.976 vb.) ve filmdeki kesinti/ek sahne farklarını (parça parça kayma) çözer. Referans (`film.ai.<dil>.srt`) yoksa önce 🎙 ile üretilir, bitince senkron otomatik başlar. Sonuç `<altyazı>.synced.srt` olarak kaydedilir ve yüklenir; orijinal dosya değişmez. Referans ve altyazı farklı dillerde olabilir. Eşleşme zayıfsa uyarı verir. Çalışırken tekrar basmak iptal eder.
  - **Ek sahneler:** Filmde olup altyazında bulunmayan sahneler (yönetmen kurgusu, çevirmenin atladığı bölümler) kayma sıçramasından tespit edilir; referans (ör. İngilizce) satırları **italik** olarak o boşluğa eklenir. Yalnızca güvenilir iki eşleşme arasında zaman kayması ≥ 8 sn sıçradıysa eklenir; kullanıcının tek tek atladığı satırlar doldurulmaz. Kapatmak için `live_subtitle.json` içinde `"sync_fill_missing": false` yazın.
  - Yöntem: konuşma var/yok desenlerinin FFT ile çapraz ilintisi (kaba hizalama) → zaman örtüşmesi ve metin benzerliğiyle satır eşleştirme → sağlam doğrusal/parçalı uyum; eşleşmeyen boşluklar ise komşu eşleşmelerin arasında parça parça (lego gibi) yeniden aranır. Mantık `subtitle_sync.py` içindedir (Qt'ye bağlı değildir, yalnızca `numpy`).
  - İsteğe bağlı: `pip install sentence-transformers` kuruluysa ve altyazı referanstan farklı dildeyse çok dilli cümle gömmeleri eşleştirmeyi güçlendirir (ilk kullanımda ~470 MB model indirilir). Kurulu değilse sistem yalnızca zaman desenleriyle çalışır.
- **📦 Model Ayarları** — Whisper modellerini indirme / silme / seçme penceresi. Yarım kalan indirme kaldığı yerden devam eder. TranslateGemma 4B / 12B / 27B seçimi ve kurulu olup olmadığının görünümü de burada.
- Favori diller `%APPDATA%\UHF\live_subtitle.json` içindeki `favorite_langs` listesinden değiştirilebilir (ör. `["tr", "en", "de"]`).
- Çeviri bloklar halinde, öncesi ve sonrası satırları bağlam vererek yapılır. Tekrar döngüsü (aynı yeri sürekli üretme) algılanır ve kurtarılır; cümleler birleştirilip yeniden bölünebildiği için Türkçe söz dizimi İngilizce satır bölünmesine bağlı kalmaz.
- Oynatıcı: libmpv ile oynatma, altyazı konum/boyut ayarı, şarkı tanıma (shazamio), ffmpeg ile kırpma ve dışa aktarma.

## Gereksinimler

| Bileşen | Ne için |
|---|---|
| Python 3.10+ | Kaynak koddan çalıştırma |
| `PyQt5`, `mpv` (python-mpv), `requests` | Arayüz, oynatma, Ollama/indirme |
| libmpv (Windows: `libmpv-2.dll`) | Oynatma |
| `ffmpeg` / `ffprobe` | Sesi çıkarma, dışa aktarma |
| Sesten metne motoru (aşağıya bak) | 🎙 |
| [Ollama](https://ollama.com) + TranslateGemma | 🌐 |

```bash
pip install PyQt5 mpv requests numpy
```

### Sesten metne motoru (ikisinden biri yeter)

1. **faster-whisper** (Python içi, önerilen)
   ```bash
   pip install faster-whisper
   ```
2. **Subtitle Edit'in `faster-whisper-xxl.exe` programı.** Python'da faster-whisper yüklenemezse program, Subtitle Edit'in indirdiği Purfview Faster-Whisper-XXL'i otomatik bulup kullanır. Şu klasörlere bakar:
   `%APPDATA%\Subtitle Edit\SpeechToText\Purfview-Faster-Whisper-XXL\` ve eski sürümlerdeki `%APPDATA%\Subtitle Edit\Whisper\Purfview-Faster-Whisper-XXL\`.
   Subtitle Edit uygulamasını açman gerekmez; yalnızca motorun orada kurulu olması yeterli.

GPU için NVIDIA ve CUDA 12 DLL'leri (cuBLAS/cuDNN) gerekir. GPU yoksa ya da kullanılamazsa işlemciye otomatik düşülür (daha yavaştır).

### Çeviri (Ollama)

Ollama'yı kurup açık tutun, ardından modeli bir kez indirin:

```bash
ollama pull translategemma:4b      # varsayılan
ollama pull translategemma:12b     # daha kaliteli
ollama pull translategemma:27b     # en kaliteli, çok VRAM ister
```

Ollama kapalıysa program "Ollama'yı açın" uyarısı verir. Model kurulu değilse yalnızca yukarıdaki komutu gösterir, kendisi indirmez.

## Çalıştırma

```bash
python video_studio_player.py
```

1. Filmi açın.
2. **🎙 Altyazı**'ya basın. İlk kullanımda 📦'dan bir Whisper modeli indirmeniz istenir (`large-v3` yüksek doğruluk, `small` işlemci için dengeli).
3. Altyazı hazır olunca **🌐 Translate**'e basın, hedef dili seçin (ör. Türkçe). Kaynak dilin kendisi hedef olarak seçilemez.

### Dosya adlandırma

| Dosya | Anlamı |
|---|---|
| `film.ai.en.srt` | Whisper çıktısı (dil kodu algılanan dildir) |
| `film.ai.tr.srt`, `film.ai.de.srt`, … | TranslateGemma çıktısı (hedef dile göre) |
| `*.part` | Üretim sürerken geçici dosya, bitince yeniden adlandırılır |

## Veri konumları (Windows)

| Ne | Nerede |
|---|---|
| Whisper modelleri | `%LOCALAPPDATA%\UHF\whisper\faster-whisper-<ad>` |
| Ayarlar (seçili modeller, eklenen klasörler) | `%APPDATA%\UHF\live_subtitle.json` |

Programı başka bilgisayara taşımak bu dosyaları taşımaz; her kullanıcı kendi modellerini 📦'tan indirir. Modelleri paketlemek istemezseniz bu klasörler dağıtıma girmez.

## Exe oluşturma (PyInstaller)

`video_studio_player.py` ve `live_subtitle.py` aynı klasörde olmalıdır.

```bash
pyinstaller --onedir --windowed --noupx --name UHF --icon=icon.ico --add-data "icon.ico;." video_studio_player.py
```

Derlemeden sonra `libmpv-2.dll`, `ffmpeg.exe` ve `ffprobe.exe`'yi `dist\UHF\` içine kopyalayın. `--onedir` hızlı açılır (`_internal` klasörüyle birlikte dağıtın).

> Python ortamınızda `torch` gibi ağır paketler kuruluysa derleme uzar ve boyut büyür. Bunun için bu proje özelinde temiz bir sanal ortam kullanın ya da `--exclude-module torch --exclude-module torchvision --exclude-module tensorflow` ekleyin.

Sesten metne motorunu exe'nin içine gömmek isterseniz temiz bir ortamda `faster-whisper` kurup `--collect-data faster_whisper --collect-binaries ctranslate2` ekleyin. `--collect-data` olmadan ses etkinliği tespiti (`silero_vad_v6.onnx`) pakete girmez ve tanıma çöker.

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| "Ses tanıma bileşeni yüklenemedi" | `pip install faster-whisper` veya Subtitle Edit'ten Faster-Whisper-XXL motorunu indirin. Pencerede asıl hata metni de gösterilir. |
| "Whisper modeli bulunamadı" | 📦'ı açıp bir model indirin ve seçin. |
| "Ollama'yı açın" | Ollama uygulamasını başlatın, 📦'ta ↻ Yenile'ye basın. |
| "Model kurulu değil" | Uyarıda yazan `ollama pull ...` komutunu çalıştırın. |
| GPU kullanılamadı mesajı | İşlemciyle devam eder; GPU için CUDA DLL'lerini `exe` yanındaki `cuda` klasörüne koyun. |
| Çeviri satırları eksik/tekrarlı | Program otomatik olarak daha küçük parçalarla yeniden dener; olmazsa satır satır çevirir. |

## Bilinen sınırlar

- Hedef dil listesi yaygın diller içindir; TranslateGemma'nın bazı dillerdeki kalitesi diğerlerinden düşük olabilir.
- Whisper çıktısı ses kalitesine, konuşmaya ve modele göre hata içerebilir. Çeviri, Whisper metnini baz alır.
- whisper.cpp (`ggml`) modelleri bu programla **uyumlu değildir**; yalnızca CTranslate2 (faster-whisper) modelleri kullanılır.
- `faster-distil-whisper` gibi distil modeller çoğunlukla yalnızca İngilizce içindir.
- Çok büyük modeller (`large-v3`, TranslateGemma 27B) ciddi bellek/VRAM ister.

## Ek: bağımsız altyazı çevirici

Oynatıcıdan bağımsız kullanılabilen araçlar da vardır:

- `srt_ceviri.py` — komut satırından tüm SRT'yi bağlamlı çevirir (bloklar, yeniden deneme, kaldığı yerden devam).
- `srt_ceviri_ui.py` — yerel web arayüzü: SRT yükleme, canlı çeviri takibi, düzenleme ve indirme.

```bash
python srt_ceviri.py film.srt --src en --dst tr --model translategemma:12b
```

## Lisans

Lisansı eklemeyi unutmayın (`LICENSE` dosyası). Üçüncü taraf bileşenlerin lisansları kendilerine aittir: libmpv, ffmpeg, faster-whisper / CTranslate2, Whisper modelleri, Purfview Faster-Whisper-XXL ve TranslateGemma.
