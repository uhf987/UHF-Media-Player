# UHF

PyQt5 ve **libmpv** tabanlı, hafif bir video oynatıcı ve basit video düzenleyici (döndür / kırp / kes / dışa aktar). Oynatma tamamen mpv motoruyla yapılır, dışa aktarma ise ffmpeg ile.

Açılışta, eski televizyonların UHF kanal arama ekranını andıran piksel yazılı bir "sinyal yok" logosu görünür.

## Özellikler

### Oynatıcı
- mpv motoruyla oynatma (mp4, mov, mkv, avi, m4v, webm)
- Çoklu **ses kanalı** ve **altyazı** seçimi, harici `.srt` dosyasını sürükleyip bırakarak ekleme
- Altyazı boyutu ve konumunu **canlı** ayarlama
- Tıklanan yere doğrudan atlayan ve sürüklerken videoyu canlı gösteren ilerleme çubuğu
- **L / R mono**: bir kanalda parazit ya da sessizlik varsa sağlam kanalı iki hoparlörden dinleme
- **4:3 / 16:9** düğmeleri: görüntüyü bu orana getirir, ikisi de işaretsizse video orijinal oranına döner (yalnızca oynatma görüntüsünü etkiler)
- Ekran görüntüsü alma (masaüstüne kaydedilir, altyazılar dahil)
- Tam ekran: fare alta gelince beliren kontrol çubuğu, hareketsiz imleci otomatik gizleme
- Çerçevesiz özel pencere, her zaman üstte tutma (📌)
- Tek pencere: bir video dosyasına çift tıklayınca yeni pencere açmak yerine açık olan pencerede yüklenir
- Taramalı (interlaced) kaynaklarda oynatma sırasında otomatik deinterlace

### Düzenleme ve dışa aktarma
- **Döndürme**: 90° saat yönü, 90° saat yönü tersi, 180° (oynatmada mpv'nin gerçek `video-rotate` özelliğiyle)
- **Kırpma**: videonun üzerinde sürüklenen kırpma kutusu, çıktı boyutu ve en-boy oranı canlı gösterilir
- **Kesme (trim)**: zaman çizgisinde başlangıç/bitiş tutamaçları, ±1 saniyelik hassas ayar düğmeleri
- Döndürme, kırpma ve kesme **tek bir ffmpeg çağrısında** uygulanır
- Kaynağın bitrate'i korunur, bulunamazsa yüksek kaliteli CRF moduna geçilir
- Uygun GPU kodlayıcı (NVENC, QSV, AMF) varsa otomatik kullanılır, başarısız olursa sessizce CPU'ya (libx264) dönülür
- Oynatıcıda seçili ses izi dışa aktarılır, altyazılar korunur
- Taramalı kaynaklarda: döndürme yoksa tarama tipi korunur, döndürme varsa bob deinterlace uygulanır

### Şarkı tanıma
- 🎵 düğmesi, oynatma konumundan yaklaşık 10 saniyelik ses alıp **Shazam** ile tanır
- ✓ işaretliyse dosya otomatik olarak `Sanatçı - Şarkı` adıyla yeniden adlandırılır
- Tanınamazsa ya da hata olursa dosyaya hiç dokunulmaz

## Kısayollar

| Girdi | İşlev |
|---|---|
| `Boşluk` | Oynat / duraklat |
| `→` / `←` | 3 saniye ileri / geri |
| `Esc` | Tam ekrandan çık |
| Videoya çift tıklama | Tam ekran aç / kapat |
| Fare tekerleği (video üstünde) | Ses seviyesi |
| Üst çubuğa çift tıklama | Büyüt / eski boyut |
| `.srt` dosyasını pencereye bırakma | Altyazı ekle |

## Gereksinimler

- Python 3.8+
- [PyQt5](https://pypi.org/project/PyQt5/)
- [python-mpv](https://pypi.org/project/python-mpv/) (`import mpv`) ve sistemde **libmpv**
  - Windows: `libmpv-2.dll` uygulamayla aynı klasörde ya da PATH'te olmalı
- **ffmpeg** ve **ffprobe** (dışa aktarma için, PATH'te ya da uygulamayla aynı klasörde). Yoksa oynatıcı yine çalışır, yalnızca "Dışa Aktar" kullanılamaz.
- İsteğe bağlı, şarkı tanıma için: `numpy` ve `shazamio`

```bash
pip install PyQt5 python-mpv
pip install numpy shazamio   # isteğe bağlı: şarkı tanıma
```

## Çalıştırma

```bash
python video_studio_player.py
python video_studio_player.py "C:\Videolar\ornek.mp4"
```

## Windows için derleme (PyInstaller)

Tek exe (`--onefile`) her açılışta dosyalarını geçici klasöre çıkardığı için yavaş açılır. Hızlı açılış için **onedir** önerilir:

```bash
pyinstaller --onedir --windowed --noupx --name UHF --icon=icon.ico --add-data "icon.ico;." video_studio_player.py
```

Ardından `libmpv-2.dll`, `ffmpeg.exe` ve `ffprobe.exe` dosyalarını `dist/UHF/_internal/` klasörüne (ya da `UHF.exe`'nin yanına) koyun.

İpuçları:
- `--noupx` kullanın, sıkıştırılmış dosyaların her açılışta açılması yavaşlatır.
- Antivirüs ilk çalıştırmada taratıp açılışı geciktirebilir, `dist/UHF` klasörünü istisnalara eklemek yardımcı olur.

### Video dosyalarını UHF ile ilişkilendirme

Bir video dosyasına çift tıklandığında Windows `UHF.exe "C:\...\video.mp4"` şeklinde çalıştırır. Uygulama bu yolu okuyup videoyu otomatik açar. UHF zaten açıksa dosya mevcut pencerede yüklenir.

## Notlar

- 4:3 / 16:9 düğmeleri yalnızca oynatma görüntüsünü değiştirir, dışa aktarılan video orijinal oranında kalır.
- Dışa aktarılan dosya, kaynağın yanına `_cropped`, `_rotated`, `_kesildi` ekleriyle kaydedilir.
- 10-bit ve üzeri kaynaklarda ve taramalı kaynakları korurken GPU kodlayıcı denenmez, doğrudan CPU kullanılır.
