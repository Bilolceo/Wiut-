# WIUT Hackathon 2026 — CV Track: Arxitektura va Reja

Sep 26, 2026 · @Bilol Pardabaev

## 0. Joriy holat (2026-09-26, yakuniy harness yugurishi)

**Tayyor va real videoda tekshirilgan:**
- Part A end-to-end, 14 qoida (85 unit test). Chiqariladigan 10 klass: red_light, stop_line, jaywalking, failure_to_yield, stopped_vehicle, congestion, wrong_way + VLM tasdiqlagandagina accident, near_miss, road_obstacle.
- O'chiq: illegal_turn (lanes[].turn_allowed noma'lum), illegal_u_turn (zona qonuniyligi tasdiqlanmagan), solid_line_crossing, fire_smoke.
- VLM verifier (Qwen3-VL-2B, P(yes) logitlardan, ≤30 chaqiruv/video): sample'lardagi 98 ta yolg'on kandidatning 98 tasini rad etdi; "avtobus bormi" 0.995 / "fil bormi" 0.002. Recall tekshirilmagan (sample'larda avariya yo'q).
- Metrik kalibrlash (`scripts/calibrate_camera.py`): piyoda bo'ylari (≈4.4k) + polosa VP + 14 m stop chizig'i; ikki mustaqil masshtab 3 % ichida mos. Butun yo'lda amal qiladi (eski 4-nuqtali homografiya uzoq hududda 40–50 % kam ko'rsatardi).
- wrong_way: 325 → 0 yolg'on kandidat (harakatsiz mashinalarning "yo'nalishi" shovqin edi). congestion: svetoforga bog'langan, avtobus bekati zonasi chiqarilgan.
- Part B: TTC/DRAC → sigmoid (ttc_half) → 0.5 s barqarorlik → EMA; butun yo'l (≤ 60 m). Sample'larda risk o'rtacha ≈ 0.06–0.08, 0.5 dan yuqori 0.1–1.6 % kadr.
- Harness (Apple M4, MPS): 57 event, VALID; jami 1.59–1.70× (Part A ≈ 1.1× shundan VLM ≈ 0.3×, Part B ≈ 0.5×). Limit 3×, maqsad 1.5× dan biroz yuqori.
- Sayt: `site/` (statik, 7 bo'lim), GitHub Pages workflow.

**Ochiq xavflar:**
- **T4'da tekshirilmagan**: torch+CUDA fp16, Qwen3-VL fp16 (T4'da bf16 yo'q) va haqiqiy runtime. Birinchi navbatdagi ish.
- Ground truth yo'q — chegaralar ko'z bilan sozlangan; `labels/dev_gt.json` kerak.
- VLM recall noma'lum; fire_smoke monitori real videoda sinalmagan.
- Tracker 10 fps'da ID almashtiradi; lanes[].turn_allowed tasdiqlanmagan.

## 1. Maqsad va ball matematikasi

Maqsad: top-20 ga kirish. Yakuniy ballning 42% i Part A'dan, 25% i saytdan keladi. Lekin eng birinchi shart — paket toza mashinada ishlashi, aks holda 60% (model balli) to'liq nolga tushadi.

\text{Elim} = 0.6\,(0.7A + 0.3B) + 0.25\,W + 0.15\,C

| Komponent | Umumiy balldagi ulushi | Eng og'ir ichki mezon | Ustuvorlik |
|---|---|---|---|
| Part A (event detection) | 42% | 14 klass bo'yicha macro F1, IoU 0.3/0.5/0.7 | 1 |
| Veb-sayt | 25% | Live demo 30% (umumiydan 7.5%) | 2 |
| Part B (anticipation) | 18% | 0.4 AP + 0.4 alarm F1 + 0.2 mTTA | 3 |
| Kod | 15% | "Runs as submitted" 40% (umumiydan 6%) | 1 (gate) |

Metrikadan kelib chiqadigan qarorlar:

- Har bir klass teng og'irlikda. fire_smoke yoki road_obstacle'ni bitta to'g'ri topish congestion'ni o'nta to'g'ri topish bilan teng. Kam uchraydigan klasslarga alohida detektor kerak.
- Testda yo'q klassni bashorat qilish jazolanadi: u klass 0 bilan o'rtachaga qo'shiladi. Har bir klass faqat kalibrlangan ishonch chegarasidan o'tsa chiqariladi.
- IoU 0.7 og'ir. 5 soniyalik hodisada chegaralar ±0.7 s aniqlikda bo'lishi kerak. Chegara aniqlashtirish (boundary refinement) — alohida modul.
- Part B'da kam, lekin erta alarm yutadi. Doimiy yuqori score 0 oladi. near_miss atrofidagi kadrlar e'tiborga olinmaydi, shuning uchun near-miss'da alarm chalinishi jazolanmaydi.
- Kod ishlamasa hammasi nol. Birinchi ish kuni oxirida bo'sh, lekin toza ishlaydigan paket CI'da sinalgan bo'lishi kerak.

## 2. Cheklovlar, taxminlar va texnologik tanlovlar

Har bir tanlov T4 16 GB, internetsiz, 5 GB og'irlik va "3 × video davomiyligi" cheklovlariga sig'adigan qilib tanlandi.

Qat'iy cheklovlar: 1 × T4 (16 GB, fp16, bf16 va FlashAttention-2 yo'q), 8 CPU, 32 GB RAM, Python ≥ 3.10, internet yo'q, og'irliklar ≤ 5 GB, har video ≤ 3 × davomiylik, faqat open-weights, seed'lar qat'iy.

| Qatlam | Asosiy tanlov | Zaxira | Litsenziya | Nega |
|---|---|---|---|---|
| Video o'qish | PyAV / OpenCV (CPU decode, thread) | decord | BSD / Apache | Barqaror, pip bilan o'rnatiladi |
| Detektor | YOLO11m yoki YOLO26m, TensorRT/ONNX fp16 | RF-DETR-base | AGPL-3.0 / Apache 2.0 | T4'da tez; repo public bo'lgani uchun AGPL muammo emas |
| Tracker | BoT-SORT (ReID o'chiq, GMC o'chiq) | ByteTrack | MIT | Kamera qo'zg'almas, harakat kompensatsiyasi kerak emas |
| Olov/tutun/to'siq | YOLOE yoki YOLO-World (open-vocab) + public fire/smoke datasetida fine-tune | Qwen3-VL tekshiruvi | AGPL / GPL | Kam uchraydigan klasslar uchun alohida signal |
| Fon modeli | OpenCV MOG2 (oldinga + orqaga) | median background | Apache | AI City g'oliblari usuli: to'xtagan obyektlar |
| VLM verifier | Qwen3-VL-2B-Instruct, fp16, SDPA attention | VLM'siz (faqat qoidalar) | Apache 2.0 | ~4.3 GB og'irlik 5 GB limitga sig'adi, T4 fp16'ni native qo'llaydi, qo'shimcha kvantlash kutubxonasi kerak emas |
| Risk kalibrator | LightGBM / logistic regression | qo'lda sigmoid | MIT | Kichik, deterministik |
| Sayt | Next.js (Vercel) + FastAPI/Gradio demo (HF Spaces) | Streamlit | — | Tez, bepul hosting |

Taxminlar (sample videolarni ko'rgach tasdiqlanadi):

- Kadrda svetofor ko'rinadi. Ko'rinmasa, red_light va stop_line qizil fazani trafik oqimidan aniqlaydi (chorrahaga kiruvchi oqim to'xtagan davr = qizil).
- Kadrda piyoda o'tish joyi va stop chizig'i bor.
- Video 29.97 fps (EDA'da tasdiqlangan), 2–6 daqiqalik, 4K, kunduzgi va kechki yorug'lik aralash bo'lishi mumkin.
- Test sample'lar bilan bir xil kamera va burchakda — shuning uchun qo'lda chizilgan sahna geometriyasi testda ham to'g'ri ishlaydi.

**EDA topilmasi (2026-09-26, sample videolar):** 4 ta video (C3896 340.3 s, C3897 317.8 s, C3902 317.8 s, C3905 127.6 s), barchasi 3840×2160 @ 29.97 fps, H.264 (Sony XAVC) (dastlabki "25 fps" taxmini shu asosda tuzatildi). Yorug'lik (o'rtacha kulrang daraja 0–255): C3896/C3897 kunduzi (~94); C3902 kechga tomon asta pasayadi (69→58, o'rtacha 64); C3905 shom, faralar yonik (~43, 60-sekundda 38→52 sakrash — MOG2 va HSV chegaralari moslashuvchan bo'lishi, sakrashda fon modeli reset qilinishi kerak). Tunda olingan video yo'q.

## 3. Umumiy arxitektura

Video bir marta o'qiladi, barcha og'ir hisob "track store"ga yoziladi. Keyin qoidalar, VLM va post-processing shu keshdan ishlaydi. Part B alohida, faqat o'tgan kadrlarni ko'radigan yengil oqim.

flowchart LR  
  V[video.mp4] --> D[Decoder\n≈15 fps]  
  S[(scene.json\npolosa, stop chiziq,\no'tish joyi, svetofor ROI)] --> E  
  D --> P[Perception\nYOLO + BoT-SORT\nsvetofor HSV\nMOG2 fon\nolov/tutun]  
  P --> T[(Track store\ntreklar, tezlik, metr)]  
  T --> E[Event engine\n14 ta qoida moduli]  
  E --> C[Kandidatlar\n+ ishonch]  
  C --> VLM[VLM verifier\nQwen3-VL-2B\nfaqat shubhalilar]  
  VLM --> PP[Post-processing\nmerge, refine,\nper-class threshold]  
  PP --> OUT[events list]  
  F[step frame] --> R[RiskEstimator\nyengil detektor + tracker\nTTC / PET / DRAC\nkalibrator]  
  S --> R  
  R --> RISK[risk 0..1]

Ikki qism orasidagi chegara qat'iy:

- Part A (detect_events) videoni istalgancha o'qiy oladi: bir necha o'tish, orqaga qarab aniqlashtirish, VLM.
- Part B (RiskEstimator.step) faqat kelgan kadrni oladi va o'z holatini saqlaydi. Part A natijasidan foydalanmaydi (qoida buzilishi bo'ladi). Kod bazasi umumiy (bir xil detektor, tracker, sahna geometriyasi), lekin obyektlar alohida.
- Runtime: harness avval detect_events, keyin har kadr uchun step chaqiradi. Ikkalasi birga ≤ 3 × davomiylik. Rejadagi maqsad ≤ 1.5 × (ikki barobar zaxira).

Kod qatlamlari (src/):

| Paket | Vazifa |
|---|---|
| io/ | Video o'qish, kadr sampling, keshlash |
| scene/ | scene.json yuklash, geometriya, homografiya, polosa yo'nalishi |
| perception/ | Detektor, tracker, svetofor holati, fon modeli, olov/tutun |
| tracks/ | Track store, silliqlash, tezlik va tezlanish (m/s) |
| events/ | Har klass uchun bitta modul, umumiy EventRule interfeysi |
| verify/ | VLM verifier va prompt shablonlari |
| post/ | Segmentlarni birlashtirish, chegara aniqlashtirish, threshold'lar |
| risk/ | Part B: causal tracker, TTC/PET/DRAC, kalibrator |
| viz/ | Annotatsiyalangan video, timeline, risk egri chizig'i (sayt uchun) |

## 4. Modul 0: sahnani kalibrlash (scene.json)

Kamera qo'zg'almas, shuning uchun yo'l geometriyasini bir marta aniq chizib olish — butun yechimning poydevori. 9 ta klass to'g'ridan-to'g'ri shu geometriyaga bog'liq.

scene.json tarkibi (piksel koordinatalarda, kadr o'lchamiga normallashtirilgan):

| Element | Turi | Qaysi klasslar ishlatadi | Qanday olinadi |
|---|---|---|---|
| road_mask | polygon | hammasi (yo'ldan tashqaridagi shovqinni kesish) | Trajektoriya heatmap + qo'lda tuzatish |
| lanes[] | polygon + yo'nalish vektori + "turn_allowed" | wrong_way, illegal_turn, congestion | Avto: trajektoriya yo'nalishlarini klasterlash; qo'lda tasdiqlash |
| solid_lines[] | polyline | solid_line_crossing | Qo'lda |
| stop_lines[] | chiziq + qaysi svetoforga bog'liq | red_light, stop_line | Qo'lda |
| crosswalks[] | polygon | jaywalking, failure_to_yield | Qo'lda |
| no_u_turn_zones[] | polygon | illegal_u_turn | Qo'lda (belgilar va chiziqlarga qarab) |
| traffic_lights[] | kichik ROI + qaysi polosalarga tegishli | red_light, stop_line, stopped_vehicle | Qo'lda |
| intersection | polygon | red_light, stop_line | Qo'lda |
| homography | 3×3 matritsa | tezlik, TTC, PET (metrda) | 4 nuqta: polosa kengligi ≈ 3.5 m, chiziq uzunliklari |
| static_ignore[] | polygon | hammasi | Doim turgan mashinalar (parkovka), reklama ekranlari |

Avtomatik qism (scripts/build_scene.py):

- Barcha sample videolarda YOLO + tracker ishlatiladi, treklar yig'iladi.
- Har bir 16×16 katak uchun o'rtacha harakat yo'nalishi hisoblanadi (flow field). Qarama-qarshi yo'nalishlar bimodal taqsimot bilan ajratiladi.
- Trajektoriyalar boshlanish-tugash zonalari bo'yicha klasterlanadi (DBSCAN) — bu ruxsat etilgan "harakat yo'llari" (movements) ro'yxatini beradi. Kam uchraydigan yoki umuman yo'q yo'llar — illegal_turn / illegal_u_turn kandidatlari.
- Natija ustiga qo'lda chizilgan elementlar qo'shiladi.

Qo'lda qism: kichik HTML/OpenCV tool (tools/scene_editor) — bitta kadr ustida polygon va chiziqlar chiziladi, JSON saqlanadi. Taxminan 1–2 soat ish.

Saytdagi bonus: sahna xaritasi (polosalar, yo'nalish strelkalari, flow field) EDA bo'limida ko'rsatiladi — bu "EDA natijalari yechimni shakllantirdi" mezonini to'g'ridan-to'g'ri qondiradi.

**EDA topilmasi — sahna geometriyasi:** Referens video — C3896. Boshqa videolar reference kadrga ECC/ORB orqali moslashtiriladi (homografiya), geometriya bir marta chiziladi. O'lchangan siljish: C3902 boshqalarga nisbatan taxminan 100 px (4K) — orol svetofori normallashgan koordinatalarda (0.60, 0.355) → (0.575, 0.375). Bitta video ichida kamera qimirlamaydi. Ikkita zebra (asosiy ko'ndalang va old plandagi diagonal) va stop chizig'i (1080p'da y≈480–520) barcha videolarda joyi bir xil — faqat kamera siljishi hisobga olinadi.

## 5. Modul 1: perception va track store

Perception videoni ≈15 fps'da (29.97 fps manbadan har 2-kadr) bir marta o'qiydi va har bir obyekt uchun metrlardagi silliq trek qaytaradi. Boshqa barcha modullar faqat shu treklar bilan ishlaydi.

5.1 Detektor

- Model: YOLO11m (yoki YOLO26m), imgsz 960–1280 (uzoqdagi piyodalar kichik bo'ladi), TensorRT yoki ONNX fp16.
- Klasslar: person, bicycle, car, motorcycle, bus, truck (COCO) + ixtiyoriy fine-tune.
- Domen moslashtirish (distillation): sample kadrlarda YOLO11x @1280 va Qwen3-VL bilan oflayn pseudo-label chiqarib, YOLO11m'ni 10–20 epoch fine-tune qilamiz. Bu kechki va uzoq obyektlarda aniqlikni oshiradi.
- Confidence 0.25, NMS 0.6; road_maskdan tashqaridagi detektsiyalar tashlanadi (piyodalar bundan mustasno — ular yo'l chetida ham kerak).

5.2 Tracker

- BoT-SORT, ReID va GMC o'chiq (kamera qo'zg'almas), track_buffer ≈ 2 s (qisqa okklyuziyalar uchun).
- Treklar o'rtasidagi uzilishlar (ID switch) post-hoc bog'lanadi: oxirgi va birinchi nuqta masofasi + yo'nalish + vaqt oralig'i < 1.5 s.

5.3 Track store (tracks/ — har video uchun bitta NumPy/Parquet jadval)

| Maydon | Ta'rif |
|---|---|
| track_id, cls, t | Obyekt, klass, vaqt (s) |
| bbox, conf | Piksel box va ishonch |
| xy_m | Box pastki markazi homografiya orqali metrda |
| v, a, heading | Tezlik (m/s), tezlanish (m/s²), yo'nalish; Savitzky–Golay silliqlash |
| lane_id, zone_flags | Qaysi polosada, qaysi zonada (crosswalk, intersection, stop_line oldi/orqa) |
| light_state | O'sha polosaga tegishli svetofor holati |

5.4 Svetofor holati

- Har traffic_lights[] ROI'da HSV bo'yicha qizil/sariq/yashil piksel ulushi.
- 0.5 s oynada ko'pchilik ovozi + holat mashinasi (qizil → yashil → sariq → qizil) bilan shovqin yo'qotiladi.
- Zaxira: svetofor ko'rinmasa yoki tunda noaniq bo'lsa, fazani oqimdan aniqlaymiz — stop chizig'idan hech kim o'tmaydigan va navbat turgan davr = qizil.

5.5 Fon modeli (MOG2)

- Oldinga va orqaga (videoni teskari o'qib) ikki MOG2 modeli, 2 fps'da. Orqaga modelda to'xtagan obyektlar aniqroq ko'rinadi (AI City g'oliblari usuli).
- Uzoq vaqt turgan, lekin hech qaysi trekka tegishli bo'lmagan foreground bloblar → road_obstacle kandidatlari.

5.6 Olov va tutun

- YOLOE / YOLO-World promptlari ("fire", "smoke", "debris", "fallen object", "animal") 1 fps'da + HSV/tekstura filtri (tutun: kulrang, sekin kengayadigan, past kontrast).
- Barcha kandidatlar VLM verifierdan o'tadi (bu klasslarda false positive juda qimmat).

**EDA topilmasi — dekodlash tezligi:** Lokal Apple M4 (10 yadro) da cv2 bilan 4K ketma-ket dekod 79 fps (≈2.6× real vaqt). Bu dasturiy, ko'p oqimli CPU dekodi (harness ishlaganda ~6.7 yadro band bo'ldi), apparat dekoder emas. 8 yadroli T4/Linux eval mashinasida bu ancha sekin bo'lishi mumkin — GPU dekod yo'lini (NVDEC: ffmpeg -hwaccel cuda / PyNvVideoCodec) birinchi kunning oxirigacha T4'da (Colab/Kaggle) alohida tekshirish shart, xavf hali yopilmagan.

## 6. Modul 2: event engine — 14 klass uchun qoidalar

Har klass — alohida modul, umumiy interfeys: rule.run(track_store, scene) -> list[Candidate(start, end, label, conf, evidence)]. Start/end tashkilotchilarning annotatsiya konvensiyasiga aynan mos qilib yoziladi.

| Klass | Trigger (qoida) | Start | End | Qiyinlik | VLM? |
|---|---|---|---|---|---|
| stopped_vehicle | v < 0.5 m/s ≥ 10 s, yo'lda, svetofor navbatida emas (oldida harakatlanuvchi navbat yo'q yoki chiroq yashil) | To'xtagan payt (orqaga backtrack) | Qayta harakatlangan / kadrdan chiqqan | Oson | Yo'q |
| congestion | Bir yo'nalishning barcha polosalarida o'rtacha v < 2 m/s va zichlik > chegara, ≥ 15 s (yashil fazada ham) | Navbat to'xtagan payt | Navbat tarqalgan payt | Oson | Yo'q |
| wrong_way | Trek heading'i polosa yo'nalishiga > 120°, ≥ 1 s va ≥ 3 m | Qarama-qarshi polosaga kirgan payt | To'g'ri polosaga qaytgan / chiqqan | O'rta | Ixtiyoriy |
| red_light | Qizil paytda old qism stop chizig'ini kesib, chorrahaga kiradi | Stop chizig'ini kesgan payt | Chorraha/kadrdan chiqqan | O'rta | Yo'q |
| stop_line | Qizil paytda stop chizig'idan o'tib to'xtaydi, chorrahaga kirmaydi | To'xtagan payt | Chiroq yashil bo'lgan payt | O'rta | Yo'q |
| jaywalking | Piyoda carriageway'da, crosswalk tashqarisida, ≥ 1 s | Yo'lga qadam qo'ygan payt | Yo'ldan chiqqan payt | O'rta | Yo'q |
| failure_to_yield | Crosswalk'da (yoki unga qadam qo'yayotgan) piyoda bor paytda mashina crosswalk'dan o'tadi | Mashina crosswalk'ga kirgan | Crosswalk'dan chiqqan | O'rta | Yo'q |
| solid_line_crossing | Trek nuqtasi yaxlit chiziqni kesadi va polosa almashadi | Chiziqni kesgan payt | To'liq yangi polosada | O'rta | Yo'q |
| illegal_turn | Movement klasteri ruxsat etilmagan (lane.turn_allowed) yoki noto'g'ri polosadan burilish | Burilish boshlangan (heading o'zgarishi > 15°) | Burilish tugagan | Qiyin | Ha |
| illegal_u_turn | Kumulyativ heading o'zgarishi > 150° no_u_turn zonasida | Burilish boshlangan | Burilish tugagan | O'rta | Ha |
| accident | Box'lar IoU > 0 + keskin sekinlashish (> 4 m/s²) + keyin ikkala obyekt to'xtaydi; yoki frame-difference/optical-flow pik | Birinchi kontakt kadri | Hamma obyekt to'xtagan / chiqqan | Qiyin | Ha |
| near_miss | TTC < 1.0 s yoki PET < 1.0 s + keskin tormoz/burilish, kontakt yo'q | Evasiv harakat boshlangan | Obyektlar ajralgan | Qiyin | Ha |
| road_obstacle | MOG2 statik blob, trekka tegishli emas, yo'lda ≥ 3 s; yoki open-vocab detektsiya | Paydo bo'lgan | Olib tashlangan | Qiyin | Ha |
| fire_smoke | Open-vocab detektsiya + HSV/tekstura, ≥ 2 s barqaror | Birinchi tutun | Tarqalgan / video oxiri | Qiyin | Ha |

Umumiy qoidalar:

- Klasslar chegarasi: accident bo'lsa, o'sha juftlik uchun near_miss chiqarilmaydi. stop_line bo'lsa, o'sha mashina uchun red_light chiqarilmaydi (chorrahaga kirmagan).
- Bir vaqtda bir xil klass: ikki bir vaqtdagi stopped_vehicle bitta segmentga birlashtiriladi (FAQ shunday deydi).
- Parametrlar configs/rules.yaml ichida, dev set bo'yicha grid-search bilan sozlanadi, kodda "sehrli son" yo'q.
- Accident anatomiyasi: avval arzon signal (IoU + tezlanish + optical flow piki) yuqori recall bilan kandidat beradi, keyin VLM precision'ni tiklaydi. Kontakt kadri ±0.5 s oynada native 29.97 fps'da qayta aniqlanadi.

## 7. Modul 3: VLM verifier va Modul 4: post-processing

VLM hech qachon butun videoni ko'rmaydi — faqat qoidalar bergan shubhali oynalarni tasdiqlaydi yoki rad etadi. Bu ACCIDENT@CVPR 2026'dagi eng yaxshi yechimlar ishlatgan coarse-to-fine sxema, lekin T4 byudjetiga moslashtirilgan.

7.1 VLM verifier (Qwen3-VL-2B-Instruct, fp16)

- Kandidat oynasi: [start − 2 s, end + 2 s], 8–12 kadr, 448–640 px.
- Kadrlarga tegishli obyektlarning box'lari va ID'lari chizilgan holda beriladi, matnda koordinatalar: [Car 3: (0.49,0.62)-(0.59,0.69)].
- Prompt klass ta'rifini aynan tashkilotchilar jadvalidan oladi va qat'iy JSON so'raydi: {"is_event": bool, "confidence": 0-1, "start_frame_idx": int, "end_frame_idx": int}.
- Greedy decoding, max_new_tokens ≈ 64 — deterministik va tez.
- Yakuniy ishonch = qoida ishonchi × VLM ishonchi (dev set'da kalibrlanadi). VLM qaytargan kadr indekslari chegaralarni aniqlashtirish uchun ishlatiladi.
- Xatoga chidamlilik: JSON buzilsa yoki OOM bo'lsa → qoida natijasi o'zgarishsiz qoladi. Byudjet tugasa → VLM o'chiriladi (9-bo'lim).

VLM ishlatiladigan klasslar: accident, near_miss, fire_smoke, road_obstacle, illegal_turn, illegal_u_turn, ixtiyoriy wrong_way. Bitta chaqiruv T4'da taxminan 1–3 s (o'lchanadi); bir videoga ≤ 30 chaqiruv limiti.

7.2 Post-processing (post/)

- Hysteresis: hodisa yuqori chegarada ochiladi, pastki chegarada yopiladi (titrashni yo'qotadi).
- Merge: bir klassning < 1.5 s oraliqdagi segmentlari birlashtiriladi.
- Min davomiylik: klassga xos (masalan, stopped_vehicle ≥ 10 s, jaywalking ≥ 1 s, accident ≥ 1 s). Qisqa bliplar tashlanadi.
- Chegara aniqlashtirish: ≈15 fps natijasi ±1 s oynada native 29.97 fps'da qayta hisoblanadi. Start — tashkilotchi konvensiyasi bo'yicha (masalan, stopped_vehicle uchun v < 0.5 m/s bo'lgan birinchi kadr).
- Per-class threshold: dev set'da har klass uchun F1 (IoU 0.3/0.5/0.7 o'rtachasi) maksimallashadigan chegara tanlanadi. Dev set'da ishonchli signal bo'lmagan klass o'chirib qo'yiladi (bu testda noto'g'ri klass qo'shib o'rtachani tushirishdan yaxshiroq).
- Validatsiya: 0 ≤ start < end ≤ duration, bir klass ichida overlap yo'q, label faqat CLASSES'dan — harness tashlashidan oldin o'zimiz tozalaymiz.

## 8. Modul 5: Part B — RiskEstimator

Risk fizik konflikt signallaridan (TTC, PET, DRAC) hisoblanadi va kichik kalibrator bilan ehtimollikka aylantiriladi. Og'ir o'qitiladigan anticipation modeli kerak emas: dashcam modellari (RARE va boshqalar) CCTV'da sinalmagan.

8.1 Oqim (har step chaqiruvida)

- Har 3-kadrda (≈ 10 fps) yengil detektor (YOLO11s, 640 px, TensorRT fp16) + ByteTrack. Oraliq kadrlarda oxirgi score qaytariladi.
- Treklar homografiya orqali metrga o'tkaziladi, tezlik va tezlanish faqat o'tgan kadrlar bo'yicha (causal EMA / Kalman) hisoblanadi.
- Yaqin juftliklar (≤ 25 m) uchun belgilar hisoblanadi, keyin kalibrator score beradi.
- Score EMA bilan silliqlanadi, [0, 1] oralig'ida qaytariladi. reset(meta) barcha holatni tozalaydi.

8.2 Belgilar (features)

| Belgi | Ma'nosi |
|---|---|
| min_TTC | Doimiy tezlik farazida juftlik to'qnashuvigacha vaqt (s), box o'lchamlari hisobga olinadi |
| min_PET_pred | Bashorat qilingan kesishuv nuqtasiga kirish vaqtlari farqi |
| max_DRAC | To'qnashuvdan qochish uchun kerak bo'ladigan sekinlashish (m/s²) |
| max_decel, max_yaw_rate | Keskin tormoz yoki keskin burilish |
| wrong_way_active, red_runner_active | Xavfli qoidabuzarlik davom etyapti |
| ped_on_road_near_vehicle | Yo'ldagi piyoda va unga yaqinlashayotgan mashina |
| motion_energy_z | Kadrlar farqining z-score'i (to'satdan o'zgarish) |
| n_vehicles_intersection | Chorrahadagi mashinalar soni (kontekst) |

8.3 Kalibrator va o'qitish ma'lumoti

- Model: LightGBM (≤ 200 daraxt) yoki logistic regression; natija isotonic regression bilan kalibrlanadi, shunda 0.5 ≈ "5 s ichida avariya ehtimoli yuqori".
- Pozitivlar: accident start s dan oldingi [s − 5, s) kadrlar. Manbalar: o'z dev labellarimiz, ACCIDENT datasetining real CCTV qismi (impact kadri belgilangan; CC BY-NC-ND — README'da ko'rsatiladi), CARLA sintetik qismi.
- Near-miss atrofidagi kadrlar e'tiborga olinmaydi (metrika ham shunday qiladi) — o'qitishda ham chiqarib tashlanadi.
- Zaxira (agar o'qitish ma'lumoti yetmasa): qo'lda sigmoid, risk = σ(a · (1.5 − min_TTC) + b · max_DRAC + …).

8.4 Alarm strategiyasi (Score_B = 0.4 AP + 0.4 F1_alarm + 0.2 mTTA/10)

- Score ≥ 0.5 faqat signal ≥ 0.4 s barqaror bo'lganda — bu yolg'on alarmlar sonini kamaytiradi (precision).
- Bir hodisa uchun bitta alarm: score bir marta ko'tarilgach, konflikt tugaguncha tushirilmaydi (2 s ichidagi alarmlar baribir birlashtiriladi).
- mTTA uchun: TTC 2–3 s bo'lganda ko'tarila boshlasin, 0.5 dan oshishi esa accident oldidan 2–4 s oldin bo'lsin.
- Ruxsat etilgan yo'nalish: Part A Part B risk egri chizig'idan accident kandidatlari uchun foydalanishi mumkin, teskarisi mumkin emas.

## 9. Runtime byudjeti va determinizm

Maqsad: Part A + Part B birga ≤ 1.5 × video davomiyligi (limit 3 ×). Quyidagi raqamlar 5 daqiqalik 4K 29.97 fps video (≈ 9 000 kadr) uchun taxmin; birinchi kunning o'zida T4 (yoki Colab/Kaggle T4) da o'lchanadi.

| Bosqich | Hajm | Taxminiy vaqt |
|---|---|---|
| Decode (Part A) | 9 000 kadr (4K), CPU thread | ~115 s (M4'da o'lchangan; NVDEC bilan kamroq) |
| Detektor + tracker (Part A) | 4 500 kadr, YOLO11m fp16 @960 | ~70–95 s |
| MOG2 oldinga + orqaga | 2 fps, 2 × 600 kadr | ~10 s |
| Olov/tutun open-vocab | 1 fps, 300 kadr | ~15 s |
| Qoidalar + post-processing | track store ustida | ~10 s |
| VLM verifier | ≤ 30 chaqiruv × ~2 s | ≤ 60 s |
| Chegara aniqlashtirish (native 29.97 fps oynalar) | kandidatlar atrofida | ~20 s |
| Part B (step) | 9 000 chaqiruv, detektor har 3-kadrda | ~70–95 s |
| Harness'ning 4K dekodi (Part B uchun) | 9 000 kadr, cv2 CPU | ~120 s (M4'da o'lchangan ≈ 0.4 ×) |
| Jami |  | ~8–9 daqiqa ≈ 1.6–1.8 × — 1.5 × maqsaddan oshadi; Part A uchun NVDEC yoki siyrakroq sampling shart |

Byudjet himoyasi (BudgetGuard):

- detect_events boshida taymer ishga tushadi; limit = 1.5 × davomiylik (Part B uchun joy qoladi).
- Bosqichlar ustuvorlik tartibida ishlaydi: perception → qoidalar → post-processing → VLM → native 29.97 fps refine. Limitning 60% iga yetilsa, VLM va refine o'tkazib yuboriladi va natija darhol qaytariladi.
- Modellar bir marta yuklanadi (modul darajasidagi lazy singleton), birinchi video oldidan warm-up.
- step ichida har chaqiruv < 3 ms bo'lishi kerak (detektor kadrlarida < 15 ms).

Determinizm:

- random, numpy, torch seed'lari qat'iy; torch.backends.cudnn.deterministic = True, benchmark = False.
- Kadr sampling qat'iy (har 2-kadr, indeks bo'yicha), VLM greedy decoding.
- Chiqish ro'yxati (start, label) bo'yicha saralanadi, float'lar 3 xonagacha yaxlitlanadi.
- TensorRT engine'lari GPU'ga bog'liq — shuning uchun standart backend PyTorch fp16, TensorRT faqat bayroq bilan va xatoda avtomatik PyTorch'ga qaytadi. Bu "runs as submitted" (Code rubric 40%) uchun xavfsizroq.

**EDA topilmasi — harness o'lchovi:** Stub solution bilan run_submission.py (M4) — Part B uchun kadrlarni o'qishning o'zi video davomiyligining ~0.4× ini oladi (C3896: 340 s video → 136 s). Eval CPU'da ~0.6–0.8× kutiladi; bu 3× byudjetdan oldindan ayriladi.

**Annotatsiya/test uchun nomzodlar (bitta kadr asosida, tasdiqlanmagan):** C3902 oxirida (≈315 s) qizil chiroqda zebra ustida to'xtagan mashinalar — stop_line/red_light uchun aniq nomzod. C3896 diagonal zebrada (≈113 s) kuryer moped — jaywalking yoki mixed-class edge case.

## 10. Repo tuzilmasi, kod sifati va submission

Repo tashkilotchi layoutiga aynan mos bo'ladi va ikki buyruq bilan toza mashinada ishlaydi; bu CI'da har commit'da tekshiriladi.

```
your-repo/
├── solution.py              # yupqa qatlam: CLASSES, detect_events, RiskEstimator → src/
├── run_submission.py        # starter kit, o'zgartirilmagan
├── evaluate.py              # starter kit, o'zgartirilmagan
├── requirements.txt         # versiyalar qat'iy (==)
├── Dockerfile               # zaxira, o'sha ikki buyruq
├── weights/download.sh      # HF Hub'dan, sha256 tekshiruvi bilan (≤ 5 GB)
├── configs/                 # scene.json, rules.yaml, thresholds.yaml, risk.yaml
├── src/                     # io, scene, perception, tracks, events, verify, post, risk, viz
├── scripts/                 # build_scene.py, train_detector.py, train_risk.py, tune_thresholds.py, render_samples.py
├── tools/scene_editor/      # geometriya chizish uchun kichik tool
├── labels/dev_gt.json       # o'z annotatsiyamiz (evaluate.py formatida)
├── tests/                   # format, overlap, determinizm, runtime testlari
├── notebooks/               # faqat EDA, asosiy kod src/ da
├── predictions_samples.json
└── README.md
```

Og'irliklar (taxminiy hajm, jami ≤ 5 GB):

| Fayl | Hajm |
|---|---|
| YOLO11m (fine-tuned) | ~40 MB |
| YOLO11s (Part B) | ~20 MB |
| YOLOE / YOLO-World (olov, tutun, to'siq) | ~50–100 MB |
| Qwen3-VL-2B-Instruct (fp16 safetensors) | ~4.3 GB (jami ~4.5 GB, limit ichida) |
| LightGBM risk modeli | < 1 MB |

Qaror: Qwen3-VL-2B-Instruct, fp16. Og'irliklar ~4.3 GB, jami ~4.5 GB — 5 GB limit ichida. T4 fp16'ni native qo'llaydi, AWQ/GPTQ kutubxonalari esa toza mashinada o'rnatish xavfini qo'shadi. 4B 4-bit'ning kichik aniqlik ustunligi "runs as submitted" xavfiga arzimaydi. Zaxira kichik bo'lgani uchun YOLO modellari jami ≤ 300 MB bo'lishi, weights/download.sh esa umumiy hajmni tekshirishi kerak.

Kod sifati (Code rubric):

- solution.py — 20–30 qator, faqat src/ ga yo'naltiradi.
- Type hint, docstring, ruff + black; o'lik kod va notebook-only mantiq yo'q.
- GitHub Actions: toza konteynerda pip install → bitta qisqa sample'da run_submission.py → evaluate.py --validate-only.
- Testlar: chiqish formati, bir klass ichida overlap yo'q, ikki marta ishga tushirilganda natija bir xil, runtime ≤ 1.5 ×.
- Har detect_events va step ichidagi xatolar ushlanadi va log qilinadi (bo'sh natija qaytarish — crash emas).

README majburiy bo'limlari: o'rnatish va ishga tushirish, og'irliklarni olish, arxitektura (learned vs rule-based), datasetlar va ularning litsenziyalari (ACCIDENT, UA-DETRAC, fire/smoke dataseti, COCO), seed'lar, jamoa va kim nima qilgani, ishlatilgan open-source kod atributsiyasi.

Submission: public GitHub repo + v1.0 tag (commit hash) + sayt linki. Deadline'dan kamida 2 soat oldin topshiriladi, keyin faqat saytga o'zgartirish kiritiladi.

## 11. Dev set, annotatsiya, baholash va ablationlar

Dev set'siz barcha threshold'lar taxmin bo'lib qoladi — shuning uchun sample videolarni annotatsiya qilish birinchi 6 soat ichidagi ishlardan biri.

11.1 Annotatsiya jarayoni

- Sample videolar 2–3 odam o'rtasida taqsimlanadi. Vosita: CVAT yoki Label Studio (video timeline), yoki oddiy jadval: video, start, end, label, izoh.
- Tezlashtirish: avval pipeline'ning o'zi (yuqori recall rejimida) va katta open VLM (ijaraga olingan GPU'da Qwen3-VL-32B, oflayn) kandidatlar chiqaradi; odam faqat tasdiqlaydi va chegaralarni tuzatadi.
- Konvensiya: start/end aynan tashkilotchi jadvalidagi ta'riflar bo'yicha. Shubhali holatlar labels/notes.md ga yoziladi.
- Natija labels/dev_gt.json (evaluate.py formatida). 20% videolar qayta tekshiriladi (ikkinchi odam).
- Samplelarda yo'q klasslar uchun: ACCIDENT datasetidan real CCTV accident kliplari, public fire/smoke kliplari va sun'iy "sanity" testlar (masalan, videoni teskari aylantirish → wrong_way).

11.2 Baholash

- python evaluate.py --pred predictions_dev.json --gt labels/dev_gt.json — Score A, Score B va har klass jadvali.
- Overfitting'dan saqlanish: threshold'lar leave-one-video-out bilan tanlanadi. Samplelar kam bo'lgani uchun parametrlar "fizik ma'noli" chegaralarda qoladi (masalan, TTC 0.5–2 s).
- Har o'zgarishdan keyin make eval — natijalar results/experiments.csv ga yoziladi (sayt uchun ham).

11.3 Ablationlar (saytda jadval sifatida)

| Tajriba | O'lchanadi |
|---|---|
| YOLO11m vs YOLO11s vs RF-DETR | Score A, runtime |
| 5 / 15 / 29.97 fps | Score A @IoU 0.7, runtime |
| Tracker: BoT-SORT vs ByteTrack | ID switch soni, Score A |
| VLM bilan vs VLM'siz | Har klass precision/recall |
| Chegara aniqlashtirish bilan vs usiz | F1 @IoU 0.7 |
| Part B: qo'lda sigmoid vs LightGBM | AP, F1_alarm, mTTA |

Xato tahlili: confusion matrix (accident ↔ near_miss, red_light ↔ stop_line, stopped_vehicle ↔ congestion), har klass bo'yicha 2–3 ta muvaffaqiyatsiz misol — saytga halol yoziladi.

## 12. Veb-sayt va live demo arxitekturasi

Sayt ikki qismdan iborat: statik Next.js sayt (Vercel) barcha oldindan hisoblangan natijalarni ko'rsatadi, alohida demo backend (Hugging Face Space, Docker) esa yuklangan videoni qayta ishlaydi. Demo ishdan chiqsa ham saytning qolgan qismi ishlaydi.

flowchart LR  
  U[Foydalanuvchi] --> W[Next.js sayt\nVercel]  
  W -->|statik JSON, MP4| CDN[(Sample natijalari\nannotated video, timeline,\nrisk, EDA)]  
  W -->|POST /jobs video| API[FastAPI\nHF Space, Docker]  
  API --> Q[Job navbati\n1 worker]  
  Q --> PIPE[Xuddi shu src/ pipeline\nCPU rejimi]  
  PIPE --> R[(events.json\nrisk.json\nannotated.mp4)]  
  W -->|GET /jobs/id SSE progress| API  
  API --> R

12.1 Demo backend

- POST /jobs (mp4, ≤ 2 daqiqa, ≤ 100 MB) → job_id; GET /jobs/{id} → progress (SSE, foizlar va bosqich nomi); GET /jobs/{id}/result → events, risk, annotated MP4 (H.264, ffmpeg).
- CPU rejimi: YOLO11s @640, 5 fps sampling, VLM o'chiq — bu saytda ochiq yoziladi. 2 daqiqalik video uchun maqsad ≤ 2 daqiqa.
- Himoya: fayl turi va davomiyligi tekshiruvi, bitta vaqtda 1 ta job, navbat holati ko'rsatiladi, xatoda tushunarli xabar ("crash yo'q" — rubric).
- Zaxira: HF Space uxlab qolsa — "sample videoni sinab ko'rish" tugmasi oldindan hisoblangan natijani ko'rsatadi; ikkinchi zaxira sifatida Render yoki o'z VPS.

12.2 Sahifalar (rubric bilan bog'langan)

| Sahifa | Mazmuni | Rubric |
|---|---|---|
| Bosh sahifa | Bir jumlada yechim, asosiy raqamlar (dev Score A/B), demo tugmasi | Design 10% |
| Live demo | Yuklash, progress bar, annotated video + interaktiv timeline (bosilganda video o'sha joyga o'tadi) + risk egri chizig'i + hodisalar jadvali + JSON yuklab olish | Demo 30% |
| Samplelar natijalari | Har sample video: annotated playback, timeline, risk, har klassdan misol kliplar, muvaffaqiyatsiz holatlar | Vizualizatsiya 20% |
| EDA | Rezolyutsiya/fps/davomiylik, yorug'lik, vaqt bo'yicha obyektlar soni, harakat heatmap, trajektoriyalar va polosa yo'nalishi maydoni, zichlik; har grafik ostida "bu yechimga qanday ta'sir qildi" | EDA 15% |
| Yondashuv va hisobot | Pipeline diagrammasi, modellar va datasetlar, learned vs rule-based jadvali, ablationlar, nima ishladi/ishlamadi, keyingi qadamlar | Approach 15% |
| Operator dashboard | Soat, polosa va klass bo'yicha hodisalar, eng xavfli zonalar xaritasi | Extras |
| Jamoa | A'zolar, rollar, kim nima qildi, GitHub/LinkedIn/portfolio, oldingi loyihalar | Team 10% |
| Linklar | Repo, og'irliklar, predictions_samples.json | Majburiy |

12.3 Texnik tafsilotlar

- Grafiklar: interaktiv (ECharts yoki Plotly), statik rasm emas.
- Mobilga mos, Lighthouse ≥ 90, video lazy-load.
- viz/ moduli bir xil kod bilan saytdagi barcha annotated video va JSON'larni generatsiya qiladi (scripts/render_samples.py) — repo va sayt natijalari mos kelishi kafolatlanadi (Reproducibility 25%).
- Qo'shimcha: brauzer webcam orqali jonli rejim (WebSocket, 2–5 fps) — vaqt qolsa.

## 13. Jamoa rollari va soatbay reja

Deadline'gacha ~35 soat qoldi (27-sentabr, 23:59). Reja har bosqich oxirida "gate" bilan: gate o'tmasa, keyingi bosqichga emas, zaxira variantga o'tiladi.

Rollar

| Rol | Mas'uliyat | Modullar |
|---|---|---|
| A — Perception va Part B | Detektor, tracker, homografiya, risk estimator, runtime | perception/, tracks/, risk/ |
| B — Event va baholash | scene.json, 14 qoida, post-processing, VLM, dev labellar, threshold tuning | scene/, events/, post/, verify/ |
| C — Sayt, demo, paket (Bilol) | Next.js sayt, FastAPI demo, EDA, render, README, Docker, CI | viz/, site/, demo/, repo infra |

Jadval (Toshkent vaqti)

| Vaqt | A | B | C | Gate |
|---|---|---|---|---|
| 26-sen 13:00–16:00 | Samplelarni ko'rish, T4 muhit (Kaggle/Colab), YOLO + tracker → track store | Samplelarni ko'rish, scene_editor, scene.json v1 | Repo skeleti, starter kit, bo'sh solution.py, CI, Dockerfile | G1: bo'sh paket toza muhitda ikki buyruq bilan ishlaydi |
| 16:00–20:00 | Homografiya, tezlik/tezlanish, runtime o'lchovi | Dev labellar (hamma birga 2 soat), oson qoidalar: stopped_vehicle, congestion, wrong_way | EDA skriptlari, sayt skeleti (Vercel), sahifa strukturasi | G2: dev_gt.json tayyor; 3 klass evaluate.py'da ball beradi |
| 20:00–02:00 | Part B: TTC/PET/DRAC, qo'lda sigmoid versiya | red_light, stop_line, jaywalking, failure_to_yield, solid_line | FastAPI demo backend + HF Space, render_samples.py | G3: 8+ klass, Part B ishlaydi, demo bitta video bilan ishlaydi |
| 02:00–08:00 | Uyqu (navbat bilan) | Uyqu | Uyqu | — |
| 27-sen 08:00–13:00 | Fine-tune detektor (pseudo-label), risk kalibratori | accident, near_miss, turns, road_obstacle, fire_smoke + VLM verifier | Demo UI: timeline, risk chart, annotated player | G4: hamma klass; to'liq pipeline ≤ 1.5 × |
| 13:00–18:00 | Runtime optimizatsiya, determinizm testlari | Threshold tuning, ablationlar, xato tahlili | Samplelar sahifasi, dashboard, Jamoa sahifasi | G5: predictions_samples.json va sayt natijalari bir xil |
| 18:00–21:00 | Toza mashinada yakuniy test, Docker | README yondashuv qismi, hisobot sahifasi | Dizayn, mobil, Lighthouse | G6: code freeze 21:00 |
| 21:00–22:00 | Tag v1.0, submit | Hisobotni tekshirish | Linklarni tekshirish | Submit ≤ 22:00 |
| 22:00–23:59 | Zaxira vaqt | Zaxira vaqt | Faqat sayt tuzatishlari | — |

Ish tartibi: har 3–4 soatda 10 daqiqalik sinxronizatsiya; mainga faqat CI yashil bo'lsa merge; har bir modul avval eng oddiy ishlaydigan versiyada, keyin yaxshilanadi.

## 14. Risklar, zaxira rejalar va ochiq savollar

Eng katta xavf — paketning tashkilotchi mashinasida ishlamay qolishi (model balli 0). Shuning uchun toza muhit testi birinchi gate va har commit'da takrorlanadi.

| Risk | Ehtimollik | Ta'sir | Zaxira reja |
|---|---|---|---|
| Paket toza mashinada ishlamaydi | O'rta | Model balli 0 | CI'da toza konteyner, versiyalar qat'iy, Dockerfile, PyTorch fp16 standart backend |
| Runtime limitdan oshadi | O'rta | Video bo'sh deb baholanadi | BudgetGuard, VLM va refine o'chiriladi, fps pasayadi |
| VLM (2B) T4'da sekin yoki aniqligi yetmaydi | Yuqori | Kam uchraydigan klasslarda precision tushadi | Chaqiruvlar limitini kamaytirish yoki VLM'ni o'chirish; qattiqroq qoida threshold'lari |
| Svetofor ko'rinmaydi / tunda noaniq | O'rta | red_light, stop_line ishlamaydi | Fazani oqimdan aniqlash |
| Samplelarda ba'zi klasslar yo'q | Yuqori | Threshold'lar tuning qilinmaydi | ACCIDENT va fire/smoke datasetlari, fizik ma'noli standart qiymatlar, ishonch past bo'lsa klassni o'chirish |
| Tracker ID switch'lari | O'rta | Noto'g'ri wrong_way/turn | Post-hoc trek bog'lash, minimal davomiylik |
| HF Space uxlaydi yoki qulaydi | O'rta | Demo 30% yo'qoladi | Oldindan hisoblangan misol, zaxira hosting, keep-alive ping |
| Vaqt yetmaydi | Yuqori | Chala topshirish | Gate'lar, 21:00 code freeze, 22:00 submit |

Ochiq savollar (javoblar rejani o'zgartiradi):

- ☐ Nechta sample video bor, qancha davom etadi, kunduzmi yoki kechasi ham bormi?
- ☐ Kadrda svetofor, stop chizig'i va piyoda o'tish joyi ko'rinadimi?
- ☐ Qaysi GPU bor (o'zimizniki, Colab/Kaggle T4, ijara)? Katta VLM bilan pseudo-label uchun A100 ijarasi kerakmi?
- ☐ VLM: Qwen3-VL-2B fp16 tanlandi. G4 gate'da runtime va dev set aniqligi o'lchanadi; foyda bermasa, VLM o'chiriladi.
- ☐ Jamoaning qolgan ikki a'zosi kim va qaysi rollarni oladi?
- ☐ Tashkilotchilar hardware yoki qoidalarni o'zgartirganmi (Telegram guruh va FAQ tekshiriladi)?

## Manbalar

- Zero-Shot Traffic Accident Detection via a Coarse-to-Fine VLM-Tracking Pipeline
- Two-Pass Zero-Shot Temporal-Spatial Grounding of Rare Traffic Events
- ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos
- A Modular Zero-Shot Pipeline for Accident Detection (ACCIDENT@CVPR 2026)
- Good Practices and A Strong Baseline for Traffic Anomaly Detection
- RARE: Real-time Traffic Accident Anticipation with Feature Reuse
- Qwen3-VL-4B-Instruct (Hugging Face)
- Qwen3-VL-4B vs 8B: VRAM guide
- Best Object Detection Models 2026 (Roboflow)
- WIUT Hackathon 2026 — Elimination Task
