# RAG код цэвэрлэх төлөвлөгөө

## Зорилго

Кодыг зүгээр олон файлд хуваах биш, бодит шаардлагагүй логик, давхардсан өгөгдөл дамжуулалт, fallback болон салааллыг багасгаж RAG урсгалыг ойлгоход хялбар болгоно. Refactor хийх үед одоогийн гаднаас ажиглагдах зан төлөвийг хадгална. Шинэ reliability fix болон retrieval behavior-ийн өөрчлөлтийг цэвэрлэгээнээс тусад нь хийнэ.

## Одоогийн зураглал

Python-ийн production файлуудын хэмжээг мөрөөр тоолоход нийт **5,710 LOC** байна. Энэ нь blank/comment мөрийг хассан тоо биш, зөвхөн эхний хэмжээний baseline.

| Файл | LOC | Эхний үүргийн зураглал |
| --- | ---: | --- |
| `app.py` | 2,193 | Streamlit UI, session болон үндсэн RAG orchestration |
| `semantic_router.py` | 878 | Intent, request routing болон structured route боловсруулах |
| `query_rewriter.py` | 580 | Query rewrite ба validation flow |
| `context_resolver.py` | 415 | Follow-up болон өмнөх conversation context шийдэх |
| `multi_query_retrieval.py` | 344 | Олон query retrieval, fusion/deduplication |
| `query_precheck.py` | 257 | Хөнгөн шалгалт, урьдчилсан ангилал |
| `query_faithfulness.py` | 263 | Rewrite-ийн эх асуултад үнэнч байдлын шалгалт |
| `query_normalizer.py` | 214 | Query хэлбэржүүлэлт, normalization |
| `pipeline_trace.py` | 208 | Pipeline stage instrumentation, redaction, trace UI |
| `query_utils.py` | 90 | Query-тэй холбоотой туслах функцүүд |
| `targeted_retrieval.py` | 78 | Нэг чиглэлийн retrieval туслах |
| Бусад 4 жижиг модуль | 190 | `restriction_modules.py`, `chat_shortcuts.py`, `conversation_export.py`, `domain_rules.py` |

Файлын хэмжээ дангаараа хэт төвөгтэй эсвэл устгах шаардлагатайг нотлохгүй. Дараагийн audit нь функцийн caller, branch, өгөгдөл давхар хувиргаж буй цэг, тестийн хамаарлыг хамтад нь шалгана.

## `pipeline_trace`-ийн шийдвэр

`pipeline_trace.py` нь RAG-ийн үндсэн боловсруулалтад зайлшгүй шаардлагагүй instrumentation юм. Харин `app.py` олон үе шатанд trace бичиж, session message-д хадгалан UI-д үзүүлдэг; тусдаа тест ч бий. Иймд одоогоор шууд устгах нь зохимжгүй.

Цэвэрлэгээний audit дээр дараахыг шийднэ:

- Trace-ийг хөгжүүлэлт, оношилгоо, эсвэл хэрэглэгчид аль үед харуулах бодит шаардлага байна вэ?
- Нэг prompt/evidence олон алхмын `input`, `output`, `handoff` талбарт давхар хадгалагдаж байна уу?
- Давхардлыг цомхон event/diagnostic record болгон бууруулж болох уу?
- UI болон session-д trace хадгалахыг тохиргоогоор унтраах боломж шаардлагатай юу?
- Redaction ба failure оношилгоог хадгалахын тулд хамгийн багадаа ямар өгөгдөл хэрэгтэй вэ?

Шийдвэрийн зарчим: trace нь зөвхөн хөгжүүлэлтийн хэрэгсэл хэвээр бол lightweight, optional байлгах; бодит хэрэглээнд хэрэггүй бол caller болон тестүүдийг шалгасны дараа бүхэлд нь авах. Retrieval, routing, answer generation нь trace-ээс хамааралгүй хэвээр байх ёстой.

## Зорьж буй энгийн урсгал

```text
Хэрэглэгчийн мессеж + шаардлагатай history
        ↓
Route ба restriction шалгалт
        ↓
Evidence retrieval (шаардлагатай үед; олон query байвал fusion)
        ↓
Evidence-д тулгуурласан answer ба citations
        ↓
Одоогийн session/application layer-д хадгалах
```

Энэ бол логик урсгалын зорилт болохоос файлыг нэн даруй `pipeline.py`, `routing.py`, `retrieval.py`, `generation.py` болгон хуваах даалгавар биш. Одоогийн модулийг дахин ашиглах, нэгтгэхийг шинэ файл үүсгэхээс түрүүнд авч үзнэ.

## Ажлын дараалал

1. **Read-only audit:** production кодын функцийг caller-тэй нь зураглах; dead code, duplicated logic, давхар state, хэт олон branch/fallback болон config flag-ийг баримтжуулах. Мөн жижиг baseline хэмжилт гаргах.
2. **KEEP / SIMPLIFY / MERGE / DELETE тайлан:** санал бүрд нотолгоо, тестийн хамрах хүрээ, regression risk тэмдэглэх. Caller-гүй харагдсан кодыг external entry point-ийг шалгалгүйгээр устгахгүй.
3. **Dead code устгах:** импорт, helper, flag зэргийг зөвхөн ашиглалтгүй нь батлагдсаны дараа нэг багцаар цэвэрлэх.
4. **Давхардал нэгтгэх:** request хэлбэр, query validation, deduplication, formatting болон response processing-д canonical хэрэгжүүлэлт тогтоох.
5. **Control flow хялбарчлах:** үнэ цэнгүй wrapper, fallback болон стратегийн салааллыг зөвхөн одоогийн behavior-тэй тулган багасгах.
6. **Behavior-preserving verification:** refactor бүрийн дараа focused regression tests ажиллуулж, шаардлагатай тохиолдолд өмнөх ба дараах output/API call тоог харьцуулах.
7. **Reliability fix-үүдийг дараа нь хийх:** rewrite guard, relevance, retry зэрэг behavior-ийн өөрчлөлтийг тусдаа өөрчлөлт болгон гаргах.

## Refactor-д хамгаалах behavior

- Cyrillic, Latin/transliteration, English асуултын routing/rewrite
- Follow-up history болон өмнөх turn-ийн context
- Олон intent-тэй хүсэлтийн боловсруулалт
- Restriction/policy шалгалт ба refusal
- Citation-ийн эх сурвалж, document grounding
- Evidence хүрэлцэхгүй үед үүнийг зөв хэлэх
- Retrieval болон generation-ийн алдааны handling
- Ижил хүсэлтэд LLM/API call нэмэгдэхгүй байх

## Цэвэрлэгээ амжилттай эсэх

- Нийт production Python LOC буурсан эсвэл нэмэгдсэн мөр бүрийн шалтгаан тодорхой
- Функцийн caller болон data flow ойлгомжтой болсон
- Нэг хариуцлагад нэг үндсэн хэрэгжүүлэлттэй болсон
- Өндөр эрсдэлтэй branch/fallback цөөрсөн
- Адил хүсэлтэд API call нэмэгдээгүй
- Холбогдох regression tests бүгд амжилттай
- Unverified retrieval quality-г refactor-ийн үр дүн гэж зарлаагүй

## Хязгаар ба дараагийн алхам

Энэ баримт нь эхний хэмжээс болон цэвэрлэгээний төлөвлөгөө; одоогоор бүх функцийн caller graph, dead code, branch complexity-г баталгаажуулсан аудит биш. Дараагийн ажил бол эхлээд read-only audit хийж, мөр/функцийн түвшний KEEP / SIMPLIFY / MERGE / DELETE тайлан гаргах. Тэр тайланг хянасны дараа л код өөрчилнө.
