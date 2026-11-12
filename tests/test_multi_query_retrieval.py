import unittest
from threading import Barrier
from types import SimpleNamespace

from multi_query_retrieval import (
    build_intent_query_groups,
    build_query_variants,
    format_evidence,
    search_intent_groups,
    source_filenames,
)


def hit(file_id, filename, text, attributes=None, score=None):
    return SimpleNamespace(
        file_id=file_id,
        filename=filename,
        attributes=attributes,
        score=score,
        content=[SimpleNamespace(text=text)],
    )


class FakeVectorStores:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.results_by_query[kwargs["query"]])


def search_one_intent(client, vector_store_id, variants, **kwargs):
    return search_intent_groups(
        client,
        vector_store_id,
        [{"intent": "default", "queries": variants}],
        **kwargs,
    )


class MultiQueryRetrievalTests(unittest.TestCase):
    def test_search_preserves_small_scalar_file_attributes(self):
        client = SimpleNamespace(
            vector_stores=FakeVectorStores(
                {"policy query": [hit("f1", "policy.docx", "Policy text", {"status": "draft", "year": 2026}, 0.91)]}
            )
        )
        results = search_one_intent(
            client,
            "vs_test",
            [("canonical", "policy query")],
        )

        self.assertEqual(results[0]["attributes"], {"status": "draft", "year": 2026})
        self.assertEqual(results[0]["attributes_status"], "available")
        self.assertEqual(results[0]["final_rank"], 1)
        self.assertEqual(results[0]["provider_scores"], {"canonical": 0.91})
        self.assertEqual(len(results[0]["passage_id"]), 64)

    def test_one_query_preserves_provider_order_and_scores_without_rrf(self):
        client = SimpleNamespace(vector_stores=FakeVectorStores({
            "query": [
                hit("f1", "first.docx", "First provider result", score=0.4),
                hit("f2", "second.docx", "Second provider result", score=0.95),
            ]
        }))
        results = search_one_intent(client, "vs_test", [("canonical", "query")])

        self.assertEqual([item["filename"] for item in results], ["first.docx", "second.docx"])
        self.assertEqual(
            [item["provider_scores"]["canonical"] for item in results],
            [0.4, 0.95],
        )
        self.assertEqual([item["rrf_score"] for item in results], [0.0, 0.0])

    def test_passage_id_is_stable_for_file_and_exact_content(self):
        first = search_one_intent(
            SimpleNamespace(vector_stores=FakeVectorStores({
                "query": [hit("file-1", "policy.docx", "Same text")]
            })),
            "vs_test",
            [("canonical", "query")],
        )[0]
        second = search_one_intent(
            SimpleNamespace(vector_stores=FakeVectorStores({
                "query": [hit("file-1", "renamed.docx", "Same text")]
            })),
            "vs_test",
            [("canonical", "query")],
        )[0]
        changed = search_one_intent(
            SimpleNamespace(vector_stores=FakeVectorStores({
                "query": [hit("file-1", "policy.docx", "Changed text")]
            })),
            "vs_test",
            [("canonical", "query")],
        )[0]
        self.assertEqual(first["passage_id"], second["passage_id"])
        self.assertNotEqual(first["passage_id"], changed["passage_id"])

    def test_builds_distinct_original_canonical_and_alternative_queries(self):
        variants = build_query_variants(
            "ajil deer yu yu hiij boldoggui we",
            "Ажил дээр юу хийхийг хориглодог вэ?",
            [
                "Ажлын байранд юу хийхийг хориглодог вэ?",
                "Ажил дээр юу хийхийг хориглодог вэ?",
            ],
        )
        self.assertEqual(
            [label for label, _ in variants],
            ["original", "canonical", "alternative_1"],
        )

    def test_single_intent_keeps_the_existing_query_budget(self):
        groups = build_intent_query_groups(
            "latin original",
            [
                {
                    "intent": "Журмын хүсэлт",
                    "canonical_query": "Журмын хүсэлт юу вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": ["Журмын хүсэлтийг тайлбарла."],
                }
            ],
        )
        self.assertEqual(
            [query for _, query in groups[0]["queries"]],
            ["latin original", "Журмын хүсэлт юу вэ?", "Журмын хүсэлтийг тайлбарла."],
        )

    def test_strategy_b_v2_searches_only_the_canonical_query(self):
        request = [{
            "intent": "Чөлөөний журам",
            "canonical_query": "Чөлөөний хүсэлт хэрхэн батлуулах вэ?",
            "search_documents": True,
            "is_ambiguous": False,
            "alternatives": ["Чөлөөг хэн батлах вэ?"],
        }]
        expected = {
            "A": ["эх асуулт", "Чөлөөний хүсэлт хэрхэн батлуулах вэ?", "Чөлөөг хэн батлах вэ?"],
            "B": ["Чөлөөний хүсэлт хэрхэн батлуулах вэ?"],
            "C": ["Чөлөөний хүсэлт хэрхэн батлуулах вэ?"],
        }
        for strategy, queries in expected.items():
            with self.subTest(strategy=strategy):
                groups = build_intent_query_groups(
                    "эх асуулт", request, strategy=strategy
                )
                self.assertEqual(
                    [query for _, query in groups[0]["queries"]], queries
                )

    def test_strategy_b_v2_drops_latin_original_and_preserves_explicit_names(self):
        canonical = "SQL болон Skyteams, IPCC-ийн журам"
        groups = build_intent_query_groups(
            "sql bolon skyteams, ipcc-iin juram",
            [{
                "intent": "Баримтын журам",
                "canonical_query": canonical,
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": ["Ижил утгатай давхар query"],
            }],
            strategy="B",
        )
        self.assertEqual(groups[0]["queries"], [("canonical", canonical)])

    def test_strategy_b_searches_leave_without_clock_range_but_keeps_request_intact(self):
        canonical = (
            "Өдрийн 15:00–18:00 хүртэл чөлөө авахад хэн батлах, "
            "хэдэн цагийн өмнө мэдэгдэх шаардлагатай вэ? 3 цаг"
        )
        request = {
            "intent": "Чөлөөний журам",
            "canonical_query": canonical,
            "search_documents": True,
            "is_ambiguous": False,
            "alternatives": [],
        }

        groups = build_intent_query_groups("raw", [request], strategy="B")

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["queries"],
            [(
                "canonical",
                "чөлөө авахад хэн батлах хэдэн цагийн өмнө мэдэгдэх "
                "шаардлагатай вэ? 3 цаг",
            )],
        )
        self.assertEqual(request["canonical_query"], canonical)

    def test_strategy_b_keeps_ranges_for_non_leave_intents(self):
        query = "15:00–18:00 хүртэлх ажлын цаг"
        groups = build_intent_query_groups("raw", [{
            "intent": "Ажлын цагийн журам",
            "canonical_query": query,
            "search_documents": True,
            "is_ambiguous": False,
            "alternatives": [],
        }], strategy="B")
        self.assertEqual(groups[0]["queries"], [("canonical", query)])

    def test_compound_requests_are_searched_in_separate_groups(self):
        groups = build_intent_query_groups(
            "leave request and weather tomorrow",
            [
                {
                    "intent": "Чөлөө хүсэх журам",
                    "canonical_query": "Чөлөө авах хүсэлтийг хэрхэн гаргах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": ["Чөлөөний хүсэлтийг хаана өгөх вэ?"],
                },
                {
                    "intent": "Маргаашийн цаг агаар",
                    "canonical_query": "Маргааш цаг агаар ямар байх вэ?",
                    "search_documents": False,
                    "is_ambiguous": False,
                    "alternatives": [],
                },
                {
                    "intent": "Чөлөө батлах хүн",
                    "canonical_query": "Чөлөөний хүсэлтийг хэн батлах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": ["Хэн зөвшөөрөх вэ?"],
                },
            ],
        )

        self.assertEqual([group["intent"] for group in groups], ["Чөлөө хүсэх журам", "Чөлөө батлах хүн"])
        self.assertEqual(sum(len(group["queries"]) for group in groups), 4)
        self.assertNotIn(
            "leave request and weather tomorrow",
            [query for group in groups for _, query in group["queries"]],
        )

    def test_ambiguous_and_general_requests_are_not_retrieved(self):
        groups = build_intent_query_groups(
            "query",
            [
                {
                    "intent": "Хоёрдмол хүсэлт",
                    "canonical_query": "Хоёрдмол хүсэлт юу вэ?",
                    "search_documents": False,
                    "is_ambiguous": True,
                    "alternatives": [],
                },
                {
                    "intent": "Гадаад мэдээлэл",
                    "canonical_query": "Маргаашийн цаг агаар ямар вэ?",
                    "search_documents": False,
                    "is_ambiguous": False,
                    "alternatives": [],
                },
            ],
        )
        self.assertEqual(groups, [])

    def test_group_search_fuses_per_intent_then_interleaves_results(self):
        client = SimpleNamespace(
            vector_stores=FakeVectorStores(
                {
                    "leave query": [
                        hit("l1", "leave.docx", "Leave request steps."),
                        hit("l2", "leave.docx", "Leave approval details."),
                    ],
                    "approval query": [
                        hit("l2", "leave.docx", "Leave approval details."),
                        hit("l3", "leave.docx", "Leave date constraints."),
                    ],
                    "laptop query": [
                        hit("i1", "it.docx", "IT ticket steps."),
                        hit("i2", "it.docx", "IT contact details."),
                    ],
                }
            )
        )
        groups = [
            {
                "intent": "Чөлөөний журам",
                "queries": [("canonical", "leave query"), ("alternative_1", "approval query")],
            },
            {
                "intent": "Компьютерийн асуудал",
                "queries": [("canonical", "laptop query")],
            },
        ]
        results = search_intent_groups(client, "vs_test", groups, final_k=4)

        self.assertEqual(len(client.vector_stores.calls), 3)
        self.assertEqual([item["intents"][0] for item in results[:2]], ["Чөлөөний журам", "Компьютерийн асуудал"])
        self.assertEqual(len(results), 4)
        self.assertEqual([item["final_rank"] for item in results], [1, 2, 3, 4])
        self.assertTrue(all(len(item["passage_id"]) == 64 for item in results))
        self.assertIn("Relevant to: Чөлөөний журам", format_evidence(results))

    def test_search_trace_records_each_raw_query_and_final_fused_evidence(self):
        client = SimpleNamespace(vector_stores=FakeVectorStores({
            "leave query": [hit("f1", "leave.docx", "Leave evidence", {"year": 2026}, 0.8)],
            "approval query": [hit("f1", "leave.docx", "Leave evidence", {"year": 2026}, 0.7)],
        }))
        trace = {}

        results = search_intent_groups(
            client,
            "vs_test",
            [{"intent": "Чөлөө", "queries": [
                ("canonical", "leave query"),
                ("alternative_1", "approval query"),
            ]}],
            trace_capture=trace,
        )

        self.assertEqual([item["query"] for item in trace["queries"]], ["leave query", "approval query"])
        self.assertEqual(trace["queries"][0]["raw_results"][0]["text"], "Leave evidence")
        self.assertEqual(trace["queries"][0]["raw_results"][0]["rank"], 1)
        self.assertEqual(trace["fused_results"], results)
        self.assertEqual(trace["fused_results"][0]["final_rank"], 1)
        self.assertNotIn("vector_store_id", str(trace))

    def test_deduplicated_passage_keeps_each_intents_query_rank(self):
        client = SimpleNamespace(vector_stores=FakeVectorStores({
            "leave query": [hit("f1", "policy.docx", "Shared evidence", score=0.9)],
            "attendance query": [hit("f1", "policy.docx", "Shared evidence", score=0.7)],
        }))
        results = search_intent_groups(client, "vs_test", [
            {"intent": "Чөлөө", "queries": [("canonical", "leave query")]},
            {"intent": "Ирц", "queries": [("canonical", "attendance query")]},
        ])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["intents"], ["Чөлөө", "Ирц"])
        self.assertEqual(
            [item["intent"] for item in results[0]["query_provenance"]],
            ["Чөлөө", "Ирц"],
        )
        self.assertEqual(
            results[0]["provider_scores"],
            {"Чөлөө::canonical": 0.9, "Ирц::canonical": 0.7},
        )
        self.assertEqual(
            results[0]["rrf_score_by_intent"],
            {"Чөлөө": 0.0, "Ирц": 0.0},
        )

    def test_rrf_merges_duplicate_passages_and_preserves_source(self):
        client = SimpleNamespace(
            vector_stores=FakeVectorStores(
                {
                    "query one": [
                        hit("f1", "work-rules.docx", "No smoking at work."),
                        hit("f2", "safety.docx", "Report workplace hazards."),
                    ],
                    "query two": [
                        hit("f1", "work-rules.docx", "No smoking at work."),
                        hit("f3", "safety-2.docx", "Wear required protective equipment."),
                    ],
                }
            )
        )
        results = search_one_intent(
            client,
            "vs_test",
            [("canonical", "query one"), ("alternative_1", "query two")],
            per_query=2,
            final_k=3,
        )

        self.assertEqual(len(client.vector_stores.calls), 2)
        self.assertEqual(results[0]["filename"], "work-rules.docx")
        self.assertEqual(results[0]["ranks"], {"canonical": 1, "alternative_1": 1})
        self.assertEqual(
            source_filenames(results),
            ["work-rules.docx", "safety-2.docx", "safety.docx"],
        )
        self.assertIn("No smoking at work.", format_evidence(results))

    def test_search_requests_overlap_before_rrf(self):
        barrier = Barrier(3)

        class ConcurrentVectorStores:
            def search(self, **kwargs):
                barrier.wait(timeout=2)
                return SimpleNamespace(
                    data=[
                        hit(
                            kwargs["query"],
                            f"{kwargs['query']}.docx",
                            f"Evidence for {kwargs['query']}",
                        )
                    ]
                )

        results = search_one_intent(
            SimpleNamespace(vector_stores=ConcurrentVectorStores()),
            "vs_test",
            [("q1", "one"), ("q2", "two"), ("q3", "three")],
            per_query=1,
            final_k=3,
        )
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
