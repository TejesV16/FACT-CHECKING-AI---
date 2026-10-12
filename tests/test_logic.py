import unittest
from datetime import date
from unittest.mock import patch

from agent_checker import _parse_final_answer
from fact_checker import (
    _map_verdict, _evidence_strength, _tenure_duration_conflict,
    is_tenure_duration_claim, _current_office_support, _direct_claim_support,
    _compound_actor_award_support, _orbital_relation, score_source_against_claim,
    fetch_page_text,
)
from web_search import official_office_query
from api_server import _counterevidence_query
from summarizer import _fallback_summary
from fact_checker import verify_claim
from timing import total_seconds, merge_timings
from claim_extractor import _extract_claims_spacy, extract_claims


class ParsingAndDecisionTests(unittest.TestCase):
    def test_absolute_ev_lifecycle_claim_gets_counterevidence_query(self):
        self.assertEqual(
            _counterevidence_query("Electric vehicles produce zero emissions over their entire life cycle."),
            "electric vehicle lifecycle emissions manufacturing electricity generation study",
        )

    @patch("fact_checker._classify_pair", return_value={
        "contradiction": 0.07, "entailment": 0.10, "neutral": 0.83,
    })
    def test_neutral_nli_result_does_not_pass_evidence_threshold(self, _classify):
        result = score_source_against_claim(
            "A claim unrelated to this passage.", "An unrelated passage about astronomy.",
            tokenizer=None, model=None,
        )
        self.assertLess(result["score"], 0.55)

    def test_orbital_relation_keeps_subject_and_object_order(self):
        self.assertEqual(_orbital_relation("The Earth revolves around the Sun."), ("earth", "sun"))
        self.assertEqual(_orbital_relation("The Sun revolves around the Earth."), ("sun", "earth"))

    def test_reversed_earth_sun_claim_is_directly_contradicted(self):
        result = score_source_against_claim(
            "The Sun revolves around the Earth.",
            "Earth revolves around the Sun.", tokenizer=None, model=None,
        )
        self.assertEqual(result["label"], "contradiction")

    def test_actor_national_award_compound_claim_uses_two_retrieved_cues(self):
        self.assertTrue(_compound_actor_award_support(
            "Dhanush is a Tamil actor and he has won National Awards.",
            [
                {"title": "Dhanush - Wikipedia", "snippet": "Dhanush is an Indian actor."},
                {"title": "Dhanush wins Best Tamil Film at 72nd National Awards", "snippet": "Award result."},
            ],
        ))

    def test_direct_affirmative_sentence_beats_historical_opposite_context(self):
        evidence = (
            "The Earth revolves around the Sun. Heliocentrism is opposed to "
            "geocentrism, which placed the Earth at the center."
        )
        self.assertEqual(
            _direct_claim_support("The Earth revolves around the Sun.", evidence),
            "The Earth revolves around the Sun.",
        )

    def test_direct_match_does_not_treat_negated_claim_as_support(self):
        self.assertEqual(
            _direct_claim_support("The Earth revolves around the Sun.", "The Earth does not revolve around the Sun."),
            "",
        )

    def test_heuristic_extractor_keeps_orbital_fact(self):
        self.assertEqual(
            _extract_claims_spacy("The Earth revolves around the Sun."),
            ["The Earth revolves around the Sun."],
        )

    @patch("claim_extractor._extract_claims_llm", return_value=[])
    @patch("claim_extractor._ollama_available", return_value=True)
    def test_empty_llm_extraction_falls_back_to_heuristic(self, _available, _llm):
        self.assertEqual(
            extract_claims("The Earth revolves around the Sun."),
            ["The Earth revolves around the Sun."],
        )

    def test_official_query_for_current_indian_prime_minister_claim(self):
        self.assertEqual(
            official_office_query("Narendra Modi is the prime minister of India."),
            'site:pmindia.gov.in "Narendra Modi" "Prime Minister"',
        )

    def test_official_query_for_dated_tamil_nadu_chief_minister_claim(self):
        self.assertEqual(
            official_office_query(
                "As of October 12, 2026, Thiru C. Joseph Vijay is the Chief Minister of Tamil Nadu."
            ),
            'site:lokbhavan.tn.gov.in "Joseph Vijay" "Chief Minister of Tamil Nadu"',
        )

    def test_official_query_ignores_unrelated_claim(self):
        self.assertIsNone(official_office_query("Water boils at 100 degrees Celsius."))

    def test_official_query_targets_long_ruling_claim(self):
        self.assertEqual(
            official_office_query("Narendra Modi has ruled India for 30 years."),
            'site:pmindia.gov.in "Narendra Modi" "head of government" 2001',
        )

    def test_generic_ruling_duration_uses_dated_government_start(self):
        claim = "Narendra Modi has ruled India for 30 years."
        evidence = (
            "Narendra Modi enters the 25th year of serving as head of a government. "
            "He took oath as Chief Minister of Gujarat on this day in 2001."
        )
        conflict = _tenure_duration_conflict(claim, evidence, as_of_year=2026)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["start_year"], 2001)
        self.assertTrue(is_tenure_duration_claim(claim))

    @patch("fact_checker.requests.get")
    def test_page_fetch_includes_government_table_text(self, get):
        response = type("Response", (), {
            "text": "<html><body><table><tr><td>Narendra Modi</td><td>Prime Minister</td></tr></table></body></html>",
            "raise_for_status": lambda self: None,
        })()
        get.return_value = response
        text = fetch_page_text("https://www.pmindia.gov.in/en/portfolios")
        self.assertIn("Narendra Modi", text)
        self.assertIn("Prime Minister", text)

    def test_valid_agent_json(self):
        self.assertEqual(_parse_final_answer('{"verdict":"True","summary":"Supported.","sources":[]}')["verdict"], "True")

    def test_fenced_json(self):
        self.assertEqual(_parse_final_answer('```json\n{"verdict":"False","summary":"No."}\n```')["verdict"], "False")

    def test_invalid_json(self):
        self.assertIsNone(_parse_final_answer("not json"))

    def test_invalid_verdict(self):
        self.assertIsNone(_parse_final_answer('{"verdict":"Maybe","summary":"x"}'))

    def test_missing_summary(self):
        self.assertIsNone(_parse_final_answer('{"verdict":"True"}'))

    def test_source_filtering(self):
        value = _parse_final_answer('{"verdict":"True","summary":"x","sources":[{"url":"https://a.test"},{}]}')
        self.assertEqual(len(value["sources"]), 1)

    def test_verdict_support_only(self):
        self.assertEqual(_map_verdict(2, 0), "True")

    def test_verdict_refute_only(self):
        self.assertEqual(_map_verdict(0, 1), "False")

    def test_verdict_conflict_is_misleading(self):
        self.assertEqual(_map_verdict(1, 1), "Misleading")

    def test_verdict_mixed_evidence_is_misleading_even_when_one_side_has_more_sources(self):
        self.assertEqual(_map_verdict(3, 1), "Misleading")

    def test_verdict_no_decisive_evidence(self):
        self.assertEqual(_map_verdict(0, 0), "Unverified")

    def test_twenty_year_prime_minister_claim_conflicts_with_2014_start(self):
        conflict = _tenure_duration_conflict(
            "Narendra Modi has been Prime Minister for 20 years.",
            "Narendra Modi was sworn in as Prime Minister on 26 May 2014.",
            as_of_year=2026,
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["start_year"], 2014)

    def test_tenure_duration_claim_uses_fixed_verdict_path(self):
        self.assertTrue(is_tenure_duration_claim("Modi has been PM for 20 years."))

    def test_false_fallback_summary_compares_claim_and_dated_fact(self):
        summary = _fallback_summary(
            "False", 0, 1,
            "Modi has been Prime Minister for 20 years",
            [{"stance": "contradiction", "excerpt": "Modi was sworn in as Prime Minister on 26 May 2014."}],
        )
        self.assertIn("20 years", summary)
        self.assertIn("2014", summary)

    @patch("fact_checker.generate_summary")
    @patch("fact_checker.score_source_against_claim")
    @patch("fact_checker.fetch_page_text", return_value="")
    def test_golden_modi_claim_is_false_even_if_nli_says_support(
        self, _fetch, score, make_summary
    ):
        score.return_value = {
            "label": "entailment", "score": 0.99,
            "chunk": "Generic biography text about Narendra Modi.",
        }
        make_summary.side_effect = lambda claim, verdict, evidence, **kwargs: _fallback_summary(
            verdict, 0, 1, claim, evidence
        )
        result = verify_claim(
            "Narendra Modi has been Prime Minister for 20 years.",
            [{
                "title": "Official appointment record",
                "url": "https://example.test/appointment",
                "snippet": "Narendra Modi was sworn in as Prime Minister on 26 May 2014.",
            }],
            tokenizer=None,
            model=None,
        )
        self.assertEqual(result["verdict"], "False")
        self.assertIn("2014", result["summary"])

    @patch("fact_checker.generate_summary", return_value="The official timeline starts in 2001, so 30 years is too long.")
    @patch("fact_checker.score_source_against_claim")
    @patch("fact_checker.fetch_page_text", return_value="")
    def test_golden_thirty_year_ruling_claim_is_false(
        self, _fetch, score, _summary
    ):
        score.return_value = {
            "label": "neutral", "score": 0.4,
            "chunk": "Official timeline excerpt.",
        }
        checked = verify_claim(
            "Narendra Modi has ruled India for 30 years.",
            [{
                "title": "Official PM India timeline",
                "url": "https://www.pmindia.gov.in/en/news_updates/official-timeline/",
                "snippet": (
                    "Narendra Modi entered his 25th year as head of a government. "
                    "He took oath as Chief Minister of Gujarat in 2001."
                ),
            }],
            tokenizer=None,
            model=None,
        )
        self.assertEqual(checked["verdict"], "False")

    def test_tenure_guard_needs_matching_office_and_dated_evidence(self):
        self.assertIsNone(_tenure_duration_conflict(
            "The Prime Minister has served for 20 years.",
            "Narendra Modi won the election in 2014.",
            as_of_year=2026,
        ))

    def test_current_official_listing_supports_present_office_claim(self):
        current_year = date.today().year
        self.assertTrue(_current_office_support(
            "Narendra Modi is the Prime Minister of India.",
            "https://www.pmindia.gov.in/en/portfolios",
            f"Portfolios as on 25.07.{current_year}: Narendra Modi | Prime Minister",
            as_of_year=current_year,
        ))

    @patch("fact_checker.generate_summary", return_value="Official listing confirms the current role.")
    @patch("fact_checker.score_source_against_claim")
    @patch("fact_checker.fetch_page_text", return_value="")
    def test_current_official_listing_resolves_stale_conflict(
        self, _fetch, score, _summary
    ):
        score.return_value = {
            "label": "contradiction", "score": 0.95,
            "chunk": "A stale conflicting excerpt.",
        }
        current_year = date.today().year
        results = [
            {
                "title": "Current Union Council portfolios",
                "url": "https://www.pmindia.gov.in/en/portfolios",
                "snippet": f"Portfolios as on 25.07.{current_year}: Narendra Modi | Prime Minister",
            },
            {
                "title": "Old article",
                "url": "https://news.example.com/old",
                "snippet": "An older article with a contradictory statement.",
            },
        ]
        checked = verify_claim(
            "Narendra Modi is the Prime Minister of India.",
            results, tokenizer=None, model=None,
        )
        self.assertEqual(checked["verdict"], "True")

    def test_strength_thresholds(self):
        self.assertEqual(_evidence_strength(0, 0), "Limited evidence")
        self.assertEqual(_evidence_strength(3, 0), "Strong evidence")

    def test_timing_helpers(self):
        result = {}
        merge_timings(result, {"search": 0.2})
        merge_timings(result, {"search": 0.3}, "agent_")
        self.assertAlmostEqual(total_seconds(result), 0.5)


if __name__ == "__main__":
    unittest.main()
