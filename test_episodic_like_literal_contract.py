"""Contrato da busca episódica: curinga LIKE não é consulta literal, e entrada inválida não imita sucesso.

Hipóteses (código de main 616a9b2, antes da correção):
- H1: FTS já interpreta '%' e '_' como coringa. Falsifica se a consulta não tem termo \\w
  (ou o FTS de '_' devolve vazio) e mesmo assim o resultado é o conjunto inteiro.
- H2: o fallback LIKE monta f"%{query}%" sem ESCAPE, então '%' e '_' são curingas.
  Falsifica se só títulos/gatilhos com o caractere literal entram no resultado.
- H3: string vazia e não-string falso (False, 0, ()) já são rejeitados. Falsifica se
  devolvem a mesma lista da ausência documentada (None), que é sucesso.
- H4: None deve continuar sendo a ausência documentada (EM1 e chamadores). Falsifica
  se search_episodes(None) deixar de listar por recência.

A decisão dos casos inválidos no subprocesso é o código de saída: 2 = entrada inválida,
0 = sucesso. Não há pipe no caminho dessa decisão.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from core.cognitivo.episodic_memory import RealEpisodicMemory

ROOT = Path(__file__).resolve().parent


class EpisodicLikeLiteralContractTests(unittest.TestCase):
    def setUp(self):
        self.memory = RealEpisodicMemory(":memory:")
        self.addCleanup(self.memory.close)
        self._closed("100%")
        self._closed("plain")
        self._closed("a_b")
        self._closed("axb")
        self._closed("rate_limit")
        self._closed("alpha release")
        self._closed("note", trigger_query="save 100%")
        self._open = self.memory.start_episode("100% open")

    def _closed(self, title, trigger_query=None):
        episode_id = self.memory.start_episode(title, trigger_query=trigger_query)
        self.memory.close_episode(episode_id)
        return episode_id

    def _titles(self, query, **kwargs):
        return [episode.title for episode in self.memory.search_episodes(query, **kwargs)]

    def test_percent_matches_only_literal_percent(self):
        self.assertCountEqual(self._titles("%"), ["100%", "note"])

    def test_underscore_matches_only_literal_underscore(self):
        self.assertCountEqual(self._titles("_"), ["a_b", "rate_limit"])

    def test_doubled_percent_is_not_a_match_all(self):
        self.assertEqual(self._titles("%%"), [])

    def test_open_episode_stays_out_of_the_default_status(self):
        self.assertNotIn("100% open", self._titles("%"))
        self.assertEqual(self._titles("%", status="open"), ["100% open"])

    def test_limit_still_bounds_literal_matches(self):
        self.assertEqual(len(self._titles("%", limit=1)), 1)

    def test_ordinary_substring_and_backslash_still_resolve(self):
        self._closed("a\\b")
        self.assertEqual(self._titles("alp"), ["alpha release"])
        self.assertIn("a\\b", self._titles("a\\b"))

    def test_absent_none_remains_recency_listing(self):
        self.assertEqual(len(self._titles(None)), 7)
        self.assertEqual(len(self.memory.search_episodes()), 7)

    def test_empty_string_is_invalid_before_sql(self):
        statements = []
        self.memory.conn.set_trace_callback(statements.append)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.memory.search_episodes("")
        self.memory.conn.set_trace_callback(None)
        self.assertEqual(statements, [])

    def test_non_string_query_is_invalid_before_sql(self):
        statements = []
        self.memory.conn.set_trace_callback(statements.append)
        for query in (False, True, 0, 1, 1.5, b"%", bytearray(b"a"), ()):
            with self.subTest(query=query):
                with self.assertRaisesRegex(TypeError, "query must be a string"):
                    self.memory.search_episodes(query)
        self.memory.conn.set_trace_callback(None)
        self.assertEqual(statements, [])

    def test_invalid_input_exits_2_and_literal_success_exits_0(self):
        script = r"""
import sys
from core.cognitivo.episodic_memory import RealEpisodicMemory

kind = sys.argv[1]
with RealEpisodicMemory(":memory:") as mem:
    for title, trigger in (("100%", None), ("plain", None), ("note", "save 100%")):
        episode_id = mem.start_episode(title, trigger_query=trigger)
        mem.close_episode(episode_id)
    try:
        if kind == "empty":
            mem.search_episodes("")
        elif kind == "false":
            mem.search_episodes(False)
        elif kind == "bytes":
            mem.search_episodes(b"%")
        elif kind == "percent":
            titles = [episode.title for episode in mem.search_episodes("%")]
            sys.exit(0 if sorted(titles) == ["100%", "note"] else 1)
        else:
            sys.exit(9)
    except ValueError as exc:
        sys.exit(2 if "non-empty" in str(exc) else 4)
    except TypeError as exc:
        sys.exit(2 if "query must be a string" in str(exc) else 4)
    else:
        sys.exit(0)
"""
        expected = {"empty": 2, "false": 2, "bytes": 2, "percent": 0}
        for kind, code in expected.items():
            with self.subTest(kind=kind):
                completed = subprocess.run(
                    [sys.executable, "-c", script, kind],
                    cwd=ROOT,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONPATH": str(ROOT),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    check=False,
                )
                self.assertEqual(completed.returncode, code)


if __name__ == "__main__":
    unittest.main()
