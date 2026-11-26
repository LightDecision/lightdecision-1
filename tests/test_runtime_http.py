"""HTTP wire contract without downloading weights or allocating a GPU."""
import importlib.util
import unittest

from lightdecision.api.typesafe import InvalidDecisionRequest, format_answer, parse_request
from lightdecision.runtime.server import create_app

HAS_HTTP = all(importlib.util.find_spec(x) for x in ("fastapi", "httpx"))


class FixtureEngine:
    alias = "lightdecision-1"
    model_id = "lightdecision-fixture"

    def __init__(self):
        self.calls = []

    def predict(self, body):
        try:
            questions = parse_request(body, model_ids={self.alias, self.model_id})
        except (ValueError, TypeError) as exc:
            raise InvalidDecisionRequest(str(exc)) from exc
        self.calls.append(body)
        answers = {}
        for q in questions:
            count = len(q.decision.options)
            ps = [.6, .3, .1] if count == 3 else [1/count] * count
            answers[q.decision.id] = format_answer(q, ps)
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": 123, "output_tokens": 0, "executed_input_tokens": 234},
                "runtime": {"execution": "shared_prefix"}}


@unittest.skipUnless(HAS_HTTP, "requires the existing serving environment")
class RuntimeHTTPTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.engine = FixtureEngine()
        self.client = self.enterContext(TestClient(create_app(lambda: self.engine, api_key="test-key")))
        self.headers = {"Authorization": "Bearer test-key"}
        self.body = {"model": self.engine.alias, "state": {"ticket": ["duplicate", "charge"]}, "questions": {
            "route": {"type": "choice", "instructions": "Route", "criteria": {"billing": None, "sales": "Sales", "other": ["Fallback"]}},
            "score": {"type": "score", "instructions": {"task": "Rate"}, "criteria": ["Low", "Medium", "High"]},
            "noul": {"type": "noul", "instructions": "Payment?"}}}

    def test_official_envelope_and_all_primitives(self):
        response = self.client.post("/v1/systemone", json=self.body, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(set(result), {"model", "answers", "usage"})
        self.assertEqual(set(result["usage"]), {"input_tokens", "output_tokens"})
        self.assertEqual(set(result["answers"]), set(self.body["questions"]))
        self.assertAlmostEqual(result["answers"]["route"]["confidence"], .4)
        self.assertAlmostEqual(result["answers"]["score"]["confidence"], .25)
        self.assertEqual(result["answers"]["noul"], {"type": "noul", "noul": .5})

    def test_authorization_and_bad_requests_never_reach_inference(self):
        for auth in (None, "Basic test-key", "Bearer wrong"):
            headers = {} if auth is None else {"Authorization": auth}
            response = self.client.post("/v1/systemone", json=self.body, headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers["www-authenticate"], "Bearer")
        for body in ([], {}, {**self.body, "model": []}, {**self.body, "images": "bad"}, {**self.body, "questions": {}}):
            self.assertEqual(self.client.post("/v1/systemone", json=body, headers=self.headers).status_code, 422)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers=self.headers).status_code, 200)

    def test_1024_choices_512_score_levels_and_images_pass_through(self):
        body = {**self.body, "images": ["data:image/png;base64,fixture"], "questions": {
            "choice": {"type": "choice", "instructions": "Choose", "criteria": {f"option {i}": None for i in range(1024)}},
            "score": {"type": "score", "instructions": "Rate", "criteria": [f"level {i}" for i in range(512)]}}}
        response = self.client.post("/v1/systemone", json=body, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answers"]
        self.assertEqual(len(answer["choice"]["probabilities"]), 1024)
        self.assertEqual(len(answer["score"]["probabilities"]), 512)
        self.assertEqual(answer["score"]["score"], 255.5)
        self.assertEqual(self.engine.calls[-1]["images"], body["images"])

    def test_local_mode_and_opt_in_runtime_extension(self):
        from fastapi.testclient import TestClient
        with TestClient(create_app(lambda: self.engine, include_runtime_metrics=True)) as client:
            result = client.post("/v1/systemone", json=self.body)
            self.assertEqual(result.status_code, 200)
            self.assertIn("runtime", result.json())
            self.assertIn("executed_input_tokens", result.json()["usage"])
