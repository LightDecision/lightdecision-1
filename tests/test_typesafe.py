import unittest

from lightdecision.api.typesafe import format_answer, parse_request
from lightdecision.models.decision.serializer import DecisionSerializer


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def request(questions, **kwargs):
    return {"model": "lightdecision-1", "state": "x", "questions": questions, **kwargs}


def parse(body):
    return parse_request(body, model_ids={"lightdecision-1"})


class TypeSafeMappingTests(unittest.TestCase):
    def test_three_primitives_and_structured_fields(self):
        body = request({
            "route": {"type": "choice", "instructions": ["Route"], "criteria": {"billing": {"handles": "refunds"}, "other": None}},
            "refund": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "Asked for refund", "false": "No request"}},
            "urgency": {"type": "score", "instructions": "Rate urgency", "criteria": ["Can wait", {"impact": "Blocking"}]},
        }, state={"message": "duplicate charge"})
        questions = parse(body)
        self.assertEqual(format_answer(questions[0], [.8, .2])["choice"], "billing")
        self.assertEqual(format_answer(questions[1], [.3, .7]), {"type": "noul", "noul": .7})
        self.assertEqual(format_answer(questions[2], [.25, .75])["score"], .75)
        self.assertEqual(format_answer(questions[2], [.25, .75])["legend"]["1"], {"impact": "Blocking"})

    def test_question_ids_and_score_indices_do_not_enter_text(self):
        base = {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]}
        a = parse(request({"sensitive-id": base}))[0]
        b = parse(request({"different-id": base}))[0]
        serializer = DecisionSerializer(Tokenizer())
        self.assertEqual(serializer(a.decision).branches, serializer(b.decision).branches)
        self.assertEqual([item.name for item in a.decision.options], ["Low", "High"])
        self.assertNotIn("sensitive-id", bytes(serializer(a.decision).branches[0].input_ids).decode())

    def test_partial_or_nonfinite_probabilities_are_rejected(self):
        q = parse(request({"q": {"type": "noul", "instructions": "true?"}}))[0]
        for p in ([1.0], [.9, .9], [float("nan"), 0], [-.1, 1.1]):
            with self.subTest(p=p), self.assertRaises(RuntimeError):
                format_answer(q, p)

    def test_single_level_degenerate_extension(self):
        for primitive, criteria in (("score", ["Only level"]), ("choice", {"Only level": None})):
            q = parse(request({"q": {"type": primitive, "instructions": "Apply", "criteria": criteria}}))[0]
            answer = format_answer(q, [1.0])
            self.assertEqual(answer["confidence"], 1.0)
            self.assertEqual(answer["score"] if primitive == "score" else answer["choice"], 0.0 if primitive == "score" else "Only level")

    def test_official_confidence_fixtures(self):
        fixtures = [
            ([.6, .3, .1], .4, .25),
            ([.6, .2, .2], .4, .1),
            ([0, .5, .5], .25, .25),
            ([.5, 0, .5], .25, 0),
            ([0, .57, .43], .355, .355),
            ([1/3, 1/3, 1/3], 0, 0),
            ([0, 1, 0], 1, 1),
        ]
        for ps, choice_confidence, score_confidence in fixtures:
            for primitive, expected in (("choice", choice_confidence), ("score", score_confidence)):
                with self.subTest(ps=ps, primitive=primitive):
                    levels = ["low", "middle", "high"]
                    criteria = dict.fromkeys(levels) if primitive == "choice" else levels
                    q = parse(request({"q": {"type": primitive, "instructions": "Rate", "criteria": criteria}}))[0]
                    answer = format_answer(q, ps)
                    self.assertAlmostEqual(answer["confidence"], expected)
                    self.assertEqual(list(answer["probabilities"].values()), ps)

    def test_large_choice_and_score_return_every_option_in_order(self):
        for count in (11, 255, 256, 1024):
            for primitive in ("choice", "score"):
                with self.subTest(count=count, primitive=primitive):
                    levels = [f"自定义动作 {i}" for i in range(count)]
                    criteria = dict.fromkeys(levels) if primitive == "choice" else levels
                    q = parse(request({"q": {"type": primitive, "instructions": "Decide", "criteria": criteria}}))[0]
                    ps = [0.] * count
                    ps[-1] = 1.
                    answer = format_answer(q, ps)
                    self.assertEqual(len(answer["probabilities"]), count)
                    self.assertEqual(answer["confidence"], 1)
                    if primitive == "choice":
                        self.assertEqual(list(answer["probabilities"]), levels)
                        self.assertEqual(answer["choice"], levels[-1])
                    else:
                        self.assertEqual(list(answer["probabilities"]), list(map(str, range(count))))
                        self.assertEqual(answer["score"], count - 1)
                        self.assertEqual(answer["legend"][str(count-1)], levels[-1])

    def test_model_is_required_and_validated_before_lookup(self):
        body = request({"q": {"type": "noul", "instructions": "Yes?"}})
        missing = dict(body)
        missing.pop("model")
        for invalid in (missing, *({**body, "model": value} for value in (None, "", [], {}, 1, "wrong"))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse(invalid)

    def test_images_extension_is_explicit_and_validated(self):
        body = request({"q": {"type": "noul", "instructions": "Visible?"}})
        for values in ([], ["/server/scene.png"], ["data:image/png;base64,abc"]):
            self.assertEqual(len(parse({**body, "images": values})), 1)
        for value in (None, "image.png", {}, [None], [""], [{"image_url": "image.png"}]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "images"):
                parse({**body, "images": value})

    def test_bad_criteria_rejected(self):
        for question in ({"type": "choice", "instructions": "q", "criteria": []},
                         {"type": "choice", "instructions": "q", "criteria": {}},
                         {"type": "score", "instructions": "q", "criteria": []},
                         {"type": "score", "instructions": "q", "criteria": [None]},
                         {"type": "noul", "instructions": "q", "criteria": {"maybe": "x"}}):
            with self.subTest(question=question), self.assertRaises(ValueError):
                parse(request({"q": question}))
