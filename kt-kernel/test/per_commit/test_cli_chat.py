import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0.1, suite="default")

from kt_kernel.cli.commands import chat as chat_cmd

PROXY_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]


def _fake_client():
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="served-model")]
    response = MagicMock()
    response.choices[0].message.content = "hello"
    response.usage = None
    client.chat.completions.create.return_value = response
    return client


class TestChatCommand(unittest.TestCase):
    def test_chat_starts_when_tokenizer_cannot_be_loaded(self):
        client = _fake_client()
        inputs = iter(["hi", "/quit"])
        env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
        env["KT_LANG"] = "en"

        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"transformers": None}),
            patch.object(chat_cmd, "OpenAI", return_value=client, create=True),
            patch.object(chat_cmd, "HAS_OPENAI", True),
            patch.object(chat_cmd, "get_settings"),
            patch.object(chat_cmd.console, "input", side_effect=lambda *args, **kwargs: next(inputs)),
        ):
            chat_cmd.chat(
                host="127.0.0.1",
                port=30000,
                model=None,
                temperature=0.7,
                max_tokens=16,
                system_prompt=None,
                save_history=False,
                history_file=None,
                stream=False,
            )

        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "served-model")
        self.assertEqual(request["messages"][0], {"role": "user", "content": "hi"})


if __name__ == "__main__":
    unittest.main()
