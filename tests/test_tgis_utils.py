import sys

import pytest
from vllm.utils import FlexibleArgumentParser

from vllm_tgis_adapter.tgis_utils.args import EnvVarArgumentParser


class TestEnvVarArgumentParser:
    @pytest.fixture(autouse=True)
    def _override_sys_argv(self, monkeypatch):
        # Required to avoid parsing pytest's commandline
        monkeypatch.setattr(sys, "argv", ["vllm_tgis_adapter"])

    def test_str_flag(self, monkeypatch, _override_sys_argv):
        expected_value = "custom-value"
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_FLAG", expected_value)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-flag", type=str, default="default_value")
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_flag == expected_value

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("True", True),
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
        ],
    )
    def test_bool_flag(self, monkeypatch, _override_sys_argv, env_var, expected):
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_BOOL_FLAG", env_var)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-bool-flag", type=bool)
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_bool_flag == expected

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("1", 1),
            ("42", 42),
            ("-1", -1),
        ],
    )
    def test_int_flag(self, monkeypatch, _override_sys_argv, env_var, expected):
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_INT_FLAG", env_var)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-int-flag", type=int)
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_int_flag == expected
