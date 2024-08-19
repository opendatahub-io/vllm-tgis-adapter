import sys

import pytest
from vllm.engine.arg_utils import StoreBoolean
from vllm.utils import FlexibleArgumentParser

from vllm_tgis_adapter.tgis_utils.args import (
    EnvVarArgumentParser,
    _bool_from_string,
)


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
    def test_bool_from_string_flag(
        self, monkeypatch, _override_sys_argv, env_var, expected
    ):
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_BOOL_FROM_STRING_FLAG", env_var)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-bool-from-string-flag", type=_bool_from_string)
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_bool_from_string_flag == expected

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
    def test_store_true_flag(self, monkeypatch, _override_sys_argv, env_var, expected):
        """VLLM has 'store_true' boolean args that we want to handle."""
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_STORE_TRUE_FLAG", env_var)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-store-true-flag", action="store_true")
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_store_true_flag == expected

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
    def test_store_false_flag(self, monkeypatch, _override_sys_argv, env_var, expected):
        """VLLM doesn't have 'store_false' boolean args yet..."""
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_STORE_FALSE_FLAG", env_var)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-store-false-flag", action="store_false")
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_store_false_flag == expected

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
    def test_store_boolean_flag(
        self, monkeypatch, _override_sys_argv, env_var, expected
    ):
        """VLLM has a custom StoreBoolean action for --enable-chunked-prefill."""
        with monkeypatch.context():
            monkeypatch.setenv("DUMMY_STORE_BOOLEAN_FLAG", env_var)

            parser = FlexibleArgumentParser("testing parser")
            parser.add_argument("--dummy-store_boolean-flag", action=StoreBoolean)
            parser = EnvVarArgumentParser(parser=parser)
            args = parser.parse_args()
            assert args.dummy_store_boolean_flag == expected

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
