from __future__ import annotations

import argparse
import os

from vllm_tgis_adapter.grpc.validation import MAX_TOP_N_TOKENS
from vllm_tgis_adapter.logging import init_logger

logger = init_logger(__name__)


def _to_env_var(arg_name: str) -> str:
    return arg_name.upper().replace("-", "_")


def _switch_action_default(action: argparse.Action) -> None:
    """Switch to using env var fallback for all args."""
    val = os.environ.get(_to_env_var(action.dest))
    if val:
        if action.type == bool:
            val = val.lower() == "true" or val == "1"
        elif action.type == int:
            val = int(val)
        action.default = val


class EnvVarArgumentParser(argparse.ArgumentParser):
    """Allows env var fallback for all args."""

    class _EnvVarHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def _get_help_string(self, action: argparse.Action) -> str:
            help_ = super()._get_help_string(action)
            assert help_ is not None

            if action.dest != "help":
                help_ += f" [env: {_to_env_var(action.dest)}]"
            return help_

    def __init__(
        self,
        parser: argparse.ArgumentParser | None = None,
        *,
        formatter_class: type[
            argparse.ArgumentDefaultsHelpFormatter
        ] = _EnvVarHelpFormatter,
        **kwargs,  # noqa: ANN003
    ):
        parents = []
        if parser:
            parents.append(parser)
            for action in parser._actions:  # noqa: SLF001
                if type(action) == argparse._HelpAction:  # noqa: SLF001
                    continue
                _switch_action_default(action)
        super().__init__(
            formatter_class=formatter_class, parents=parents, add_help=False, **kwargs
        )

    def _add_action(self, action: argparse.Action) -> argparse.Action:
        _switch_action_default(action)
        return super()._add_action(action)


def add_tgis_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # map to model
    parser.add_argument(
        "--model-name",
        type=str,
        help="name or path of the huggingface model to use",
    )
    # map to max_model_len
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        help="model context length. If unspecified, "
        "will be automatically derived from the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="maximum allowed new (generated) tokens per request",
    )
    # map to max_num_seqs ... TBD
    parser.add_argument("--max-batch-size", type=int)
    # map to dtype
    parser.add_argument("--dtype-str", type=str, help="deprecated, use dtype")
    # map to quantization
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["awq", "gptq", "squeezellm", None],
        help="Method used to quantize the weights. If "
        "None, we first check the `quantization_config` "
        "attribute in the model config file. If that is "
        "None, we assume the model weights are not "
        "quantized and use `dtype` to determine the data "
        "type of the weights.",
    )
    # map to tensor_parallel_size
    parser.add_argument("--num-gpus", type=int)
    # map to tensor_parallel_size
    parser.add_argument("--num-shard", type=int)
    # TODO check boolean behaviour for env vars and defaults
    parser.add_argument("--output-special-tokens", type=bool, default=False)
    parser.add_argument(
        "--default-include-stop-seqs", type=bool, default=True
    )  # TODO TBD
    parser.add_argument("--grpc-port", type=int, default=8033)

    # map to ssl_certfile
    parser.add_argument("--tls-cert-path", type=str)
    # map to ssl_keyfile
    parser.add_argument("--tls-key-path", type=str)
    # map to ssl_ca_certs
    parser.add_argument("--tls-client-ca-cert-path", type=str)
    # add a path when peft adapters will be loaded from
    parser.add_argument("--adapter-cache", type=str)

    # TODO check/add other args here

    # revision, dtype, trust-remote-code already covered by llmengine args
    return parser


def postprocess_tgis_args(args: argparse.Namespace) -> argparse.Namespace:  # noqa: C901,PLR0912
    if args.model_name:
        args.model = args.model_name
    if args.max_sequence_length is not None:
        if args.max_model_len not in (None, args.max_sequence_length):
            raise ValueError(
                "Inconsistent max_model_len and max_sequence_length arg values"
            )
        args.max_model_len = args.max_sequence_length
    if args.dtype_str is not None:
        if args.dtype not in (None, "auto", args.dtype_str):
            raise ValueError("Inconsistent dtype and dtype_str arg values")
        args.dtype = args.dtype_str
    if args.quantize:
        if args.quantization and args.quantization != args.quantize:
            raise ValueError("Inconsistent quantize and quantization arg values")
        args.quantization = args.quantize
    if args.num_gpus is not None or args.num_shard is not None:
        if (
            args.num_gpus is not None
            and args.num_shard is not None
            and args.num_gpus != args.num_shard
        ):
            raise ValueError("Inconsistent num_gpus and num_shard arg values")
        num_gpus = args.num_gpus if args.num_gpus is not None else args.num_shard
        if args.tensor_parallel_size not in [None, 1, num_gpus]:
            raise ValueError(
                "Inconsistent tensor_parallel_size and num_gpus/num_shard arg values"
            )
        args.tensor_parallel_size = num_gpus
    if args.max_logprobs < MAX_TOP_N_TOKENS + 1:
        logger.info("Setting max_logprobs to %d", MAX_TOP_N_TOKENS + 1)
        args.max_logprobs = MAX_TOP_N_TOKENS + 1
    # Turn off vLLM per-request logging because the TGIS server logs each
    # response
    if not args.disable_log_requests:
        args.disable_log_requests = True

    if args.max_batch_size is not None:
        # Existing MAX_BATCH_SIZE settings in TGIS configs may not necessarily
        # be best for vLLM so we'll just log a warning for now
        logger.warning(
            "max_batch_size is set to %d but will be ignored for now."
            "max_num_seqs can be used if this is still needed.",
            args.max_batch_size,
        )

    if args.tls_cert_path:
        args.ssl_certfile = args.tls_cert_path
    if args.tls_key_path:
        args.ssl_keyfile = args.tls_key_path
    if args.tls_client_ca_cert_path:
        args.ssl_ca_certs = args.tls_client_ca_cert_path

    return args
