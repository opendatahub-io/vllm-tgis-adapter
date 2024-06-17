import typing
from enum import Enum

from vllm import SamplingParams

from .pb.generation_pb2 import DecodingMethod, Parameters

MAX_TOP_N_TOKENS = 10

MAX_STOP_SEQS = 6
MAX_STOP_SEQ_LENGTH = 240

# Whether to reject requests if sampling parameters are provided
# in greedy mode, or to silently ignore them
STRICT_PARAMETER_VALIDATION = False


class TGISValidationError(str, Enum):
    """Holds all TGIS parameter validation failure cases.

    See the equivalent enumeration in TGIS here:
    https://github.ibm.com/ai-foundation/fmaas-inference-server/blob/main/router/src/validation.rs#L238-L271
    """

    Temperature = "temperature must be >= 0.05"
    TopP = "top_p must be > 0.0 and <= 1.0"
    TopK = "top_k must be strictly positive"
    TypicalP = "typical_p must be <= 1.0"
    RepetitionPenalty = "repetition_penalty must be > 0.0 and <= 2.0"
    LengthPenalty = "length_penalty.decay_factor must be >= 1.0 and <= 10.0"
    MaxNewTokens = "max_new_tokens must be <= {0}"
    MinNewTokens = "min_new_tokens must be <= max_new_tokens"
    InputLength = (
        "input tokens ({0}) plus prefix length ({1}) plus "
        "min_new_tokens ({2}) must be <= {3}"
    )
    InputLength2 = "input tokens ({0}) plus prefix length ({1}) must be < {2}"
    Tokenizer = "tokenizer error {0}"
    StopSequences = (
        "can specify at most {0} non-empty stop sequences, each "
        "not more than {1} UTF8 bytes"
    )
    TokenDetail = (
        "must request input and/or generated tokens to request extra token detail"
    )
    PromptPrefix = "can't retrieve prompt prefix with id '{0}': {1}"
    SampleParametersGreedy = (
        "sampling parameters aren't applicable in " "greedy decoding mode"
    )

    # Additions that are _not_ in TGIS
    TopN = "top_n_tokens ({0}) must be <= {1}"
    AdapterNotFound = "can't retrieve adapter with id '{0}': {1}"
    AdaptersDisabled = "adapter_id supplied but no adapter store was configured"
    AdapterUnsupported = "adapter type {0} is not currently supported"
    InvalidAdapterID = (
        "Invalid adapter id '{0}', must contain only " "alphanumeric, _ and - and /"
    )

    def error(self, *args, **kwargs) -> typing.NoReturn:  # noqa: ANN002,ANN003
        """Raise a ValueError with a nicely formatted string."""
        raise ValueError(self.value.format(*args, **kwargs))


def validate_input(
    sampling_params: SamplingParams,
    token_num: int,
    max_model_len: int,
) -> None:
    """Raise ValueError if the input is too long."""
    # TODO: add in the prefix length once soft prompt tuning is supported
    if token_num >= max_model_len:
        TGISValidationError.InputLength2.error(token_num, 0, max_model_len)

    if token_num + sampling_params.min_tokens > max_model_len:
        TGISValidationError.InputLength.error(
            token_num, 0, sampling_params.min_tokens, max_model_len
        )


def validate_params(  # noqa: C901
    params: Parameters,
    max_max_new_tokens: int,
) -> None:
    """Raise ValueError from the TGISValidationError enum if Parameters is invalid."""
    # TODO: split into checks that are covered by vllm.SamplingParams
    # vs. checks that are not

    resp_options = params.response
    sampling = params.sampling
    stopping = params.stopping
    decoding = params.decoding

    # Decoding parameter checks
    if decoding.HasField("length_penalty") and not (
        1.0 <= decoding.length_penalty.decay_factor <= 10.0
    ):
        TGISValidationError.LengthPenalty.error()

    if not (0 <= decoding.repetition_penalty <= 2):
        # (a value of 0 means no penalty / unset)
        TGISValidationError.RepetitionPenalty.error()

    # Stopping parameter checks
    if stopping.max_new_tokens > max_max_new_tokens:
        TGISValidationError.MaxNewTokens.error(max_max_new_tokens)

    if stopping.min_new_tokens > (stopping.max_new_tokens or max_max_new_tokens):
        TGISValidationError.MinNewTokens.error()

    if (
        stopping.stop_sequences
        and (len(stopping.stop_sequences) > MAX_STOP_SEQS)
        or not all(0 < len(ss) <= MAX_STOP_SEQ_LENGTH for ss in stopping.stop_sequences)
    ):
        TGISValidationError.StopSequences.error(MAX_STOP_SEQS, MAX_STOP_SEQ_LENGTH)

    # Response options validation
    if resp_options.top_n_tokens > MAX_TOP_N_TOKENS:
        TGISValidationError.TopN.error(resp_options.top_n_tokens, MAX_TOP_N_TOKENS)

    if (
        resp_options.token_logprobs
        or resp_options.token_ranks
        or resp_options.top_n_tokens
    ) and not (resp_options.input_tokens or resp_options.generated_tokens):
        TGISValidationError.TokenDetail.error()

    # Sampling options validation
    greedy = params.method == DecodingMethod.GREEDY
    if (
        STRICT_PARAMETER_VALIDATION
        and greedy
        and (
            sampling.temperature
            or sampling.top_k
            or sampling.top_p
            or sampling.typical_p
        )
    ):
        TGISValidationError.SampleParametersGreedy.error()
    if sampling.temperature and sampling.temperature < 0.05:
        TGISValidationError.Temperature.error()
    if sampling.top_k < 0:
        TGISValidationError.TopK.error()
    if not (0 <= sampling.top_p <= 1):
        TGISValidationError.TopP.error()
    if sampling.typical_p > 1:
        TGISValidationError.TypicalP.error()
