from __future__ import annotations

try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:  # vllm <0.11.0
    from vllm.sampling_params import (
        GuidedDecodingParams as StructuredOutputsParams,  # hack: API is identical for our purposes # noqa: E501
    )


from vllm_tgis_adapter.grpc.pb.generation_pb2 import DecodingParameters


def get_structured_output_params(
    decoding_params: DecodingParameters,
) -> StructuredOutputsParams | None:
    if not (guided := decoding_params.WhichOneof("guided")):
        return None

    if guided == "json_schema":
        return StructuredOutputsParams(json=decoding_params.json_schema)

    if guided == "regex":
        return StructuredOutputsParams(regex=decoding_params.regex)

    if guided == "choice":
        choice_list = decoding_params.choice.choices
        if len(choice_list) < 2:
            raise ValueError("Must provide at least two choices")
        return StructuredOutputsParams(choice=list(choice_list))

    if guided == "grammar":
        return StructuredOutputsParams(grammar=decoding_params.grammar)

    if decoding_params.format == DecodingParameters.JSON:
        return StructuredOutputsParams(json_object=True)

    raise ValueError(guided)
