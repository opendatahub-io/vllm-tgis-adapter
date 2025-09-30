from __future__ import annotations

from vllm.sampling_params import GuidedDecodingParams

from vllm_tgis_adapter.grpc.pb.generation_pb2 import DecodingParameters


def get_guided_decoding_params(
    decoding_params: DecodingParameters,
) -> GuidedDecodingParams | None:
    if not (guided := decoding_params.WhichOneof("guided")):
        return None

    if guided == "json_schema":
        return GuidedDecodingParams(json=decoding_params.json_schema)

    if guided == "regex":
        return GuidedDecodingParams(regex=decoding_params.regex)

    if guided == "choice":
        choice_list = decoding_params.choice.choices
        if len(choice_list) < 2:
            raise ValueError("Must provide at least two choices")
        return GuidedDecodingParams(choice=list(choice_list))

    if guided == "grammar":
        return GuidedDecodingParams(grammar=decoding_params.grammar)

    if decoding_params.format == DecodingParameters.JSON:
        return GuidedDecodingParams(json_object=True)

    raise ValueError(f"{guided=}")
