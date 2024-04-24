
target_path := "src/vllm_tgis_adapter/grpc/pb"
gen-protos:
	# Compile protos
	pip install grpcio-tools==1.62.1 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4'
	mkdir -p $(target_path)
	python -m grpc_tools.protoc -Iproto --python_out=$(target_path) \
		--grpc_python_out=$(target_path) --mypy_out=$(target_path) proto/generation.proto
	find $(target_path)/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch $(target_path)/__init__.py
