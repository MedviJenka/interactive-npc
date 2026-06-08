ARG version=3.12-slim
FROM app AS build
RUN pip install uv
RUN uv sync
COPY . .

FROM build AS npc_service
CMD ["uviconr", "uv", "run", "main.py"]
