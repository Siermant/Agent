FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GTA_DATA=/data GTA_BIND=0.0.0.0
WORKDIR /app
RUN groupadd -g 10001 gta && useradd -u 10001 -g gta -M gta && mkdir /data && chown gta:gta /data
COPY global_agent/ /app/global_agent/
USER 10001:10001
EXPOSE 8080
CMD ["python", "-m", "global_agent"]
