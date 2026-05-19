import time

import docker
import requests


class BenchmarkContainer:
    def __init__(
        self, image_name: str, logger, internal_port: int = 8000, max_retries: int = 10
    ):
        self.client = docker.from_env()
        self.image_name = image_name
        self.logger = logger
        self.internal_port = internal_port
        self.host_port = None
        self.container = None
        self.max_retries = max_retries

    def __enter__(self):
        self.logger.debug(f"Starting Docker container '{self.image_name}'...")
        port_key = f"{self.internal_port}/tcp"
        try:
            self.container = self.client.containers.run(
                self.image_name,
                detach=True,
                ports={port_key: None},
                remove=True,
            )
        except docker.errors.ImageNotFound:
            raise RuntimeError(f"Docker image '{self.image_name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Failed to start container: {e}")

        for _ in range(self.max_retries):
            try:
                self.container.reload()
                if self.container.status == "exited":
                    logs = self.container.logs().decode("utf-8")
                    raise RuntimeError(f"Container exited unexpectedly. Logs:\n{logs}")
                port_info = self.container.attrs["NetworkSettings"]["Ports"]

                if port_info.get(port_key) and len(port_info[port_key]) > 0:
                    self.host_port = int(
                        port_info[f"{self.internal_port}/tcp"][0]["HostPort"]
                    )
                    break
            except docker.errors.NotFound:
                raise RuntimeError("Container was removed unexpectedly.")

            time.sleep(1)
        else:
            try:
                self.container.stop()
            except Exception:
                pass
            raise RuntimeError("Failed to map container port to host port.")

        self.logger.debug(f"Docker container started. Host port: {self.host_port}")
        self._wait_for_server()

        return "localhost", self.host_port

    def __exit__(self, exc_type, exc_value, traceback):
        if self.container:
            self.logger.debug("Stopping Docker container...")
            try:
                self.container.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping container: {e}")

    def _wait_for_server(self, timeout: int = 30):
        url = f"http://localhost:{self.host_port}/docs"
        start = time.time()

        while time.time() - start < timeout:
            try:
                if requests.get(url, timeout=1).status_code == 200:
                    self.logger.debug("Benchmark server is ready.")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)

        self.container.stop()
        raise RuntimeError("Timeout waiting for benchmark server to start.")
