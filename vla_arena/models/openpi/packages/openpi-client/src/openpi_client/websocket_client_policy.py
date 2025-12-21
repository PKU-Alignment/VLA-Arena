# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing_extensions import override

import websockets.sync.client
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        if host.startswith('ws'):
            self._uri = host
        else:
            self._uri = f'ws://{host}'
        if port is not None:
            self._uri += f':{port}'
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f'Waiting for server at {self._uri}...')
        while True:
            try:
                headers = (
                    {'Authorization': f'Api-Key {self._api_key}'}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info('Still waiting for server...')
                time.sleep(5)

    @override
    def infer(self, obs: dict) -> dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f'Error in inference server:\n{response}')
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
