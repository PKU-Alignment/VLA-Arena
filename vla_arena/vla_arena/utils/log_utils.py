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

from termcolor import colored


class VlaArenaColorFormatter(logging.Formatter):
    """This color format is for logging user's project wise information"""

    format_str = '[Project %(levelname)s] '
    debug_message_str = '%(message)s (%(filename)s:%(lineno)d)'
    message_str = '%(message)s'
    FORMATS = {
        logging.DEBUG: format_str + debug_message_str,
        logging.INFO: message_str,
        logging.WARNING: colored(format_str, 'yellow', attrs=['bold'])
        + message_str,
        logging.ERROR: colored(format_str, 'red', attrs=['bold'])
        + message_str,
        logging.CRITICAL: colored(format_str, 'red', attrs=['bold', 'reverse'])
        + message_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class VlaArenaDefaultLogger:
    def __init__(self, logger_config_path, project_name='vla_arena'):
        config = YamlConfig(logger_config_path).as_easydict()
        config['loggers'][project_name] = config['loggers']['project']
        os.makedirs('logs', exist_ok=True)
        logging.config.dictConfig(config)


ProjectDefaultLogger(logger_config_path, project_name)


def get_project_logger(project_name='vla_arena', logger_config_path=None):
    """This function returns a logger that follows the deoxys convention"""
    logger = logging.getLogger(project_name)
    return logger
