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

"""
This is a script for creating various files frrom templates. This is to ease the process for users who want to extend vla_arena, creating new tasks. You would still need to make necessary changes based on the template to serve your own need, but the hope is that we save you much time by providing the necessar templates.
"""

import os
import xml.etree.ElementTree as ET

from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.envs.textures import get_texture_file_list


def create_problem_class_from_file(class_name):
    template_source_file = os.path.join(
        get_vla_arena_path('benchmark_root'),
        '../../templates/problem_class_template.py',
    )
    with open(template_source_file) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if 'YOUR_CLASS_NAME' in line:
            line = line.replace('YOUR_CLASS_NAME', class_name)
        new_lines.append(line)
    with open(f'{class_name.lower()}.py', 'w') as f:
        f.writelines(new_lines)
    print(f'Creating class {class_name} at the file: {class_name.lower()}.py')


def create_scene_xml_file(scene_name):
    """This is just an example for you to jump start. For more advanced editing, you will need to figure out yourself. You can take a look at all the available xml files for reference."""
    template_source_file = os.path.join(
        get_vla_arena_path('benchmark_root'),
        '../../templates/scene_template.xml',
    )
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(template_source_file, parser)
    root = tree.getroot()

    basic_elements = [
        ('Floor', 'texplane'),
        ('Table', 'tex-table'),
        ('Table legs', 'tex-table-legs'),
        ('Walls', 'tex-wall'),
    ]

    for element_name, texture_name in basic_elements:
        element = root.findall(f'.//texture[@name="{texture_name}"]')[0]
        type = None
        if 'floor' in element_name.lower():
            type = 'floor'
        elif 'table' in element_name.lower():
            type = 'table'
        elif 'wall' in element_name.lower():
            type = 'wall'
        # If you want to change the path of the texture file, you can pass in texture_path variable to change it.
        texture_list = get_texture_file_list(type=type, texture_path='../')
        for i, (texture_name, texture_file_path) in enumerate(texture_list):
            print(f'[{i}]: {texture_name}')
        choice = int(
            input(f'Please select which texture to use for {element_name}: ')
        )
        element.set('file', texture_list[choice][1])
    tree.write(f'{scene_name}.xml', encoding='utf-8')
    print(f'Creating scene {scene_name} at the file: {scene_name}.xml')
    print(
        '\n [Notice] The texture fiile paths are specified in the relative path format assuming your scene xml will be placed in the path vla_arena/vla_arena/assets/scenes/. ',
    )
    return


def main():
    # use keyboard to select which file to create
    choices = [
        'problem_class',
        'scene',
        'object',
        'arena',
    ]

    for i, choice in enumerate(choices):
        print(f'[{i}]: {choice}')
    choice = int(input('Please select which file to create: '))

    if choices[choice] == 'problem_class':
        # Ask user to specify the class name
        class_name = input('Please specify the class name: ')
        assert ' ' not in class_name, 'space is not allowed in the naming'
        parts = class_name.split('_')
        class_name = '_'.join([part.lower().capitalize() for part in parts])
        create_problem_class_from_file(class_name)
    elif choices[choice] == 'scene':
        # Ask user to specify the scene name
        scene_name = input('Please specify the scene name: ')
        scene_name = scene_name.lower()
        assert ' ' not in scene_name, 'space is not allowed in the naming'
        create_scene_xml_file(scene_name)


if __name__ == '__main__':
    main()
