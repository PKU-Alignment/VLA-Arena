# Copyright (c) 2024-2025 VLA-Arena Team. All Rights Reserved.
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
# ==============================================================================

"""
VLA-Arena Task Asset Manager

This module provides a complete workflow for packaging, sharing, and installing
task assets in VLA-Arena.

Workflow: Design -> Pack -> Upload -> Download -> Install -> Use
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


try:
    from termcolor import colored
except ImportError:

    def colored(text, color=None, **kwargs):
        return text


try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# Lightweight imports - avoid importing robosuite which has heavy dependencies
from vla_arena.vla_arena import get_vla_arena_path


# Lazy imports for heavy modules
_bddl_utils_loaded = False
_objects_loaded = False


def _lazy_load_bddl_utils():
    """Lazy load BDDL utils to avoid importing robosuite at module load time."""
    global _bddl_utils_loaded
    if not _bddl_utils_loaded:
        try:
            from vla_arena.vla_arena.envs.bddl_utils import robosuite_parse_problem

            globals()['robosuite_parse_problem'] = robosuite_parse_problem
            _bddl_utils_loaded = True
        except ImportError:
            pass
    return _bddl_utils_loaded


def _lazy_load_objects():
    """Lazy load objects module to avoid importing robosuite at module load time."""
    global _objects_loaded
    if not _objects_loaded:
        try:
            from vla_arena.vla_arena.envs.objects import get_object_dict

            globals()['get_object_dict'] = get_object_dict
            _objects_loaded = True
        except ImportError:
            pass
    return _objects_loaded


def lightweight_parse_bddl(bddl_path: str) -> Dict:
    """
    Lightweight BDDL parser that doesn't require robosuite.
    Parses BDDL files using simple regex patterns.
    """
    with open(bddl_path) as f:
        content = f.read()

    result = {
        'problem_name': '',
        'language_instruction': [],
        'fixtures': {},
        'objects': {},
        'obj_of_interest': [],
        'initial_state': [],
        'goal_state': [],
    }

    # Extract problem name
    problem_match = re.search(r'\(problem\s+(\w+)\)', content)
    if problem_match:
        result['problem_name'] = problem_match.group(1)

    # Extract language instruction
    lang_match = re.search(r'\(:language\s+([^)]+)\)', content)
    if lang_match:
        result['language_instruction'] = lang_match.group(1).strip().split()

    # Extract fixtures
    fixtures_match = re.search(r'\(:fixtures([^)]*)\)', content, re.DOTALL)
    if fixtures_match:
        fixtures_content = fixtures_match.group(1).strip()
        # Parse "name - type" patterns
        for match in re.finditer(r'(\w+)\s*-\s*(\w+)', fixtures_content):
            obj_name, obj_type = match.groups()
            if obj_type not in result['fixtures']:
                result['fixtures'][obj_type] = []
            result['fixtures'][obj_type].append(obj_name)

    # Extract objects
    objects_match = re.search(r'\(:objects([^)]*)\)', content, re.DOTALL)
    if objects_match:
        objects_content = objects_match.group(1).strip()
        # Parse "name - type" patterns
        for match in re.finditer(r'(\w+)\s*-\s*(\w+)', objects_content):
            obj_name, obj_type = match.groups()
            if obj_type not in result['objects']:
                result['objects'][obj_type] = []
            result['objects'][obj_type].append(obj_name)

    # Extract objects of interest
    interest_match = re.search(r'\(:obj_of_interest([^)]*)\)', content, re.DOTALL)
    if interest_match:
        interest_content = interest_match.group(1).strip()
        result['obj_of_interest'] = interest_content.split()

    return result


def find_problem_class_file(problem_name: str) -> Optional[str]:
    """
    Find the Python file containing the Problem class.

    Args:
        problem_name: Name of the problem class (e.g., "Tabletop_Manipulation")

    Returns:
        Path to the .py file, or None if not found
    """
    try:
        problems_dir = os.path.join(get_vla_arena_path('benchmark_root'), 'envs', 'problems')
    except:
        return None

    if not os.path.exists(problems_dir):
        return None

    # Convert problem name to potential file name (e.g., "Tabletop_Manipulation" -> "tabletop_manipulation.py")
    potential_filename = problem_name.lower() + '.py'
    potential_path = os.path.join(problems_dir, potential_filename)

    if os.path.exists(potential_path):
        return potential_path

    # Search in all .py files for the class definition
    for filename in os.listdir(problems_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(problems_dir, filename)
            try:
                with open(filepath) as f:
                    content = f.read()
                    # Look for class definition
                    if re.search(rf'class\s+{problem_name}\s*\(', content):
                        return filepath
            except:
                continue

    return None


def extract_scene_xml_from_problem(problem_file: str) -> Optional[str]:
    """
    Extract scene_xml path from a Problem class file.

    Args:
        problem_file: Path to the Problem class .py file

    Returns:
        The scene_xml path (e.g., "scenes/tabletop_warm_style.xml"), or None
    """
    if not problem_file or not os.path.exists(problem_file):
        return None

    try:
        with open(problem_file) as f:
            content = f.read()

        # Look for scene_xml patterns like:
        # kwargs.update({'scene_xml': 'scenes/tabletop_warm_style.xml'})
        # or: 'scene_xml': 'scenes/xxx.xml'
        patterns = [
            r"['\"]scene_xml['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            r"scene_xml\s*=\s*['\"]([^'\"]+)['\"]",
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)

        return None
    except:
        return None


def parse_scene_xml_assets(scene_xml_path: str) -> Dict[str, List[str]]:
    """
    Parse a scene XML file to extract referenced assets (textures, meshes).

    Args:
        scene_xml_path: Full path to the scene XML file

    Returns:
        Dictionary with 'textures' and 'meshes' lists
    """
    result = {
        'textures': [],
        'meshes': [],
        'includes': [],
    }

    if not scene_xml_path or not os.path.exists(scene_xml_path):
        return result

    try:
        with open(scene_xml_path) as f:
            content = f.read()

        scene_dir = os.path.dirname(scene_xml_path)

        # Find texture files: file="..." in texture tags
        texture_pattern = r'<texture[^>]+file\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(texture_pattern, content):
            texture_path = match.group(1)
            # Resolve relative path
            full_path = os.path.normpath(os.path.join(scene_dir, texture_path))
            if os.path.exists(full_path):
                result['textures'].append(full_path)

        # Find mesh files: file="..." in mesh tags or mesh="..." in geom tags
        mesh_patterns = [
            r'<mesh[^>]+file\s*=\s*["\']([^"\']+)["\']',
            r'mesh\s*=\s*["\']([^"\']+\.(?:obj|stl|msh))["\']',
        ]
        for pattern in mesh_patterns:
            for match in re.finditer(pattern, content):
                mesh_path = match.group(1)
                full_path = os.path.normpath(os.path.join(scene_dir, mesh_path))
                if os.path.exists(full_path):
                    result['meshes'].append(full_path)

        # Find include files
        include_pattern = r'<include\s+file\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(include_pattern, content):
            include_path = match.group(1)
            full_path = os.path.normpath(os.path.join(scene_dir, include_path))
            if os.path.exists(full_path):
                result['includes'].append(full_path)

        return result
    except Exception as e:
        print(f'[Warning] Error parsing scene XML: {e}')
        return result


@dataclass
class SceneInfo:
    """Information about a scene and its assets."""

    problem_name: str
    problem_file: Optional[str]
    scene_xml: Optional[str]
    scene_xml_full_path: Optional[str]
    textures: List[str] = field(default_factory=list)
    meshes: List[str] = field(default_factory=list)

    def has_custom_scene(self) -> bool:
        """Check if this is a custom scene that needs to be packaged."""
        return self.scene_xml is not None and self.scene_xml_full_path is not None


def analyze_problem_and_scene(bddl_path: str) -> SceneInfo:
    """
    Analyze a BDDL file to extract problem class and scene information.

    Args:
        bddl_path: Path to the BDDL file

    Returns:
        SceneInfo object with all extracted information
    """
    parsed = lightweight_parse_bddl(bddl_path)
    problem_name = parsed.get('problem_name', '')

    # Find problem class file
    problem_file = find_problem_class_file(problem_name)

    # Extract scene_xml from problem class
    scene_xml = extract_scene_xml_from_problem(problem_file) if problem_file else None

    # Resolve full path of scene_xml
    scene_xml_full_path = None
    if scene_xml:
        try:
            assets_root = get_vla_arena_path('assets')
            scene_xml_full_path = os.path.join(assets_root, scene_xml)
            if not os.path.exists(scene_xml_full_path):
                scene_xml_full_path = None
        except:
            pass

    # Parse scene XML for assets
    textures = []
    meshes = []
    if scene_xml_full_path:
        scene_assets = parse_scene_xml_assets(scene_xml_full_path)
        textures = scene_assets['textures']
        meshes = scene_assets['meshes']

    return SceneInfo(
        problem_name=problem_name,
        problem_file=problem_file,
        scene_xml=scene_xml,
        scene_xml_full_path=scene_xml_full_path,
        textures=textures,
        meshes=meshes,
    )


# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, snapshot_download, upload_folder

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


# =============================================================================
# Constants and Configuration
# =============================================================================

PACKAGE_EXTENSION = '.vlap'  # VLA-Arena Package
MANIFEST_FILENAME = 'manifest.json'
PACKAGE_VERSION = '1.0.0'

# Asset source directories mapping
ASSET_SOURCE_MAPPING = {
    'google_scanned': 'stable_scanned_objects',
    'hope': 'stable_hope_objects',
    'turbosquid': 'turbosquid_objects',
    'articulated': 'articulated_objects',
    'scenes': 'scenes',
}

# Fixtures that are handled by Arena classes, not as standalone assets
# These are created automatically by the simulation environment
BUILTIN_FIXTURES = {
    'table',
    'main_table',
    'kitchen_table',
    'coffee_table',
    'living_room_table',
    'study_table',
    'floor',
}

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AssetInfo:
    """Information about a single asset."""

    name: str
    category: str  # e.g., "google_scanned", "hope", "turbosquid"
    source_path: str
    size_bytes: int = 0
    file_count: int = 0
    checksum: str = ''


@dataclass
class TaskManifest:
    """Manifest file for a task package."""

    # Basic info
    package_name: str
    version: str = '1.0.0'
    package_format_version: str = PACKAGE_VERSION

    # Task info
    task_name: str = ''
    language_instruction: str = ''
    description: str = ''
    problem_class: str = ''  # Problem class name

    # Author info
    author: str = ''
    email: str = ''
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Contents
    bddl_files: List[str] = field(default_factory=list)
    init_files: List[str] = field(default_factory=list)
    problem_files: List[str] = field(default_factory=list)  # Custom Problem class files
    scene_files: List[str] = field(default_factory=list)  # Scene XML files
    scene_assets: List[str] = field(default_factory=list)  # Scene textures/meshes
    assets: List[Dict] = field(default_factory=list)  # Object assets

    # Dependencies
    fixtures: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)

    # Checksums
    total_size_bytes: int = 0
    package_checksum: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskManifest':
        return cls(**data)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'TaskManifest':
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Asset Dependency Analyzer
# =============================================================================


class AssetDependencyAnalyzer:
    """Analyzes BDDL files to extract asset dependencies."""

    def __init__(self, use_lightweight_parser: bool = True):
        """
        Initialize the analyzer.

        Args:
            use_lightweight_parser: If True, use regex-based parser that doesn't
                                   require robosuite. If False, try to use the
                                   full robosuite-based parser.
        """
        self.use_lightweight_parser = use_lightweight_parser
        self.object_dict = {}
        self.asset_mapping = {}
        self._build_asset_mapping()

    def _build_asset_mapping(self):
        """Build a mapping from object names to their asset sources."""
        # Try to load object dict from robosuite-based module
        if _lazy_load_objects():
            self.object_dict = get_object_dict()
            for name, cls in self.object_dict.items():
                module_name = cls.__module__
                if 'google_scanned' in module_name:
                    self.asset_mapping[name] = ('google_scanned', 'stable_scanned_objects')
                elif 'hope' in module_name:
                    self.asset_mapping[name] = ('hope', 'stable_hope_objects')
                elif 'turbosquid' in module_name:
                    self.asset_mapping[name] = ('turbosquid', 'turbosquid_objects')
                elif 'articulated' in module_name:
                    self.asset_mapping[name] = ('articulated', 'articulated_objects')

        # Also build mapping from filesystem scanning (fallback/supplement)
        self._build_asset_mapping_from_filesystem()

    def _build_asset_mapping_from_filesystem(self):
        """Build asset mapping by scanning the assets directory."""
        try:
            assets_root = get_vla_arena_path('assets')
        except:
            return

        # Mapping of directory names to categories
        dir_to_category = {
            'stable_scanned_objects': 'google_scanned',
            'stable_hope_objects': 'hope',
            'turbosquid_objects': 'turbosquid',
            'articulated_objects': 'articulated',
        }

        for dir_name, category in dir_to_category.items():
            dir_path = os.path.join(assets_root, dir_name)
            if os.path.exists(dir_path):
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        # Register the asset name
                        asset_name = item.lower()
                        if asset_name not in self.asset_mapping:
                            self.asset_mapping[asset_name] = (category, dir_name)

    def analyze_bddl(self, bddl_path: str) -> Tuple[Set[str], Set[str], Dict]:
        """
        Analyze a BDDL file to extract required assets.

        Args:
            bddl_path: Path to the BDDL file

        Returns:
            Tuple of (fixtures, objects, parsed_info)
        """
        # Try full parser first if available and not forced to lightweight
        if not self.use_lightweight_parser and _lazy_load_bddl_utils():
            try:
                parsed = robosuite_parse_problem(bddl_path)
            except Exception:
                parsed = lightweight_parse_bddl(bddl_path)
        else:
            parsed = lightweight_parse_bddl(bddl_path)

        fixtures = set()
        objects = set()

        # Extract fixtures (e.g., tables)
        for fixture_type, fixture_list in parsed.get('fixtures', {}).items():
            fixtures.add(fixture_type)
            for f in fixture_list:
                fixtures.add(f)

        # Extract objects
        for obj_type, obj_list in parsed.get('objects', {}).items():
            objects.add(obj_type)

        return fixtures, objects, parsed

    def get_asset_paths(self, object_names: Set[str]) -> List[AssetInfo]:
        """
        Get asset file paths for given object names.

        Args:
            object_names: Set of object type names

        Returns:
            List of AssetInfo objects
        """
        try:
            assets_root = get_vla_arena_path('assets')
        except Exception as e:
            print(f'[Warning] Could not get assets path: {e}')
            return []

        asset_infos = []

        for name in object_names:
            name_lower = name.lower()

            if name_lower not in self.asset_mapping:
                # Try to find asset by scanning filesystem
                found = False
                for source_dir in [
                    'stable_scanned_objects',
                    'stable_hope_objects',
                    'turbosquid_objects',
                    'articulated_objects',
                ]:
                    asset_path = os.path.join(assets_root, source_dir, name_lower)
                    if os.path.exists(asset_path):
                        category = {
                            'stable_scanned_objects': 'google_scanned',
                            'stable_hope_objects': 'hope',
                            'turbosquid_objects': 'turbosquid',
                            'articulated_objects': 'articulated',
                        }.get(source_dir, source_dir)
                        self.asset_mapping[name_lower] = (category, source_dir)
                        found = True
                        break

                if not found:
                    # Skip built-in fixtures (handled by Arena, not standalone assets)
                    if name_lower not in BUILTIN_FIXTURES:
                        print(f'[Warning] Unknown object type: {name}')
                    continue

            category, source_dir = self.asset_mapping[name_lower]

            # Get the actual asset name - try object class first, then fall back to name
            obj_name = name_lower
            if self.object_dict and name_lower in self.object_dict:
                obj_cls = self.object_dict[name_lower]
                try:
                    obj_instance = obj_cls()
                    obj_name = getattr(obj_instance, 'name', name_lower)
                except:
                    pass

            asset_path = os.path.join(assets_root, source_dir, obj_name)

            if os.path.exists(asset_path):
                # Calculate size and file count
                size = 0
                count = 0
                for root, dirs, files in os.walk(asset_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            size += os.path.getsize(fp)
                        except:
                            pass
                        count += 1

                asset_infos.append(
                    AssetInfo(
                        name=obj_name,
                        category=category,
                        source_path=asset_path,
                        size_bytes=size,
                        file_count=count,
                        checksum=self._calculate_dir_checksum(asset_path),
                    ),
                )
            else:
                print(f'[Warning] Asset not found: {asset_path}')

        return asset_infos

    def _calculate_dir_checksum(self, dir_path: str) -> str:
        """Calculate MD5 checksum for a directory."""
        hasher = hashlib.md5()
        for root, dirs, files in sorted(os.walk(dir_path)):
            dirs.sort()
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                with open(filepath, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
        return hasher.hexdigest()


# =============================================================================
# Task Packager
# =============================================================================


class TaskPackager:
    """Packages task files and assets into a distributable format."""

    def __init__(self):
        self.analyzer = AssetDependencyAnalyzer()

    def _find_init_file_for_bddl(self, bddl_path: str) -> Optional[str]:
        """
        Auto-detect the corresponding init file for a BDDL file.

        Looks for a .pruned_init file with the same name in the parallel init_files directory.

        Args:
            bddl_path: Path to the BDDL file

        Returns:
            Path to the init file, or None if not found
        """
        bddl_path = os.path.abspath(bddl_path)
        bddl_name = Path(bddl_path).stem

        # Get the relative path from bddl_files directory
        try:
            bddl_root = get_vla_arena_path('bddl_files')
            init_root = get_vla_arena_path('init_states')

            # Calculate relative path
            rel_path = os.path.relpath(bddl_path, bddl_root)
            rel_dir = os.path.dirname(rel_path)

            # Construct potential init file path
            init_file = os.path.join(init_root, rel_dir, f'{bddl_name}.pruned_init')

            if os.path.exists(init_file):
                return init_file
        except:
            pass

        # Fallback: look in the same directory structure but under init_files
        bddl_dir = os.path.dirname(bddl_path)
        if 'bddl_files' in bddl_dir:
            init_dir = bddl_dir.replace('bddl_files', 'init_files')
            init_file = os.path.join(init_dir, f'{bddl_name}.pruned_init')
            if os.path.exists(init_file):
                return init_file

        return None

    def pack(
        self,
        bddl_path: str,
        output_dir: str,
        init_path: Optional[str] = None,
        package_name: Optional[str] = None,
        author: str = '',
        email: str = '',
        description: str = '',
        include_assets: bool = True,
        include_problem: bool = True,
        include_scene: bool = True,
    ) -> str:
        """
        Package a task with its dependencies.

        Args:
            bddl_path: Path to the BDDL file
            output_dir: Directory to save the package
            init_path: Optional path to init file
            package_name: Name for the package (defaults to BDDL filename)
            author: Author name
            email: Author email
            description: Task description
            include_assets: Whether to include object asset files
            include_problem: Whether to include custom Problem class files
            include_scene: Whether to include scene XML and its assets

        Returns:
            Path to the created package
        """
        # Analyze dependencies
        fixtures, objects, parsed = self.analyzer.analyze_bddl(bddl_path)

        # Get object asset infos
        all_objects = fixtures | objects
        asset_infos = self.analyzer.get_asset_paths(all_objects) if include_assets else []

        # Analyze problem class and scene
        scene_info = analyze_problem_and_scene(bddl_path)

        # Auto-detect init file if not provided
        if init_path is None:
            init_path = self._find_init_file_for_bddl(bddl_path)

        # Create package name
        if package_name is None:
            package_name = Path(bddl_path).stem

        # Prepare lists for manifest
        problem_files_list = []
        scene_files_list = []
        scene_assets_list = []

        if include_problem and scene_info.problem_file:
            problem_files_list.append(os.path.basename(scene_info.problem_file))

        if include_scene and scene_info.scene_xml:
            scene_files_list.append(scene_info.scene_xml)
            # Add relative paths of scene assets
            if scene_info.textures or scene_info.meshes:
                try:
                    assets_root = get_vla_arena_path('assets')
                    for tex in scene_info.textures:
                        rel_path = os.path.relpath(tex, assets_root)
                        scene_assets_list.append(rel_path)
                    for mesh in scene_info.meshes:
                        rel_path = os.path.relpath(mesh, assets_root)
                        scene_assets_list.append(rel_path)
                except:
                    pass

        # Create manifest
        manifest = TaskManifest(
            package_name=package_name,
            task_name=parsed.get('problem_name', ''),
            language_instruction=' '.join(parsed.get('language_instruction', [])),
            description=description,
            problem_class=scene_info.problem_name,
            author=author,
            email=email,
            bddl_files=[os.path.basename(bddl_path)],
            init_files=[os.path.basename(init_path)] if init_path else [],
            problem_files=problem_files_list,
            scene_files=scene_files_list,
            scene_assets=scene_assets_list,
            assets=[asdict(a) for a in asset_infos],
            fixtures=list(fixtures),
            objects=list(objects),
            total_size_bytes=sum(a.size_bytes for a in asset_infos),
        )

        # Create package
        os.makedirs(output_dir, exist_ok=True)
        package_path = os.path.join(output_dir, f'{package_name}{PACKAGE_EXTENSION}')

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create package structure
            pkg_root = os.path.join(temp_dir, package_name)
            os.makedirs(pkg_root)

            # Copy BDDL file
            bddl_dir = os.path.join(pkg_root, 'bddl_files')
            os.makedirs(bddl_dir)
            shutil.copy2(bddl_path, bddl_dir)

            # Copy init file if provided
            if init_path and os.path.exists(init_path):
                init_dir = os.path.join(pkg_root, 'init_files')
                os.makedirs(init_dir)
                shutil.copy2(init_path, init_dir)

            # Copy Problem class file
            if include_problem and scene_info.problem_file:
                problems_dir = os.path.join(pkg_root, 'problems')
                os.makedirs(problems_dir, exist_ok=True)
                shutil.copy2(scene_info.problem_file, problems_dir)
                print(f'  + Problem class: {os.path.basename(scene_info.problem_file)}')

            # Copy scene XML and its assets
            if include_scene and scene_info.scene_xml_full_path:
                assets_dir = os.path.join(pkg_root, 'assets')
                os.makedirs(assets_dir, exist_ok=True)

                # Copy scene XML (preserve directory structure under assets/)
                scene_rel_dir = os.path.dirname(scene_info.scene_xml)  # e.g., "scenes"
                scene_dest_dir = os.path.join(assets_dir, scene_rel_dir)
                os.makedirs(scene_dest_dir, exist_ok=True)
                shutil.copy2(scene_info.scene_xml_full_path, scene_dest_dir)
                print(f'  + Scene: {scene_info.scene_xml}')

                # Copy scene textures and meshes
                try:
                    assets_root = get_vla_arena_path('assets')
                    for tex_path in scene_info.textures:
                        rel_path = os.path.relpath(tex_path, assets_root)
                        dest_path = os.path.join(assets_dir, rel_path)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        if not os.path.exists(dest_path):
                            shutil.copy2(tex_path, dest_path)

                    for mesh_path in scene_info.meshes:
                        rel_path = os.path.relpath(mesh_path, assets_root)
                        dest_path = os.path.join(assets_dir, rel_path)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        if not os.path.exists(dest_path):
                            shutil.copy2(mesh_path, dest_path)

                    if scene_info.textures or scene_info.meshes:
                        print(
                            f'  + Scene assets: {len(scene_info.textures)} textures, {len(scene_info.meshes)} meshes',
                        )
                except Exception as e:
                    print(f'  [Warning] Error copying scene assets: {e}')

            # Copy object assets
            if include_assets and asset_infos:
                assets_dir = os.path.join(pkg_root, 'assets')
                os.makedirs(assets_dir, exist_ok=True)

                for asset in tqdm(asset_infos, desc='Copying object assets'):
                    # Create category subdirectory
                    category_dir = os.path.join(
                        assets_dir,
                        ASSET_SOURCE_MAPPING.get(asset.category, asset.category),
                    )
                    os.makedirs(category_dir, exist_ok=True)

                    # Copy asset directory
                    dest_path = os.path.join(category_dir, asset.name)
                    if not os.path.exists(dest_path):
                        shutil.copytree(asset.source_path, dest_path)

            # Save manifest
            manifest.save(os.path.join(pkg_root, MANIFEST_FILENAME))

            # Create ZIP archive
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(pkg_root):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname)

        # Calculate package checksum
        with open(package_path, 'rb') as f:
            manifest.package_checksum = hashlib.md5(f.read()).hexdigest()

        print(colored(f'✓ Package created: {package_path}', 'green'))
        print(f'  - Task: {manifest.task_name}')
        print(f'  - Problem class: {scene_info.problem_name}')
        print(f"  - Scene: {scene_info.scene_xml or 'default'}")
        print(f'  - Objects: {len(asset_infos)}')
        print(f'  - Size: {manifest.total_size_bytes / 1024 / 1024:.2f} MB')

        return package_path

    def pack_task_suite(
        self,
        task_suite_name: str,
        output_dir: str,
        author: str = '',
        email: str = '',
        description: str = '',
    ) -> str:
        """
        Package an entire task suite.

        Args:
            task_suite_name: Name of the task suite (e.g., "robustness_dynamic_distractors")
            output_dir: Directory to save the package
            author: Author name
            email: Author email
            description: Task suite description

        Returns:
            Path to the created package
        """
        bddl_root = get_vla_arena_path('bddl_files')
        init_root = get_vla_arena_path('init_states')

        suite_bddl_dir = os.path.join(bddl_root, task_suite_name)
        suite_init_dir = os.path.join(init_root, task_suite_name)

        if not os.path.exists(suite_bddl_dir):
            raise FileNotFoundError(f'Task suite not found: {suite_bddl_dir}')

        # Collect all objects from all BDDL files
        all_fixtures = set()
        all_objects = set()
        bddl_files = []
        init_files = []

        for root, dirs, files in os.walk(suite_bddl_dir):
            for f in files:
                if f.endswith('.bddl'):
                    bddl_path = os.path.join(root, f)
                    fixtures, objects, _ = self.analyzer.analyze_bddl(bddl_path)
                    all_fixtures |= fixtures
                    all_objects |= objects

                    rel_path = os.path.relpath(bddl_path, suite_bddl_dir)
                    bddl_files.append(rel_path)

        # Find corresponding init files
        for root, dirs, files in os.walk(suite_init_dir):
            for f in files:
                if f.endswith('.pruned_init'):
                    rel_path = os.path.relpath(os.path.join(root, f), suite_init_dir)
                    init_files.append(rel_path)

        # Get all assets
        all_asset_objects = all_fixtures | all_objects
        asset_infos = self.analyzer.get_asset_paths(all_asset_objects)

        # Create manifest
        manifest = TaskManifest(
            package_name=task_suite_name,
            task_name=task_suite_name,
            description=description or f'Task suite: {task_suite_name}',
            author=author,
            email=email,
            bddl_files=bddl_files,
            init_files=init_files,
            assets=[asdict(a) for a in asset_infos],
            fixtures=list(all_fixtures),
            objects=list(all_objects),
            total_size_bytes=sum(a.size_bytes for a in asset_infos),
        )

        # Create package
        os.makedirs(output_dir, exist_ok=True)
        package_path = os.path.join(output_dir, f'{task_suite_name}{PACKAGE_EXTENSION}')

        with tempfile.TemporaryDirectory() as temp_dir:
            pkg_root = os.path.join(temp_dir, task_suite_name)
            os.makedirs(pkg_root)

            # Copy BDDL files (preserving directory structure)
            bddl_dest = os.path.join(pkg_root, 'bddl_files')
            shutil.copytree(suite_bddl_dir, bddl_dest)

            # Copy init files
            if os.path.exists(suite_init_dir):
                init_dest = os.path.join(pkg_root, 'init_files')
                shutil.copytree(suite_init_dir, init_dest)

            # Copy assets
            assets_dir = os.path.join(pkg_root, 'assets')
            os.makedirs(assets_dir)

            for asset in tqdm(asset_infos, desc='Copying assets'):
                category_dir = os.path.join(
                    assets_dir,
                    ASSET_SOURCE_MAPPING.get(asset.category, asset.category),
                )
                os.makedirs(category_dir, exist_ok=True)
                dest_path = os.path.join(category_dir, asset.name)
                if not os.path.exists(dest_path):
                    shutil.copytree(asset.source_path, dest_path)

            # Save manifest
            manifest.save(os.path.join(pkg_root, MANIFEST_FILENAME))

            # Create ZIP
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(pkg_root):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname)

        print(colored(f'✓ Task suite packaged: {package_path}', 'green'))
        print(f'  - Tasks: {len(bddl_files)}')
        print(f'  - Assets: {len(asset_infos)}')
        print(f'  - Size: {manifest.total_size_bytes / 1024 / 1024:.2f} MB')

        return package_path


# =============================================================================
# Package Installer
# =============================================================================


class TaskInstaller:
    """Installs task packages into the VLA-Arena environment."""

    def __init__(self):
        self.assets_root = get_vla_arena_path('assets')
        self.bddl_root = get_vla_arena_path('bddl_files')
        self.init_root = get_vla_arena_path('init_states')

    def inspect(self, package_path: str) -> TaskManifest:
        """
        Inspect a package without installing.

        Args:
            package_path: Path to the .vlap package

        Returns:
            TaskManifest object
        """
        with zipfile.ZipFile(package_path, 'r') as zf:
            # Find manifest
            manifest_path = None
            for name in zf.namelist():
                if name.endswith(MANIFEST_FILENAME):
                    manifest_path = name
                    break

            if not manifest_path:
                raise ValueError('Invalid package: manifest.json not found')

            with zf.open(manifest_path) as f:
                manifest = TaskManifest.from_dict(json.load(f))

        return manifest

    def check_conflicts(self, package_path: str) -> Dict[str, List[str]]:
        """
        Check for potential conflicts with existing assets.

        Args:
            package_path: Path to the .vlap package

        Returns:
            Dictionary with conflict information (values are relative paths)
        """
        manifest = self.inspect(package_path)
        conflicts = {
            'assets': [],
            'bddl_files': [],
            'init_files': [],
        }

        # Check assets - store relative paths
        for asset in manifest.assets:
            asset_info = AssetInfo(**asset)
            category_dir = ASSET_SOURCE_MAPPING.get(asset_info.category, asset_info.category)
            existing_path = os.path.join(self.assets_root, category_dir, asset_info.name)
            if os.path.exists(existing_path):
                # Store relative path from assets root
                relative_path = os.path.join(category_dir, asset_info.name)
                conflicts['assets'].append(f'assets/{relative_path}')

        # Check BDDL files
        for bddl_file in manifest.bddl_files:
            existing_path = os.path.join(self.bddl_root, manifest.package_name, bddl_file)
            if os.path.exists(existing_path):
                conflicts['bddl_files'].append(bddl_file)

        # Check init files
        for init_file in manifest.init_files:
            existing_path = os.path.join(self.init_root, manifest.package_name, init_file)
            if os.path.exists(existing_path):
                conflicts['init_files'].append(init_file)

        return conflicts

    def install(
        self,
        package_path: str,
        overwrite: bool = False,
        skip_assets: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """
        Install a task package.

        Args:
            package_path: Path to the .vlap package
            overwrite: Whether to overwrite existing files
            skip_assets: Skip installing assets (useful if already installed)
            dry_run: Only show what would be installed

        Returns:
            True if successful
        """
        manifest = self.inspect(package_path)

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Installing: {manifest.package_name}")
        print(f'  Task: {manifest.task_name}')
        print(f'  Description: {manifest.description}')
        print(f'  Author: {manifest.author}')

        # Check conflicts
        conflicts = self.check_conflicts(package_path)
        has_conflicts = any(len(v) > 0 for v in conflicts.values())

        if has_conflicts and not overwrite:
            print(colored('\n⚠ Conflicts detected:', 'yellow'))

            # Display BDDL conflicts
            if conflicts['bddl_files']:
                for bddl_file in conflicts['bddl_files'][:5]:
                    print(f'  - BDDL: {bddl_file} (already exists)')
                if len(conflicts['bddl_files']) > 5:
                    print(f"  - ... and {len(conflicts['bddl_files'])-5} more BDDL files")

            # Display init file conflicts
            if conflicts['init_files']:
                for init_file in conflicts['init_files'][:5]:
                    print(f'  - Init: {init_file} (already exists)')
                if len(conflicts['init_files']) > 5:
                    print(f"  - ... and {len(conflicts['init_files'])-5} more init files")

            # Display asset conflicts
            if conflicts['assets']:
                for asset_path in conflicts['assets'][:5]:
                    print(f'  - Asset: {asset_path} (already exists)')

                if len(conflicts['assets']) > 5:
                    print(f"  - ... and {len(conflicts['assets'])-5} more assets")

            print(colored('\nUse --overwrite to replace existing files.', 'yellow'))
            return False

        if dry_run:
            print(colored('\n✓ Dry run complete. No files were modified.', 'blue'))
            return True

        # Extract and install
        with zipfile.ZipFile(package_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as temp_dir:
                zf.extractall(temp_dir)

                pkg_root = os.path.join(temp_dir, manifest.package_name)

                # Install BDDL files
                src_bddl = os.path.join(pkg_root, 'bddl_files')
                if os.path.exists(src_bddl):
                    dest_bddl = os.path.join(self.bddl_root, manifest.package_name)
                    if os.path.exists(dest_bddl) and overwrite:
                        shutil.rmtree(dest_bddl)
                    if not os.path.exists(dest_bddl):
                        shutil.copytree(src_bddl, dest_bddl)
                    print(f'  ✓ BDDL files installed: {dest_bddl}')

                # Install init files
                src_init = os.path.join(pkg_root, 'init_files')
                if os.path.exists(src_init):
                    dest_init = os.path.join(self.init_root, manifest.package_name)
                    if os.path.exists(dest_init) and overwrite:
                        shutil.rmtree(dest_init)
                    if not os.path.exists(dest_init):
                        shutil.copytree(src_init, dest_init)
                    print(f'  ✓ Init files installed: {dest_init}')

                # Install Problem class files
                src_problems = os.path.join(pkg_root, 'problems')
                if os.path.exists(src_problems):
                    # Problem files go to envs/problems/
                    dest_problems = os.path.join(
                        get_vla_arena_path('benchmark_root'),
                        'envs',
                        'problems',
                    )
                    os.makedirs(dest_problems, exist_ok=True)

                    for problem_file in os.listdir(src_problems):
                        if problem_file.endswith('.py'):
                            src_file = os.path.join(src_problems, problem_file)
                            dest_file = os.path.join(dest_problems, problem_file)

                            if os.path.exists(dest_file) and not overwrite:
                                print(f'  ⚠ Problem file exists (skipped): {problem_file}')
                                continue

                            shutil.copy2(src_file, dest_file)
                            print(f'  ✓ Problem class installed: {problem_file}')

                # Install assets (including scene files)
                if not skip_assets:
                    src_assets = os.path.join(pkg_root, 'assets')
                    if os.path.exists(src_assets):
                        installed_count = 0

                        # Recursively copy all assets
                        for root, dirs, files in os.walk(src_assets):
                            # Calculate relative path from src_assets
                            rel_root = os.path.relpath(root, src_assets)
                            dest_root = os.path.join(self.assets_root, rel_root)
                            os.makedirs(dest_root, exist_ok=True)

                            # Copy files
                            for file in files:
                                src_file = os.path.join(root, file)
                                dest_file = os.path.join(dest_root, file)

                                if os.path.exists(dest_file):
                                    if overwrite:
                                        os.remove(dest_file)
                                    else:
                                        continue

                                shutil.copy2(src_file, dest_file)
                                installed_count += 1

                        print(f'  ✓ Assets installed: {installed_count} files')

        print(colored(f'\n✓ Installation complete: {manifest.package_name}', 'green'))
        return True

    def uninstall(self, package_name: str, remove_assets: bool = False) -> bool:
        """
        Uninstall a task package.

        Args:
            package_name: Name of the package to uninstall
            remove_assets: Whether to also remove associated assets

        Returns:
            True if successful
        """
        # Remove BDDL files
        bddl_dir = os.path.join(self.bddl_root, package_name)
        if os.path.exists(bddl_dir):
            shutil.rmtree(bddl_dir)
            print(f'  ✓ Removed BDDL files: {bddl_dir}')

        # Remove init files
        init_dir = os.path.join(self.init_root, package_name)
        if os.path.exists(init_dir):
            shutil.rmtree(init_dir)
            print(f'  ✓ Removed init files: {init_dir}')

        if remove_assets:
            print(
                colored(
                    '⚠ Asset removal is not recommended as they may be shared by other tasks.',
                    'yellow',
                ),
            )

        print(colored(f'✓ Uninstalled: {package_name}', 'green'))
        return True


# =============================================================================
# Cloud Storage Integration (HuggingFace Hub)
# =============================================================================


class TaskCloudManager:
    """Manages uploading and downloading task packages from cloud storage."""

    def __init__(self, repo_id: str):
        """
        Initialize TaskCloudManager with a HuggingFace repository.

        Args:
            repo_id: HuggingFace repository ID (e.g., "username/task-assets")
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                'huggingface_hub is required. Install with: pip install huggingface_hub',
            )
        self.repo_id = repo_id
        self.api = HfApi()
        self.cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'vla_arena', 'packages')
        os.makedirs(self.cache_dir, exist_ok=True)

    def upload_with_git(
        self,
        package_path: str,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Upload a task package using Git LFS (fallback method when API fails).

        Args:
            package_path: Path to the .vlap package
            token: HuggingFace API token
            commit_message: Custom commit message

        Returns:
            URL to the uploaded package
        """
        import subprocess
        import tempfile

        if not os.path.exists(package_path):
            raise FileNotFoundError(f'Package not found: {package_path}')

        package_name = Path(package_path).stem

        if commit_message is None:
            commit_message = f'Upload {package_name} task package'

        # Create temporary directory for git operations
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = os.path.join(tmpdir, 'repo')

            print(colored('Using Git LFS upload method...', 'yellow'))

            # Construct clone URL with token if provided
            if token:
                clone_url = f'https://user:{token}@huggingface.co/datasets/{self.repo_id}'
            else:
                clone_url = f'https://huggingface.co/datasets/{self.repo_id}'

            try:
                # Clone repository
                print('  Cloning repository...')
                subprocess.run(
                    ['git', 'clone', clone_url, repo_dir],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Initialize Git LFS
                print('  Setting up Git LFS...')
                subprocess.run(
                    ['git', 'lfs', 'install'],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                )

                # Track .vlap files with LFS
                subprocess.run(
                    ['git', 'lfs', 'track', '*.vlap'],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                )

                # Create packages directory
                packages_dir = os.path.join(repo_dir, 'packages')
                os.makedirs(packages_dir, exist_ok=True)

                # Copy package file
                print(f'  Copying {package_name}.vlap...')
                dest_path = os.path.join(packages_dir, f'{package_name}{PACKAGE_EXTENSION}')
                shutil.copy2(package_path, dest_path)

                # Add .gitattributes if LFS track created it
                gitattributes = os.path.join(repo_dir, '.gitattributes')
                if os.path.exists(gitattributes):
                    subprocess.run(['git', 'add', '.gitattributes'], cwd=repo_dir, check=True)

                # Add package file
                subprocess.run(
                    ['git', 'add', f'packages/{package_name}{PACKAGE_EXTENSION}'],
                    cwd=repo_dir,
                    check=True,
                )

                # Commit
                print('  Creating commit...')
                subprocess.run(
                    ['git', 'commit', '-m', commit_message],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                )

                # Push
                print('  Pushing to HuggingFace...')
                subprocess.run(['git', 'push'], cwd=repo_dir, check=True, capture_output=True)

                url = f'https://huggingface.co/datasets/{self.repo_id}/blob/main/packages/{package_name}{PACKAGE_EXTENSION}'
                print(colored(f'✓ Uploaded via Git LFS: {url}', 'green'))

                return url

            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else str(e)
                raise RuntimeError(f'Git operation failed: {error_msg}')
            except Exception as e:
                raise RuntimeError(f'Upload failed: {e!s}')

    def upload(
        self,
        package_path: str,
        private: bool = False,
        token: Optional[str] = None,
        use_git: bool = False,
    ) -> str:
        """
        Upload a task package to HuggingFace Hub.

        Args:
            package_path: Path to the .vlap package
            private: Whether the repo should be private
            token: HuggingFace API token
            use_git: Force use of Git LFS instead of API

        Returns:
            URL to the uploaded package
        """
        if not os.path.exists(package_path):
            raise FileNotFoundError(f'Package not found: {package_path}')

        # If user explicitly requests Git method, use it directly
        if use_git:
            return self.upload_with_git(package_path, token=token)

        package_name = Path(package_path).stem

        # Try API method first
        try:
            # Create repo if needed
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    private=private,
                    token=token,
                    exist_ok=True,
                )
            except Exception as e:
                print(f'Note: {e}')

            # Upload package
            print('Uploading via HuggingFace API...')
            self.api.upload_file(
                path_or_fileobj=package_path,
                path_in_repo=f'packages/{package_name}{PACKAGE_EXTENSION}',
                repo_id=self.repo_id,
                repo_type='dataset',
                token=token,
            )

            url = f'https://huggingface.co/datasets/{self.repo_id}/blob/main/packages/{package_name}{PACKAGE_EXTENSION}'
            print(colored(f'✓ Uploaded: {url}', 'green'))

            return url

        except Exception as e:
            # If API fails with 403 or storage error, try Git LFS
            error_str = str(e).lower()
            if '403' in error_str or 'storage' in error_str or 'lfs' in error_str:
                print(colored(f'\n⚠ API upload failed: {e!s}', 'yellow'))
                print(colored('Retrying with Git LFS method...\n', 'yellow'))
                return self.upload_with_git(package_path, token=token)
            # For other errors, raise them
            raise

    def list_packages(self) -> List[str]:
        """List available packages in the repository."""
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type='dataset',
            )
            packages = [
                f.replace('packages/', '').replace(PACKAGE_EXTENSION, '')
                for f in files
                if f.startswith('packages/') and f.endswith(PACKAGE_EXTENSION)
            ]
            return packages
        except Exception as e:
            print(f'Error listing packages: {e}')
            return []

    def download(
        self,
        package_name: str,
        output_dir: Optional[str] = None,
        token: Optional[str] = None,
    ) -> str:
        """
        Download a task package from HuggingFace Hub.

        Args:
            package_name: Name of the package to download
            output_dir: Directory to save the package (defaults to cache)
            token: HuggingFace API token

        Returns:
            Path to the downloaded package
        """
        if output_dir is None:
            output_dir = self.cache_dir

        os.makedirs(output_dir, exist_ok=True)

        # Download from HuggingFace
        local_path = os.path.join(output_dir, f'{package_name}{PACKAGE_EXTENSION}')

        self.api.hf_hub_download(
            repo_id=self.repo_id,
            filename=f'packages/{package_name}{PACKAGE_EXTENSION}',
            repo_type='dataset',
            local_dir=output_dir,
            token=token,
        )

        # Move to expected location
        downloaded_path = os.path.join(output_dir, 'packages', f'{package_name}{PACKAGE_EXTENSION}')
        if os.path.exists(downloaded_path) and downloaded_path != local_path:
            shutil.move(downloaded_path, local_path)

        print(colored(f'✓ Downloaded: {local_path}', 'green'))
        return local_path

    def download_and_install(
        self,
        package_name: str,
        overwrite: bool = False,
        token: Optional[str] = None,
    ) -> bool:
        """
        Download and install a package in one step.

        Args:
            package_name: Name of the package
            overwrite: Whether to overwrite existing files
            token: HuggingFace API token

        Returns:
            True if successful
        """
        package_path = self.download(package_name, token=token)
        installer = TaskInstaller()
        return installer.install(package_path, overwrite=overwrite)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface for the asset manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description='VLA-Arena Task Asset Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pack a single task
  python -m vla_arena.vla_arena.utils.asset_manager pack task.bddl -o ./packages
  
  # Pack a task suite
  python -m vla_arena.vla_arena.utils.asset_manager pack-suite robustness_dynamic_distractors -o ./packages
  
  # Install a package
  python -m vla_arena.vla_arena.utils.asset_manager install package.vlap
  
  # Upload to HuggingFace (specify your repo)
  python -m vla_arena.vla_arena.utils.asset_manager upload package.vlap --repo username/task-assets
  
  # Download and install from cloud
  python -m vla_arena.vla_arena.utils.asset_manager download my_task --repo username/task-assets --install
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Pack command
    pack_parser = subparsers.add_parser('pack', help='Pack a single task')
    pack_parser.add_argument('bddl_path', help='Path to BDDL file')
    pack_parser.add_argument('-o', '--output', default='.', help='Output directory')
    pack_parser.add_argument('--init', help='Path to init file')
    pack_parser.add_argument('--name', help='Package name')
    pack_parser.add_argument('--author', default='', help='Author name')
    pack_parser.add_argument('--email', default='', help='Author email')
    pack_parser.add_argument('--description', default='', help='Task description')
    pack_parser.add_argument('--no-assets', action='store_true', help='Skip including assets')

    # Pack suite command
    suite_parser = subparsers.add_parser('pack-suite', help='Pack a task suite')
    suite_parser.add_argument('suite_name', help='Name of the task suite')
    suite_parser.add_argument('-o', '--output', default='.', help='Output directory')
    suite_parser.add_argument('--author', default='', help='Author name')
    suite_parser.add_argument('--email', default='', help='Author email')
    suite_parser.add_argument('--description', default='', help='Suite description')

    # Install command
    install_parser = subparsers.add_parser('install', help='Install a package')
    install_parser.add_argument('package_path', help='Path to .vlap package')
    install_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    install_parser.add_argument('--skip-assets', action='store_true', help='Skip installing assets')
    install_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be installed',
    )

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a package')
    inspect_parser.add_argument('package_path', help='Path to .vlap package')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload to HuggingFace Hub')
    upload_parser.add_argument('package_path', help='Path to .vlap package')
    upload_parser.add_argument(
        '--repo',
        required=True,
        help='HuggingFace repo ID (e.g., username/task-assets)',
    )
    upload_parser.add_argument('--private', action='store_true', help='Make repo private')
    upload_parser.add_argument('--token', help='HuggingFace API token')
    upload_parser.add_argument(
        '--use-git',
        action='store_true',
        help='Use Git LFS instead of API (more reliable for large files)',
    )

    # Download command
    download_parser = subparsers.add_parser('download', help='Download from HuggingFace Hub')
    download_parser.add_argument('package_name', help='Name of the package')
    download_parser.add_argument(
        '--repo',
        required=True,
        help='HuggingFace repo ID (e.g., username/task-assets)',
    )
    download_parser.add_argument('-o', '--output', help='Output directory')
    download_parser.add_argument('--install', action='store_true', help='Install after download')
    download_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files',
    )
    download_parser.add_argument('--token', help='HuggingFace API token')

    # List command
    list_parser = subparsers.add_parser('list', help='List available packages')
    list_parser.add_argument(
        '--repo',
        required=True,
        help='HuggingFace repo ID (e.g., username/task-assets)',
    )

    # Uninstall command
    uninstall_parser = subparsers.add_parser('uninstall', help='Uninstall a package')
    uninstall_parser.add_argument('package_name', help='Name of the package')
    uninstall_parser.add_argument('--remove-assets', action='store_true', help='Also remove assets')

    args = parser.parse_args()

    if args.command == 'pack':
        packager = TaskPackager()
        packager.pack(
            bddl_path=args.bddl_path,
            output_dir=args.output,
            init_path=args.init,
            package_name=args.name,
            author=args.author,
            email=args.email,
            description=args.description,
            include_assets=not args.no_assets,
        )

    elif args.command == 'pack-suite':
        packager = TaskPackager()
        packager.pack_task_suite(
            task_suite_name=args.suite_name,
            output_dir=args.output,
            author=args.author,
            email=args.email,
            description=args.description,
        )

    elif args.command == 'install':
        installer = TaskInstaller()
        installer.install(
            package_path=args.package_path,
            overwrite=args.overwrite,
            skip_assets=args.skip_assets,
            dry_run=args.dry_run,
        )

    elif args.command == 'inspect':
        installer = TaskInstaller()
        manifest = installer.inspect(args.package_path)
        print(f'\nPackage: {manifest.package_name}')
        print(f'Version: {manifest.version}')
        print(f'Task: {manifest.task_name}')
        print(f'Description: {manifest.description}')
        print(f'Author: {manifest.author} <{manifest.email}>')
        print(f'Created: {manifest.created_at}')
        print('\nContents:')
        print(f'  BDDL files: {len(manifest.bddl_files)}')
        print(f'  Init files: {len(manifest.init_files)}')
        print(f'  Assets: {len(manifest.assets)}')
        print(f'  Total size: {manifest.total_size_bytes / 1024 / 1024:.2f} MB')
        print(
            f"\nObjects: {', '.join(manifest.objects[:10])}"
            + (f' (+{len(manifest.objects)-10} more)' if len(manifest.objects) > 10 else ''),
        )

    elif args.command == 'upload':
        cloud = TaskCloudManager(repo_id=args.repo)
        cloud.upload(
            package_path=args.package_path,
            private=args.private,
            token=args.token,
            use_git=args.use_git,
        )

    elif args.command == 'download':
        cloud = TaskCloudManager(repo_id=args.repo)
        if args.install:
            cloud.download_and_install(
                package_name=args.package_name,
                overwrite=args.overwrite,
                token=args.token,
            )
        else:
            cloud.download(
                package_name=args.package_name,
                output_dir=args.output,
                token=args.token,
            )

    elif args.command == 'list':
        cloud = TaskCloudManager(repo_id=args.repo)
        packages = cloud.list_packages()
        if packages:
            print('Available packages:')
            for pkg in packages:
                print(f'  - {pkg}')
        else:
            print('No packages found.')

    elif args.command == 'uninstall':
        installer = TaskInstaller()
        installer.uninstall(
            package_name=args.package_name,
            remove_assets=args.remove_assets,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
