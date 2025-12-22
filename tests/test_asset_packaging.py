"""
Test asset packaging workflow

Tests the complete workflow:
1. Pack individual task
2. Pack task suite
3. Inspect package
4. Install package
5. Uninstall package
"""

import os
import shutil
import tempfile

import pytest

from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.utils.asset_manager import TaskInstaller, TaskPackager


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp(prefix='vla_arena_test_')
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_bddl_file():
    """Get a sample BDDL file for testing"""
    bddl_root = get_vla_arena_path('bddl_files')

    # Find first available BDDL file
    for suite_dir in os.listdir(bddl_root):
        suite_path = os.path.join(bddl_root, suite_dir)
        if os.path.isdir(suite_path) and not suite_dir.startswith('.'):
            for root, dirs, files in os.walk(suite_path):
                bddl_files = [f for f in files if f.endswith('.bddl')]
                if bddl_files:
                    return os.path.join(root, bddl_files[0])

    pytest.skip('No BDDL files found')


class TestTaskPackaging:
    """Test task packaging functionality"""

    def test_pack_single_task(self, temp_output_dir, sample_bddl_file):
        """Test packing a single task"""
        packager = TaskPackager()

        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
            description='Test package',
        )

        # Verify package was created
        assert os.path.exists(package_path)
        assert package_path.endswith('.vlap')

        # Verify package size is reasonable (should be < 100MB for a single task)
        size_mb = os.path.getsize(package_path) / 1024 / 1024
        assert size_mb < 100, f'Package too large: {size_mb:.2f} MB'

    def test_pack_task_suite(self, temp_output_dir):
        """Test packing a task suite"""
        packager = TaskPackager()

        # Get first available task suite
        bddl_root = get_vla_arena_path('bddl_files')
        suite_name = None
        for item in os.listdir(bddl_root):
            if os.path.isdir(
                os.path.join(bddl_root, item),
            ) and not item.startswith('.'):
                suite_name = item
                break

        if not suite_name:
            pytest.skip('No task suites found')

        package_path = packager.pack_task_suite(
            task_suite_name=suite_name,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Verify package was created
        assert os.path.exists(package_path)
        assert suite_name in package_path

    def test_inspect_package(self, temp_output_dir, sample_bddl_file):
        """Test inspecting a package"""
        packager = TaskPackager()
        installer = TaskInstaller()

        # Create a package
        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Inspect the package
        manifest = installer.inspect(package_path)

        # Verify manifest contents
        assert manifest.package_name
        assert manifest.author == 'Test User'
        assert len(manifest.bddl_files) > 0
        assert len(manifest.assets) >= 0  # May be 0 if no assets

    def test_conflict_detection(self, temp_output_dir, sample_bddl_file):
        """Test conflict detection"""
        packager = TaskPackager()
        installer = TaskInstaller()

        # Create a package
        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Check for conflicts (should find some if assets already installed)
        conflicts = installer.check_conflicts(package_path)

        # Verify conflict detection structure
        assert 'assets' in conflicts
        assert 'bddl_files' in conflicts
        assert 'init_files' in conflicts
        assert isinstance(conflicts['assets'], list)

    def test_dry_run_install(self, temp_output_dir, sample_bddl_file):
        """Test dry-run installation"""
        packager = TaskPackager()
        installer = TaskInstaller()

        # Create a package
        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Dry run with skip_existing_assets should always succeed
        result = installer.install(
            package_path,
            dry_run=True,
            skip_existing_assets=True,
        )
        assert result is True


class TestAssetManager:
    """Test asset manager functionality"""

    def test_package_format(self, temp_output_dir, sample_bddl_file):
        """Test that package format is correct"""
        import zipfile

        packager = TaskPackager()
        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Verify it's a valid ZIP file
        assert zipfile.is_zipfile(package_path)

        # Verify manifest exists
        with zipfile.ZipFile(package_path, 'r') as zf:
            files = zf.namelist()
            manifest_files = [f for f in files if 'manifest.json' in f]
            assert (
                len(manifest_files) > 0
            ), 'manifest.json not found in package'

    def test_package_contains_bddl(self, temp_output_dir, sample_bddl_file):
        """Test that package contains BDDL files"""
        import zipfile

        packager = TaskPackager()
        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Verify BDDL files are included
        with zipfile.ZipFile(package_path, 'r') as zf:
            files = zf.namelist()
            bddl_files = [f for f in files if f.endswith('.bddl')]
            assert len(bddl_files) > 0, 'No BDDL files found in package'

    def test_skip_existing_assets(self, temp_output_dir, sample_bddl_file):
        """Test skip_existing_assets functionality"""
        packager = TaskPackager()
        installer = TaskInstaller()

        # Create a package
        package_path = packager.pack(
            bddl_path=sample_bddl_file,
            output_dir=temp_output_dir,
            author='Test User',
        )

        # Test with skip_existing_assets=True should not fail on conflicts
        result = installer.install(
            package_path,
            skip_existing_assets=True,
            dry_run=True,
        )
        assert result is True


class TestPathResolution:
    """Test path resolution functionality"""

    def test_get_vla_arena_path(self):
        """Test VLA-Arena path resolution"""
        # Test all standard paths
        paths_to_test = [
            'assets',
            'bddl_files',
            'init_states',
            'benchmark_root',
        ]

        for key in paths_to_test:
            path = get_vla_arena_path(key)
            assert path is not None
            assert isinstance(path, str)
            # Path should be absolute
            assert os.path.isabs(path)

    def test_assets_directory_exists(self):
        """Test that assets directory exists"""
        assets_path = get_vla_arena_path('assets')
        assert os.path.exists(assets_path)
        assert os.path.isdir(assets_path)

    def test_bddl_directory_exists(self):
        """Test that BDDL directory exists"""
        bddl_path = get_vla_arena_path('bddl_files')
        assert os.path.exists(bddl_path)
        assert os.path.isdir(bddl_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
