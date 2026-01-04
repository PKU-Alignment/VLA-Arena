"""
Test download_tasks.py script functionality
"""

import argparse
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_task_cloud_manager():
    """Mock TaskCloudManager"""
    with patch('scripts.download_tasks.TaskCloudManager') as mock:
        yield mock


@pytest.fixture
def mock_task_installer():
    """Mock TaskInstaller"""
    with patch('scripts.download_tasks.TaskInstaller') as mock:
        yield mock


class TestDownloadTasksSingleInstall:
    """Test installing a single task"""

    def test_install_single_task(self, mock_task_cloud_manager, mock_task_installer):
        """Test installing a single task suite"""
        from scripts.download_tasks import install_task

        # Mock successful installation
        mock_cloud = Mock()
        mock_cloud.download_and_install.return_value = True
        mock_task_cloud_manager.return_value = mock_cloud

        result = install_task(
            task_name='test_task',
            repo_id='test/repo',
            token=None,
            overwrite=False,
            skip_existing_assets=False,
        )

        assert result is True
        mock_cloud.download_and_install.assert_called_once_with(
            package_name='test_task',
            overwrite=False,
            skip_existing_assets=False,
            token=None,
        )


class TestDownloadTasksMultipleInstall:
    """Test installing multiple tasks"""

    def test_install_multiple_tasks_success(
        self, mock_task_cloud_manager, mock_task_installer
    ):
        """Test installing multiple task suites successfully"""
        from scripts.download_tasks import install_task

        # Mock successful installations
        mock_cloud = Mock()
        mock_cloud.download_and_install.return_value = True
        mock_task_cloud_manager.return_value = mock_cloud

        task_names = ['task1', 'task2', 'task3']
        successful = []
        failed = []

        for task_name in task_names:
            success = install_task(
                task_name=task_name,
                repo_id='test/repo',
                token=None,
                overwrite=False,
                skip_existing_assets=True,  # Important for multiple installs
            )
            if success:
                successful.append(task_name)
            else:
                failed.append(task_name)

        assert len(successful) == 3
        assert len(failed) == 0
        assert mock_cloud.download_and_install.call_count == 3

    def test_install_multiple_tasks_partial_failure(
        self, mock_task_cloud_manager, mock_task_installer
    ):
        """Test installing multiple tasks with some failures"""
        from scripts.download_tasks import install_task

        # Mock: first succeeds, second fails, third succeeds
        mock_cloud = Mock()
        mock_cloud.download_and_install.side_effect = [True, False, True]
        mock_task_cloud_manager.return_value = mock_cloud

        task_names = ['task1', 'task2', 'task3']
        successful = []
        failed = []

        for task_name in task_names:
            success = install_task(
                task_name=task_name,
                repo_id='test/repo',
                token=None,
                overwrite=False,
                skip_existing_assets=True,
            )
            if success:
                successful.append(task_name)
            else:
                failed.append(task_name)

        assert len(successful) == 2
        assert len(failed) == 1
        assert 'task2' in failed
        assert 'task1' in successful
        assert 'task3' in successful


class TestDownloadTasksCLI:
    """Test CLI argument parsing"""

    def test_install_command_accepts_multiple_tasks(self):
        """Test that install command accepts multiple task names"""
        from scripts.download_tasks import main

        with patch('sys.argv', [
            'download_tasks.py',
            'install',
            'task1',
            'task2',
            'task3',
            '--repo',
            'test/repo',
        ]):
            with patch('scripts.download_tasks.install_task') as mock_install:
                mock_install.return_value = True
                try:
                    main()
                except SystemExit:
                    pass

                # Verify install_task was called 3 times
                assert mock_install.call_count == 3

    def test_install_command_with_skip_existing_assets(self):
        """Test that skip-existing-assets flag works with multiple tasks"""
        from scripts.download_tasks import main

        with patch('sys.argv', [
            'download_tasks.py',
            'install',
            'task1',
            'task2',
            '--repo',
            'test/repo',
            '--skip-existing-assets',
        ]):
            with patch('scripts.download_tasks.install_task') as mock_install:
                mock_install.return_value = True
                try:
                    main()
                except SystemExit:
                    pass

                # Verify skip_existing_assets was passed correctly
                for call in mock_install.call_args_list:
                    assert call[1]['skip_existing_assets'] is True


class TestListTasks:
    """Test listing available tasks"""

    def test_list_available_tasks(self, mock_task_cloud_manager):
        """Test listing available task suites"""
        from scripts.download_tasks import list_available_tasks

        # Mock package list
        mock_cloud = Mock()
        mock_cloud.list_packages.return_value = [
            'task1',
            'task2',
            'task3',
        ]
        mock_task_cloud_manager.return_value = mock_cloud

        packages = list_available_tasks(repo_id='test/repo')

        assert len(packages) == 3
        assert 'task1' in packages
        mock_cloud.list_packages.assert_called_once()


class TestInstallAll:
    """Test install-all functionality"""

    @patch('builtins.input', return_value='y')
    def test_install_all_tasks(
        self,
        mock_input,
        mock_task_cloud_manager,
        mock_task_installer,
    ):
        """Test installing all task suites"""
        from scripts.download_tasks import install_all_tasks

        # Mock task list
        mock_cloud = Mock()
        mock_cloud.list_packages.return_value = ['task1', 'task2']
        mock_cloud.download_and_install.return_value = True
        mock_task_cloud_manager.return_value = mock_cloud

        with patch('scripts.download_tasks.list_available_tasks') as mock_list:
            mock_list.return_value = ['task1', 'task2']

            install_all_tasks(
                repo_id='test/repo',
                token=None,
                overwrite=False,
            )

            # Should call download_and_install for each task
            assert mock_cloud.download_and_install.call_count >= 2

    @patch('builtins.input', return_value='n')
    def test_install_all_tasks_cancelled(
        self,
        mock_input,
        mock_task_cloud_manager,
    ):
        """Test cancelling install-all"""
        from scripts.download_tasks import install_all_tasks

        mock_cloud = Mock()
        mock_cloud.list_packages.return_value = ['task1', 'task2']
        mock_task_cloud_manager.return_value = mock_cloud

        with patch('scripts.download_tasks.list_available_tasks') as mock_list:
            mock_list.return_value = ['task1', 'task2']

            install_all_tasks(
                repo_id='test/repo',
                token=None,
                overwrite=False,
            )

            # Should not call download_and_install
            mock_cloud.download_and_install.assert_not_called()


class TestGetInstalledTasks:
    """Test getting installed tasks"""

    def test_get_installed_tasks(self):
        """Test retrieving list of installed tasks"""
        from scripts.download_tasks import get_installed_tasks

        # This should return a list (might be empty in test environment)
        tasks = get_installed_tasks()
        assert isinstance(tasks, list)

    def test_show_installed_tasks(self, capsys):
        """Test showing installed tasks"""
        from scripts.download_tasks import show_installed_tasks

        with patch('scripts.download_tasks.get_installed_tasks') as mock_get:
            mock_get.return_value = ['task1', 'task2']

            show_installed_tasks()

            captured = capsys.readouterr()
            # Should show the tasks
            assert 'task1' in captured.out or 'task2' in captured.out

