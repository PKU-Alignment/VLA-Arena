# Task Asset Management Guide

VLA-Arena provides a complete asset management system for packaging, sharing, and installing custom tasks and scenes. This guide will help you understand how to use the asset manager to share your custom tasks with the community.

## Table of Contents
1. [Overview](#1-overview)
2. [Package a Single Task](#2-package-a-single-task)
3. [Package a Task Suite](#3-package-a-task-suite)
4. [Inspect a Package](#4-inspect-a-package)
5. [Install a Package](#5-install-a-package)
6. [Upload to Cloud](#6-upload-to-cloud)
7. [Download from Cloud](#7-download-from-cloud)
8. [Uninstall a Package](#8-uninstall-a-package)
9. [Package Structure](#9-package-structure)
10. [Troubleshooting](#10-troubleshooting)

## 1. Overview

The Asset Management system provides a complete workflow for sharing custom tasks:

```
Design → Pack → Upload → Download → Install → Use
```

### Key Features
- **Automatic Dependency Detection**: Automatically finds all required assets (objects, textures, meshes, Problem classes, scene XMLs)
- **Self-Contained Packages**: All dependencies are packaged into a single `.vlap` file
- **Cloud Integration**: Upload to HuggingFace Hub or your own repository
- **Version Control**: Manifest with checksums ensures integrity
- **Conflict Detection**: Warns about existing files before installation

### What Gets Packaged?
- **BDDL Files**: Task definitions with objects, fixtures, and goals
- **Init Files**: Initial state files (`.pruned_init`)
- **Object Assets**: 3D models, textures, collision meshes
- **Problem Classes**: Custom Python environment definitions
- **Scene XMLs**: MuJoCo scene descriptions with textures and meshes
- **Metadata**: Author info, descriptions, checksums

## 2. Package a Single Task

To package a single task, you need the BDDL file path:

```bash
python scripts/manage_assets.py pack <bddl_file_path> \
    -o ./packages \
    --author "Your Name" \
    --email "your.email@example.com" \
    --description "A custom pick and place task"
```

### Example
```bash
python scripts/manage_assets.py pack \
    vla_arena/vla_arena/bddl_files/robustness_static_distractors/level_0/pick_up_the_banana_and_put_it_on_the_plate.bddl \
    -o ./packages \
    --author "VLA-Arena Team" \
    --description "Pick banana task with static distractors"
```

### Options
- `-o, --output`: Output directory (default: current directory)
- `--name`: Custom package name (default: derived from BDDL filename)
- `--author`: Author name
- `--email`: Author email
- `--description`: Task description
- `--init`: Manually specify init file (auto-detected by default)
- `--no-assets`: Skip including assets (for testing)

## 3. Package a Task Suite

To package an entire task suite with multiple tasks:

```bash
python scripts/manage_assets.py pack-suite <suite_name> \
    -o ./packages \
    --author "Your Name" \
    --description "Task suite description"
```

### Example
```bash
python scripts/manage_assets.py pack-suite robustness_static_distractors \
    -o ./packages \
    --author "VLA-Arena Team" \
    --description "Robustness test with static distractors"
```

## 4. Inspect a Package

Before installing, you can inspect a package to see its contents:

```bash
python scripts/manage_assets.py inspect <package_path>
```

### Example
```bash
python scripts/manage_assets.py inspect ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap
```

## 5. Install a Package

To install a package into your VLA-Arena installation:

```bash
python scripts/manage_assets.py install <package_path>
```

### Example
```bash
python scripts/manage_assets.py install ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap
```

### Options
- `--overwrite`: Overwrite existing files (default: skip)
- `--skip-assets`: Install only BDDL and init files, skip assets
- `--dry-run`: Show what would be installed without actually installing

### Conflict Handling
If files already exist, the installer will warn you:
```
⚠ Conflicts detected:
  - Asset: ceramic_plate (already exists)
  - Asset: banana (already exists)
  - Asset: toy_train (already exists)

Use --overwrite to replace existing files.
```

## 6. Upload to Cloud

Upload a package to HuggingFace Hub (or your own repository):

```bash
python scripts/manage_assets.py upload <package_path> \
    --repo username/repository-name
```

### Prerequisites
1. **HuggingFace Account**: Sign up at https://huggingface.co
2. **Create a Repository**:
   - Go to https://huggingface.co/new-dataset
   - Create a dataset repository (e.g., `username/vla-arena-tasks`)
3. **Get Access Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a token with **write** permissions
4. **Install Git LFS** (support fallback method):

### Example
```bash
# Set token as environment variable (recommended)
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Upload package
python scripts/manage_assets.py upload \
    ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap \
    --repo username/vla-arena-tasks
```

### With Token Argument
```bash
python scripts/manage_assets.py upload \
    ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap \
    --repo username/vla-arena-tasks \
    --token hf_your_token_here \
    --private  # Make repository private (optional)
```

### Output
```
Uploading via HuggingFace API...
✓ Uploaded: https://huggingface.co/datasets/username/vla-arena-tasks/blob/main/packages/pick_up_the_banana_and_put_it_on_the_plate.vlap
```

### Automatic Fallback
If API upload fails (e.g., due to rate limits), the system automatically retries with Git LFS:
```
⚠ API upload failed: 403 Forbidden
Retrying with Git LFS method...

Using Git LFS upload method...
  Cloning repository...
  Setting up Git LFS...
  Copying pick_up_the_banana_and_put_it_on_the_plate.vlap...
  Creating commit...
  Pushing to HuggingFace...
✓ Uploaded via Git LFS: https://huggingface.co/...
```

## 7. Download from Cloud

Download and optionally install a package from HuggingFace Hub:

```bash
python scripts/manage_assets.py download <package_name> \
    --repo username/repository-name \
    --install
```

### Example: Download Only
```bash
python scripts/manage_assets.py download pick_up_the_banana_and_put_it_on_the_plate \
    --repo username/vla-arena-tasks \
    -o ./downloaded_packages
```

### Example: Download and Install
```bash
python scripts/manage_assets.py download pick_up_the_banana_and_put_it_on_the_plate \
    --repo username/vla-arena-tasks \
    --install \
    --overwrite
```

### List Available Packages
```bash
python scripts/manage_assets.py list --repo username/vla-arena-tasks
```

### Output
```
Available packages:
  - pick_up_the_banana_and_put_it_on_the_plate
  - robustness_static_distractors
  - long_horizon
```

## 8. Uninstall a Package

To remove an installed package:

```bash
python scripts/manage_assets.py uninstall <package_name>
```

### Example
```bash
python scripts/manage_assets.py uninstall pick_up_the_banana_and_put_it_on_the_plate
```

### Output
```
Uninstalling: pick_up_the_banana_and_put_it_on_the_plate
  ✓ Removed BDDL files
  ✓ Removed init files
  ⚠ Assets not removed (shared with other tasks)

✓ Uninstalled: pick_up_the_banana_and_put_it_on_the_plate
```

### Options
- `--remove-assets`: Also remove associated asset files (use with caution, as assets may be shared)

## 9. Package Structure

A `.vlap` package is a ZIP file with the following structure:

```
package_name.vlap
├── manifest.json              # Metadata and checksums
├── bddl_files/               # Task definitions
│   └── task_name.bddl
├── init_files/               # Initial states
│   └── task_name.pruned_init
├── problems/                 # Custom Problem classes
│   └── tabletop_manipulation.py
└── assets/                   # All assets
    ├── stable_scanned_objects/
    │   └── banana/
    │       ├── banana.xml
    │       ├── visual/
    │       │   ├── model_normalized_0.obj
    │       │   └── image0.png
    │       └── collision/
    │           └── model_normalized_collision_22.obj
    ├── scenes/               # Scene XMLs
    │   └── tabletop_warm_style.xml
    └── textures/             # Scene textures
        └── martin_novak_wood_table.png
```

### Manifest Format
```json
{
  "package_name": "pick_up_the_banana_and_put_it_on_the_plate",
  "version": "1.0.0",
  "task_name": "Tabletop_Manipulation",
  "description": "Pick banana task with static distractors",
  "author": "Alice Smith",
  "email": "alice@example.com",
  "created_at": "2024-12-15T10:30:00",
  "bddl_files": ["pick_up_the_banana_and_put_it_on_the_plate.bddl"],
  "init_files": ["pick_up_the_banana_and_put_it_on_the_plate.pruned_init"],
  "problem_files": ["tabletop_manipulation.py"],
  "scene_files": ["tabletop_warm_style.xml"],
  "assets": [
    {
      "object_type": "banana",
      "relative_path": "stable_scanned_objects/banana/banana.xml",
      "checksum": "a1b2c3d4..."
    }
  ],
  "objects": ["banana", "ceramic_plate", "toy_train"],
  "total_size_bytes": 7549747
}
```

## 10. Troubleshooting

### Issue: "Unknown object type" warning
**Symptom**: Warning about fixtures like `table` or `floor`
```
[Warning] Unknown object type: table
```
**Solution**: This is normal. Fixtures like `table`, `floor`, and `main_table` are part of the environment arena and are not packaged separately.

### Issue: NumPy compatibility error
**Symptom**: `AttributeError: _ARRAY_API not found`
```bash
AttributeError: numpy.core.multiarray has no attribute _ARRAY_API
```
**Solution**: Downgrade NumPy to a compatible version:
```bash
pip install "numpy>=1.24,<2"
```

### Issue: HuggingFace 403 Forbidden
**Symptom**: Upload fails with "403 Forbidden: Your storage patterns tripped our internal systems"
```
403 Forbidden: Your storage patterns tripped our internal systems!
```
**Solutions**:
1. **Wait and Retry**: HuggingFace has rate limits; wait 10-30 minutes
2. **Contact Support**: Email website@huggingface.co with your use case
3. **Install Git LFS**: The system will automatically retry with Git LFS
4. **Use Personal Repo**: Upload to your personal account instead of organization

### Issue: Git LFS not found
**Symptom**: `git: 'lfs' is not a git command`
**Solution**: Install Git LFS:
```bash
# macOS
brew install git-lfs
git lfs install

# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# Windows
# Download from https://git-lfs.github.com/
```

### Issue: Package installation conflicts
**Symptom**: Files already exist
```
⚠ Conflicts detected:
  - Asset: banana (already exists)
  - Asset: ceramic_plate (already exists)

Use --overwrite to replace existing files.
```
**Solutions**:
1. **Skip conflicts**: Default behavior, existing files are preserved
2. **Force overwrite**: Use `--overwrite` flag
3. **Dry run first**: Use `--dry-run` to see what would be installed

### Issue: Missing dependencies
**Symptom**: Task fails to run after installation
**Solution**: Ensure robosuite and all dependencies are installed:
```bash
cd vla_arena
pip install -e .
```

### Issue: Path resolution errors
**Symptom**: Cannot find BDDL or asset files
**Solution**: VLA-Arena now uses relative paths. Ensure you're running commands from the project root:
```bash
cd /path/to/VLA-Arena
python scripts/manage_assets.py <command>
```

## Notes and Best Practices

### When Packaging
- **Test First**: Always test your task locally before packaging
- **Descriptive Names**: Use clear, descriptive package names
- **Author Info**: Include author and contact information
- **Documentation**: Add a description explaining what the task does

### When Uploading
- **Token Security**: Never commit tokens to version control
- **Use Environment Variables**: Export token as `HUGGING_FACE_HUB_TOKEN`
- **Repository Naming**: Use descriptive repo names like `username/vla-arena-tasks`
- **Private vs Public**: Consider whether your tasks should be public

### When Installing
- **Inspect First**: Use `inspect` command before installing
- **Backup**: Backup your existing tasks before using `--overwrite`
- **Dry Run**: Use `--dry-run` to preview changes
- **Assets**: Be cautious with `--remove-assets` when uninstalling

### Package Size Considerations
- Single tasks: typically 5-10 MB
- Task suites: can be 30-60 MB or more
- HuggingFace free tier: 50 GB storage limit
- Consider splitting large suites into smaller packages

## Example Workflow

Here's a complete workflow from creating a custom task to sharing it:

```bash
# 1. Create your custom task (BDDL file, Problem class, assets)
# ... (see Scene Construction guide)

# 2. Package your task
python scripts/manage_assets.py pack \
    vla_arena/vla_arena/bddl_files/my_custom_task/my_task.bddl \
    -o ./packages \
    --author "Your Name" \
    --email "you@example.com" \
    --description "My awesome custom task"

# 3. Inspect the package
python scripts/manage_assets.py inspect ./packages/my_task.vlap

# 4. Upload to HuggingFace
export HUGGING_FACE_HUB_TOKEN="your_token"
python scripts/manage_assets.py upload \
    ./packages/my_task.vlap \
    --repo yourusername/vla-arena-custom-tasks

# 5. Share the download command with others
echo "To use my task, run:"
echo "python scripts/manage_assets.py download my_task --repo yourusername/vla-arena-custom-tasks --install"
```

Others can now install your task with one command:
```bash
python scripts/manage_assets.py download my_task \
    --repo yourusername/vla-arena-custom-tasks \
    --install
```

## Additional Resources

- [Scene Construction Guide](scene_construction.md) - How to create custom tasks
- [Data Collection Guide](data_collection.md) - How to collect demonstrations
- [Evaluation Guide](evaluation.md) - How to evaluate policies
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index) - Cloud storage
