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
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""


# fmt: off
OXE_NAMED_MIXTURES: dict[str, list[tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    'bridge': [
        ('bridge_oxe', 1.0),                                      # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],

    'droid': [
        ('droid', 1.0),
    ],

    # === Human-data Only ===
    'Ego4D': [
        ('ego4d_split_1', 1.0),
        ('ego4d_split_2', 1.0),
        ('ego4d_split_3', 1.0),
        ('ego4d_split_4', 1.0),
    ],


    'roboset': [
        ('roboset', 1.0),
    ],

    'stanford_robocook_converted_externally_to_rlds': [
        ('stanford_robocook_converted_externally_to_rlds', 1.0),
    ],

    # === [Moderate-Scale] Bridge++ Mixtures ===
    'bridge_rt_1': [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ('bridge_orig', 1.0),                                   # Original Version of Bridge V2 from Project Website
        ('fractal20220817_data', 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    'rt_1': [
        ('fractal20220817_data', 1.0),
    ],

    # === UniVLA Magic Soup+ ===
    'omni_magic_soup_plus': [
        ('fractal20220817_data', 0.5),
        ('kuka', 0.1),
        ('bridge_oxe', 1.0),
        ('taco_play', 2.0),
        ('jaco_play', 1.0),
        ('berkeley_cable_routing', 1.0),
        ('roboturk', 2.0),
        ('viola', 2.0),
        ('berkeley_autolab_ur5', 2.0),
        ('toto', 1.0),
        ('language_table', 0.1),
        ('stanford_hydra_dataset_converted_externally_to_rlds', 2.0),
        ('austin_buds_dataset_converted_externally_to_rlds', 1.0),
        ('nyu_franka_play_dataset_converted_externally_to_rlds', 3.0),
        ('furniture_bench_dataset_converted_externally_to_rlds', 0.1),
        ('ucsd_kitchen_dataset_converted_externally_to_rlds', 2.0),
        ('austin_sailor_dataset_converted_externally_to_rlds', 1.0),
        ('austin_sirius_dataset_converted_externally_to_rlds', 1.0),
        ('dlr_edan_shared_control_converted_externally_to_rlds', 1.0),
        ('iamlab_cmu_pickup_insert_converted_externally_to_rlds', 1.0),
        ('utaustin_mutex', 1.0),
        ('berkeley_fanuc_manipulation', 2.0),
        ('cmu_stretch', 1.0),
        ('bc_z', 0.2),
        ('fmb', 1.0),
        ('dobbe', 0.2),
        ## Datasets for Navigation
        ('berkeley_gnm_recon', 1.0),
        ('berkeley_gnm_cory_hall', 1.0),
        ('berkeley_gnm_sac_son', 1.0),
    ],

    # === UniVLA Magic Soup++ ===
    'omni_magic_soup_plus_plus': [
        ('fractal20220817_data', 0.5),
        ('kuka', 0.1),
        ('bridge_oxe', 1.0),
        ('taco_play', 2.0),
        ('jaco_play', 1.0),
        ('berkeley_cable_routing', 1.0),
        ('roboturk', 2.0),
        ('viola', 2.0),
        ('berkeley_autolab_ur5', 2.0),
        ('toto', 1.0),
        ('language_table', 0.1),
        ('stanford_hydra_dataset_converted_externally_to_rlds', 2.0),
        ('austin_buds_dataset_converted_externally_to_rlds', 1.0),
        ('nyu_franka_play_dataset_converted_externally_to_rlds', 3.0),
        ('furniture_bench_dataset_converted_externally_to_rlds', 0.1),
        ('ucsd_kitchen_dataset_converted_externally_to_rlds', 2.0),
        ('austin_sailor_dataset_converted_externally_to_rlds', 1.0),
        ('austin_sirius_dataset_converted_externally_to_rlds', 1.0),
        ('dlr_edan_shared_control_converted_externally_to_rlds', 1.0),
        ('iamlab_cmu_pickup_insert_converted_externally_to_rlds', 1.0),
        ('utaustin_mutex', 1.0),
        ('berkeley_fanuc_manipulation', 2.0),
        ('cmu_stretch', 1.0),
        ('bc_z', 0.2),
        ('fmb', 1.0),
        ('dobbe', 0.2),
        ## Datasets for Navigation
        ('berkeley_gnm_recon', 2.0),
        ('berkeley_gnm_cory_hall', 2.0),
        ('berkeley_gnm_sac_son', 2.0),
        ## Human Datasets
        ('ego4d_split_1', 1.0),
        ('ego4d_split_2', 1.0),
        ('ego4d_split_3', 1.0),
        ('ego4d_split_4', 1.0),
    ],

    # === T-DROID Dataset ===
    'tdroid_carrot_in_bowl': [
        ('tdroid_carrot_in_bowl', 1.0),
    ],
    'tdroid_pour_corn_in_pot': [
        ('tdroid_pour_corn_in_pot', 1.0),
    ],
    'tdroid_flip_pot_upright': [
        ('tdroid_flip_pot_upright', 1.0),
    ],
    'tdroid_move_object_onto_plate': [
        ('tdroid_move_object_onto_plate', 1.0),
    ],
    'tdroid_knock_object_over': [
        ('tdroid_knock_object_over', 1.0),
    ],
    'tdroid_cover_object_with_towel': [
        ('tdroid_cover_object_with_towel', 1.0),
    ],

    # === DROID Finetuning Datasets ===
    'droid_wipe': [
        ('droid_wipe', 1.0),
    ],

    # === LIBERO Datasets (Modified Versions) ===
    'libero_spatial_no_noops': [
        ('libero_spatial_no_noops', 1.0),
    ],
    'libero_object_no_noops': [
        ('libero_object_no_noops', 1.0),
    ],
    'libero_goal_no_noops': [
        ('libero_goal_no_noops', 1.0),
    ],
    'libero_10_no_noops': [
        ('libero_10_no_noops', 1.0),
    ],
    'libero_10_no_noops_mini': [
        ('libero_10_no_noops_mini', 1.0),
    ],
    'libero_goal_no_noops_mini': [
        ('libero_goal_no_noops_mini', 1.0),
    ],
    'libero_goal_no_noops_half': [
        ('libero_goal_no_noops_half', 1.0),
    ],
    'libero_10_no_noops_half': [
        ('libero_10_no_noops_half', 1.0),
    ],
    'libero_goal_no_noops_quad': [
        ('libero_goal_no_noops_quad', 1.0),
    ],
    'libero_10_no_noops_quad': [
        ('libero_10_no_noops_quad', 1.0),
    ],
    'libero_combined': [
        ('libero_combined', 1.0),
    ],
}
# fmt: on
