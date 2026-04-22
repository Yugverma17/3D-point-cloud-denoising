

from __future__ import print_function
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import os
import numpy as np
import scipy.spatial as sp
import torch.nn.functional as F
from new_Pointfilter_Utils import pca_alignment, add_noise_to_batch


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    # Detect if it's training or evaluation batch
    is_training = 'noisy_points' in batch[0]

    if is_training:
        # Simple collate for training batches
        noisy_points = [item['noisy_points'] for item in batch]
        gt_points = [item['gt_points'] for item in batch]
        return {
            'noisy_points': torch.stack(noisy_points, dim=0),
            'gt_points': torch.stack(gt_points, dim=0)
        }

    else:
        # Evaluation batch (keep your original padding logic)
        points_per_patch = batch[0]['points'].shape[1] if 'points' in batch[0] else 500
        max_points = max(item['points'].shape[0] for item in batch)

        padded_points = []
        indices_list = []
        masks = []
        extra_data = {}

        for item in batch:
            current_points = item['points'].shape[0]

            # Create mask
            mask = torch.ones(current_points, points_per_patch, dtype=torch.bool)
            if current_points < max_points:
                pad_mask = torch.zeros(max_points - current_points, points_per_patch, dtype=torch.bool)
                mask = torch.cat([mask, pad_mask], dim=0)

            # Pad points
            points_padded = F.pad(item['points'],
                                  (0, 0, 0, 0, 0, max_points - current_points),
                                  mode='constant', value=0)

            padded_points.append(points_padded)
            indices_list.append(item['indices'])
            masks.append(mask)

            # Copy extra fields
            for key in item:
                if key not in ['points', 'indices']:
                    if key not in extra_data:
                        extra_data[key] = []
                    extra_data[key].append(item[key])

        result = {
            'points': torch.cat(padded_points),
            'indices': torch.cat(indices_list),
            'mask': torch.cat(masks)
        }

        for key in extra_data:
            result[key] = default_collate(extra_data[key])

        return result

class PointcloudPatchDataset(data.Dataset):
    def __init__(self, root=None, shapes_list_file=None, patch_radius=0.05, 
                 points_per_patch=500, seed=None, train_state='train', 
                 shape_name=None, noise_type='gaussian', noise_level=0.02,
                 corruption_rate=0.1, use_augmentation=True, scale_range=[0.9, 1.1],
                 gt_exists=None, min_patch_size=10): 
        
        self.root = root
        self.shapes_list_file = shapes_list_file
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.seed = seed
        self.train_state = train_state
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.corruption_rate = corruption_rate
        self.use_augmentation = use_augmentation
        self.scale_range = scale_range
        self.min_patch_size = min_patch_size

        if min_patch_size is None:
            min_patch_size = 10 if train_state == 'evaluation' else 3
        self.min_patch_size = min_patch_size

        if self.seed is None:
            self.seed = np.random.randint(0, 2**10 - 1)
        self.rng = np.random.RandomState(self.seed)

        if gt_exists is not None:
            import warnings
            warnings.warn("gt_exists parameter is deprecated. Use train_state instead.", DeprecationWarning)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.gt_shapes = []
        self.noise_shapes = []
        self.shape_names = []

        if self.train_state == 'evaluation' and shape_name is not None:
            self._load_evaluation_data(shape_name)
        elif self.train_state == 'train':
            self._load_training_data()
   

    def shape_index(self, index):
        """Convert flat index into (shape_index, patch_index) tuple"""
        if not hasattr(self, 'shape_patch_count'):
            raise RuntimeError("Dataset not properly initialized - run _load_training_data or _load_evaluation_data first")
        
        offset = 0
        for shape_idx, patch_count in enumerate(self.shape_patch_count):
            if index < offset + patch_count:
                return shape_idx, index - offset
            offset += patch_count
        raise IndexError(
            f"Index {index} is out of bounds. "
            f"Dataset contains {offset} patches across {len(self.shape_patch_count)} shapes."
        )

    def _load_evaluation_data(self, shape_name):
        """Load single shape for evaluation"""
        pts_path = os.path.join(self.root, shape_name + '.npy')
        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"Point file not found {pts_path}")

        pts = np.load(pts_path)
        kdtree = sp.cKDTree(pts)
        self.noise_shapes.append({'pts': pts, 'kdtree': kdtree})
        self.shape_patch_count.append(pts.shape[0])
        bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
        self.patch_radius_absolute.append(bbdiag * self.patch_radius)
        self.shape_names.append(shape_name)

    def _load_training_data(self):
        """Load training dataset with synthetic noise"""
        with open(os.path.join(self.root, self.shapes_list_file)) as f:
            self.shape_names = [x.strip() for x in f.readlines() if x.strip()]
        
        for shape_name in self.shape_names:
            pts_path = os.path.join(self.root, shape_name + '.npy')
            if not os.path.exists(pts_path):
                continue
                
            pts = np.load(pts_path)
            if pts.shape[0] == 0:
                continue
                
            # Create clean+kdtree and noisy versions
            kdtree = sp.cKDTree(pts)
            self.gt_shapes.append({'pts': pts, 'kdtree': kdtree})
            
            noisy_pts = add_noise_to_batch(
                torch.from_numpy(pts).float(),
                noise_type=self.noise_type,
                noise_level=self.noise_level,
                corruption_rate=self.corruption_rate
            ).numpy()
            
            self.noise_shapes.append({
                'pts': noisy_pts,
                'kdtree': sp.cKDTree(noisy_pts)
            })
            
            self.shape_patch_count.append(pts.shape[0])
            bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.patch_radius)

    def _apply_augmentation(self, patch_pts):
        """Apply random scaling/rotation"""
        if not self.use_augmentation:
            return patch_pts
            
        # Random scaling
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1], size=3)
        patch_pts = patch_pts * scale
        
        # Random z-rotation
        angle = self.rng.uniform(0, 2*np.pi)
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return patch_pts @ rot.T
    
    def __len__(self):
        """
        Returns the total number of patches across all shapes in the dataset.
        """
        # This handles the case where the dataset might be empty
        if not self.shape_patch_count:
            return 0
        
        # The total length is the sum of patches from all shapes
        return sum(self.shape_patch_count)

    # In new_Pointfilter_DataLoader.py

    def __getitem__(self, index):
        shape_ind, patch_ind = self.shape_index(index)
        patch_radius = self.patch_radius_absolute[shape_ind]
        center = self.noise_shapes[shape_ind]['pts'][patch_ind]
        

        if self.train_state == 'train':
            # 1. Get noisy patch points (RANDOM SAMPLING)
            noise_idx = self.noise_shapes[shape_ind]['kdtree'].query_ball_point(center, patch_radius)
            if len(noise_idx) < self.min_patch_size:  # Need at least 3 points for PCA
                return None
            
            # Get ALL noisy points first
            noise_patch = self.noise_shapes[shape_ind]['pts'][noise_idx] - center
            noise_patch = self._apply_augmentation(noise_patch)
            
            try:
                noise_patch, inv_transform = pca_alignment(noise_patch)
            except np.linalg.LinAlgError:
                return None
            
            noise_patch /= patch_radius  # Normalize
            
            # RANDOMLY SAMPLE exactly 500 points (with duplication if needed)
            available_points = noise_patch.shape[0]
            if available_points < self.points_per_patch:
                # Duplicate points if we don't have enough
                repeat_times = (self.points_per_patch // available_points) + 1
                sample_idx = np.tile(np.arange(available_points), repeat_times)[:self.points_per_patch]
            else:
                # Randomly sample without replacement if we have enough points
                sample_idx = self.rng.choice(available_points, size=self.points_per_patch, replace=False)
            
            noise_patch = noise_patch[sample_idx]  # This is now guaranteed to work

            # 2. Get corresponding GT patch (must use same spatial locations)
            gt_idx = self.gt_shapes[shape_ind]['kdtree'].query_ball_point(center, patch_radius)
            if len(gt_idx) < 3:
                return None
            
            gt_patch = self.gt_shapes[shape_ind]['pts'][gt_idx] - center
            gt_patch = np.array(inv_transform @ gt_patch.T).T  # Apply same transform
            gt_patch /= patch_radius  # Same normalization
            
            # Find nearest GT points for our sampled noise points
            gt_kdtree = sp.cKDTree(gt_patch)
            _, gt_sample_idx = gt_kdtree.query(noise_patch, k=1)  # Find closest GT point for each noise point
            
            # Final check to prevent any indexing errors
            gt_sample_idx = np.clip(gt_sample_idx, 0, len(gt_patch) - 1)
            gt_patch = gt_patch[gt_sample_idx]
            
            return {
                    'noisy_points': torch.FloatTensor(noise_patch),
                    'gt_points': torch.FloatTensor(gt_patch)
                }

        # This new part is specifically for evaluation
        elif self.train_state == 'evaluation':
            shape_ind, patch_ind = self.shape_index(index)
            patch_radius = self.patch_radius_absolute[shape_ind]
            center = self.noise_shapes[shape_ind]['pts'][patch_ind]
            
            # 1. Get ALL noisy neighbors in the patch radius
            noise_idx_all = self.noise_shapes[shape_ind]['kdtree'].query_ball_point(center, patch_radius)
            if len(noise_idx_all) < self.min_patch_size:
                return None
            
            # This is the original patch with a VARIABLE number of points
            noise_patch = self.noise_shapes[shape_ind]['pts'][noise_idx_all] - center
            
            try:
                noise_patch, inv_transform = pca_alignment(noise_patch)
            except np.linalg.LinAlgError:
                return None

            noise_patch /= patch_radius

            # 2. <<< --- THE FIX IS HERE --- >>>
            # Now, we sample or pad to get a FIXED number of points for the model
            current_num_points = noise_patch.shape[0]
            if current_num_points > self.points_per_patch:
                # We have more points than needed, so we sample
                sample_idx = self.rng.choice(current_num_points, self.points_per_patch, replace=False)
                sampled_patch = noise_patch[sample_idx]
                # The final indices must correspond to this sample
                final_indices = np.array(noise_idx_all)[sample_idx]
            else:
                # We have fewer points than needed, so we pad by duplicating
                padding_needed = self.points_per_patch - current_num_points
                pad_idx = self.rng.choice(current_num_points, padding_needed, replace=True)
                
                sampled_patch = np.vstack([noise_patch, noise_patch[pad_idx]])
                # The final indices must also be duplicated to match the padding
                final_indices = np.concatenate([noise_idx_all, np.array(noise_idx_all)[pad_idx]])
            
            # 3. Return the final data. The number of points and indices now match.
            return {
                'points': torch.FloatTensor(sampled_patch),
                'indices': torch.LongTensor(final_indices), # Use the correctly sampled indices
                'noise_inv': torch.FloatTensor(inv_transform),
                'noise_disp': torch.FloatTensor(center)
            }
        

class RandomPointcloudPatchSampler:
    """Sampler that randomly selects patches with uniform probability"""
    def __init__(self, data_source, patches_per_shape, seed=None):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed or np.random.randint(0, 2**31-1)
        self.rng = np.random.RandomState(self.seed)
        
    def __iter__(self):
        population = sum(self.data_source.shape_patch_count)
        if population == 0:
            return iter([])
        return iter(self.rng.choice(population, size=min(self.patches_per_shape, population), replace=False))
        
    def __len__(self):
        return min(self.patches_per_shape, sum(self.data_source.shape_patch_count))
