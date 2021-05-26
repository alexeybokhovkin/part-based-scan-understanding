import os
import json
from argparse import Namespace
import random
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import pytorch_lightning as pl

from unet3d.model import GeoEncoder, HierarchicalDecoder
from unet3d.hierarchy import Tree
from datasets.shapenet import generate_scannet_allshapes_datasets, generate_scannet_allshapes_rot_datasets
from utils.gnn import collate_feats


class Unet3DGNNPartnetLightning(pl.LightningModule):
    def __init__(self, hparams):
        super(Unet3DGNNPartnetLightning, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams
        config = hparams.__dict__
        self.config = config
        for _k, _v in self.config.items():
            if _v is None:
                if _k == "gpus":
                    self.config[_k] = "cpu"
                else:
                    self.config[_k] = "null"

        # FOR TRAINING WITHOUT ROTATIONS OR INFERENCE
        datasets = generate_scannet_allshapes_datasets(**config)
        # FOR TRAINING WITH ROTATIONS
        # datasets = generate_scannet_allshapes_rot_datasets(**config)

        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']

        Tree.load_category_info(config['hierarchies'])
        self.encoder = GeoEncoder(**config)
        self.decoder = HierarchicalDecoder(**config)
        if self.config['encode_mask']:
            self.mask_encoder = GeoEncoder(**config)

        random.seed(config['manual_seed'])
        np.random.seed(config['manual_seed'])
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        with open(os.path.join(config['checkpoint_dir'], config['model'], config['version'], 'config.json'), 'w') as f:
            json.dump(self.config, f)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = StepLR(optimizer, gamma=self.config['gamma'],
                           step_size=self.config['decay_every'])
        return [optimizer], [scheduler]

    def forward(self, batch):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]

        x_roots = []
        mask_codes, mask_features = [], []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            if self.config['encode_mask']:
                mask_code, mask_feature = self.mask_encoder.root_latents(scannet_geos[i][None, ...])
                mask_codes += [mask_code]
                mask_features += [mask_feature]
            else:
                mask_codes += [None]
                mask_features += [None]

            x_roots += [x_root]

        all_losses = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            output = self.decoder.structure_recon_loss(x_root, gt_tree,
                                                       mask_code=mask_codes[i],
                                                       mask_feature=mask_features[i],
                                                       scan_geo=scannet_geos[i][None, ...],
                                                       encoder_features=encoder_features[i],
                                                       rotation=rotations[i]
                                                       )
            object_losses = output[0]
            all_losses += [object_losses]

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0}

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss

        del all_losses

        return losses

    def inference(self, batch):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]

        x_roots = []
        mask_codes, mask_features = [], []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            if self.config['enc_hier']:
                gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
                x_root = self.model.root_latents(gt_tree)
            else:
                x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
                encoder_features += [feature]
            if self.config['encode_mask']:
                mask_code, mask_feature = self.mask_encoder.root_latents(scannet_geos[i][None, ...])
                mask_codes += [mask_code]
                mask_features += [mask_feature]
            else:
                mask_codes += [None]
                mask_features += [None]
            x_roots += [x_root]

        all_losses = []
        predicted_trees = []
        all_priors = []
        all_leaves_geos = []
        pred_rotations = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            mask = masks[i]
            object_losses, all_geos, all_leaf_geos, all_S_priors = self.decoder.structure_recon_loss(x_root, gt_tree,
                                                                                                     mask_code=mask_codes[i],
                                                                                                     mask_feature=mask_features[i],
                                                                                                     scan_geo=scannet_geos[i][None, ...],
                                                                                                     encoder_features=encoder_features[i],
                                                                                                     rotation=rotations[i]
                                                                                                     )

            predicted_tree, S_priors, pred_rotation = self.decoder(x_root,
                                                                   mask_code=mask_codes[i],
                                                                   mask_feature=mask_features[i],
                                                                   scan_geo=scannet_geos[i][None, ...],
                                                                   full_label=gt_tree.root.label,
                                                                   encoder_features=encoder_features[i],
                                                                   rotation=rotations[i]
                                                                   )

            predicted_trees += [predicted_tree]
            pred_rotations += [pred_rotation]

            all_losses += [object_losses]
            all_leaves_geos += [all_leaf_geos]
            all_priors += [S_priors]

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0}

        for object_losses in all_losses:
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss

        output = [predicted_trees, x_roots, losses, all_leaves_geos, all_priors, pred_rotations]
        output = tuple(output)

        del all_losses

        return output

    def training_step(self, batch, batch_idx):
        Tree.load_category_info(self.config['hierarchies'])

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]

        input_batch = [scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations]

        losses = self.forward(tuple(input_batch))
        losses_unweighted = losses.copy()

        losses['geo'] *= self.config['loss_weight_geo']
        losses['geo_prior'] *= self.config['loss_weight_geo_prior']
        losses['leaf'] *= self.config['loss_weight_leaf']
        losses['exists'] *= self.config['loss_weight_exists']
        losses['semantic'] *= self.config['loss_weight_semantic']
        losses['edge_exists'] *= self.config['loss_weight_edge']
        losses['root_cls'] *= self.config['loss_weight_root_cls']
        losses['rotation'] *= self.config['loss_weight_rotation']

        total_loss = 0
        for loss in losses.values():
            total_loss += loss

        return {'loss': total_loss,
                'train_loss_components': losses,
                'train_loss_components_unweighted': losses_unweighted}

    def training_epoch_end(self, outputs):
        log = {}
        losses = losses_unweighted = {'geo': 0,
                                      'geo_prior': 0,
                                      'leaf': 0,
                                      'exists': 0,
                                      'semantic': 0,
                                      'edge_exists': 0,
                                      'root_cls': 0,
                                      'rotation': 0,
                                      }
        train_loss = torch.zeros(1).type_as(outputs[0]['loss'])

        for output in outputs:
            train_loss += output['loss']
            for key in losses:
                losses[key] += output['train_loss_components'][key]
                losses_unweighted[key] += output['train_loss_components_unweighted'][key]
        train_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)
            losses_unweighted[key] /= len(outputs)

        log.update(losses)
        log.update(losses_unweighted)
        log.update({'loss': train_loss})

        results = {'log': log}

        del outputs

        return results

    def validation_step(self, batch, batch_idx):
        Tree.load_category_info(self.config['hierarchies'])

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]

        input_batch = [scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations]

        losses = self.forward(tuple(input_batch))
        losses_unweighted = losses.copy()

        losses['geo'] *= self.config['loss_weight_geo']
        losses['geo_prior'] *= self.config['loss_weight_geo_prior']
        losses['leaf'] *= self.config['loss_weight_leaf']
        losses['exists'] *= self.config['loss_weight_exists']
        losses['semantic'] *= self.config['loss_weight_semantic']
        losses['edge_exists'] *= self.config['loss_weight_edge']
        losses['root_cls'] *= self.config['loss_weight_root_cls']
        losses['rotation'] *= self.config['loss_weight_rotation']

        total_loss = 0
        for loss_name, loss in losses.items():
            total_loss += loss

        del shape_sdfs, gt_trees

        return {'val_loss': total_loss,
                'val_loss_components': losses,
                'val_loss_components_unweighted': losses_unweighted}

    def validation_epoch_end(self, outputs):
        log = {}
        losses = losses_unweighted = {'val_geo': 0,
                                      'val_geo_prior': 0,
                                      'val_leaf': 0,
                                      'val_exists': 0,
                                      'val_semantic': 0,
                                      'val_edge_exists': 0,
                                      'val_root_cls': 0,
                                      'val_rotation': 0,
                                      }
        val_loss = torch.zeros(1).type_as(outputs[0]['val_loss'])

        for output in outputs:
            val_loss += output['val_loss']
            for key in losses:
                losses[key] += output['val_loss_components'][key[4:]]
                losses_unweighted[key] += output['val_loss_components_unweighted'][key[4:]]
        val_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)
            losses_unweighted[key] /= len(outputs)

        log.update(losses)
        log.update(losses_unweighted)
        log.update({'val_loss': val_loss})

        del outputs

        results = {'log': log}
        torch.cuda.empty_cache()

        return results

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=True,
                          num_workers=self.config['num_workers'], drop_last=True,
                          collate_fn=collate_feats)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'], drop_last=True,
                          collate_fn=collate_feats)
