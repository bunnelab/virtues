import os
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments
from utils.utils import is_rank0, set_seed, load_marker_embeddings, to_device
from modules.multiplex_virtues import MultiplexVirtues
from datasets.multiplex_dataset import MultiplexDataset
import torch
from torch.utils.data import ConcatDataset, Dataset
import wandb

class VirTuesTrainer(Trainer):

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        multiplex, channel_ids, multiplex_mask = inputs
        
        multiplex = to_device(multiplex, 'cuda') # list of tensors of shape C_i x H x W
        channel_ids = to_device(channel_ids, 'cuda') # list of tensors of shape C_i
        multiplex_mask = to_device(multiplex_mask, 'cuda') # list of tensors of shape C_i x H//patch_size x W//patch_size

        outputs = model.forward(
                            multiplex=multiplex,
                            channel_ids=channel_ids,
                            multiplex_mask=multiplex_mask
                        )
        
        reconstructions = outputs.decoded_multiplex # list of tensors of shape C_i x H x W
        reconstructions = torch.concat(reconstructions, dim=0) # sum(C_i) x H x W
        targets = torch.concat(multiplex, dim=0) # sum(C_i) x H x W
        
        loss = torch.mean(torch.pow(reconstructions - targets, 2))
        
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if DISABLE_EVAL:
            return {}
        else:
            eval_dl = self.get_eval_dataloader()
            sum = 0.0
            terms = 0
            for batch in eval_dl:
                with torch.no_grad():
                    loss = self.compute_loss(self.model, batch)
                    sum += loss.item()
                    terms += 1
            avg = sum / terms
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            if world_size > 1:
                avg_tensor = torch.tensor(avg).cuda()
                torch.distributed.all_reduce(avg_tensor, op=torch.distributed.ReduceOp.SUM)
                avg = (avg_tensor / world_size).item()

            results = {
                "epoch" : self.state.epoch,
                "test_loss" : avg,
            }

            if WANDB_AVAILABLE and is_rank0():
                wandb.log(results)
            
            return results

def build_datasets(conf, split: str) -> Dataset:
    """
    Initializes and merges collection of datasets provided in the configuration.
    """
    datasets = []
    for name in conf.datasets.keys():
        ds_conf = conf.datasets[name]
        dataset = MultiplexDataset(
            tissue_dir=ds_conf.tissue_dir,
            crop_dir=ds_conf.crop_dir,
            mask_dir=ds_conf.mask_dir,
            tissue_index=ds_conf.tissue_index,
            crop_index=ds_conf.crop_index,
            channels_file=ds_conf.channels_file,
            quantiles_file=ds_conf.quantiles_file,
            means_file=ds_conf.means_file,
            stds_file=ds_conf.stds_file,
            marker_embedding_dir=conf.marker_embedding_dir,
            split=split,
            crop_size=conf.data.crop_size,
            patch_size=conf.model.patch_size,
            masking_ratio=conf.data.masking_ratio,
            channel_fraction=conf.data.channel_fraction,
        )
        datasets.append(dataset)
    return ConcatDataset(datasets)


def train_virtues(conf):
    """
    Trains VirTues according to the provided configuration.
    """
    train_dataset = build_datasets(conf, split='train')
    test_dataset = build_datasets(conf, split='test')

    def custom_collate_fn(batch):
        multiplex = [sample[0] for sample in batch]
        channel_ids = [sample[1] for sample in batch]
        multiplex_mask = [sample[2] for sample in batch]
        return multiplex, channel_ids, multiplex_mask

    marker_embeddings = load_marker_embeddings(conf.marker_embedding_dir)

    model = MultiplexVirtues(
        use_default_config = False,
        custom_config = None,
        prior_bias_embeddings=marker_embeddings,
        prior_bias_embedding_type='esm',
        prior_bias_embedding_fusion_type='add',
        patch_size=conf.model.patch_size,
        model_dim=conf.model.model_dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        decoder_pattern=conf.model.decoder_pattern,
        num_decoder_heads=conf.model.num_decoder_heads,
        num_hidden_layers=conf.model.num_decoder_hidden_layers,
        positional_embedding_type=conf.model.positional_embedding_type,
        dropout=conf.model.dropout,
        group_layers=conf.model.group_layers,
        norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder,
        verbose=False
    )
    model.cuda()

    training_args = TrainingArguments(
        output_dir=os.path.join(conf.experiments_dir, conf.experiment.name, 'checkpoints'),
        num_train_epochs=conf.training.epochs,
        per_device_train_batch_size=conf.training.batch_size,
        per_device_eval_batch_size=conf.training.batch_size,
        eval_strategy="epoch" if not DISABLE_EVAL else "no",
        eval_steps=1, # epoch-wise evaluation
        save_strategy="epoch",  # Save the model at the end of each epoch
        save_total_limit=1,  # Only keep the last model
        logging_dir=f'{conf.experiments_dir}/{conf.experiment.name}/logs',
        logging_strategy="steps",
        logging_steps=100,
        fp16=conf.training.fp16,
        bf16=conf.training.bf16,
        dataloader_num_workers=conf.training.num_workers,
        report_to="wandb" if WANDB_AVAILABLE else "none",
        run_name=conf.experiment.name,
        gradient_accumulation_steps=conf.training.gradient_accumulation_steps,
        learning_rate=conf.training.lr,
        lr_scheduler_type=conf.training.lr_scheduler_type,
        weight_decay=conf.training.weight_decay,
        ddp_find_unused_parameters=True,
        warmup_ratio=(conf.training.warmup_epochs / conf.training.epochs) if conf.training.warmup_epochs > 0 else 0.0,
    )

    trainer = VirTuesTrainer(
        conf=conf,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=custom_collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    conf = OmegaConf.load("configs/base_config.yaml")
    cli_conf = OmegaConf.from_cli()
    if hasattr(cli_conf, 'datasets_config') and cli_conf.datasets_config is not None:
        dataset_conf = OmegaConf.load(cli_conf.datasets_config)
        conf = OmegaConf.merge(conf, dataset_conf)
    elif hasattr(conf, 'datasets_config') and conf.datasets_config is not None:
        dataset_conf = OmegaConf.load(conf.datasets_config)
        conf = OmegaConf.merge(conf, dataset_conf)

    conf = OmegaConf.merge(conf, cli_conf)

    if is_rank0():
        print("OmegaConf merged config:")
        print(OmegaConf.to_yaml(conf))

    os.makedirs(os.path.join(conf.experiments_dir, conf.experiment.name), exist_ok=True)
    os.makedirs(os.path.join(conf.experiments_dir, conf.experiment.name, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(conf.experiments_dir, conf.experiment.name, 'logs'), exist_ok=True)

    OmegaConf.save(conf, os.path.join(conf.experiments_dir, conf.experiment.name, 'config.yaml'))

    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        
    if WANDB_AVAILABLE and is_rank0():
        wandb.init(
            entity=conf.experiment.wandb_entity,
            project=conf.experiment.wandb_project,
            name=conf.experiment.name,
            mode=conf.experiment.wandb_mode,
            dir=os.path.join(conf.experiments_dir, conf.experiment.name),
        )

    set_seed(conf.experiment.seed)

    if os.environ.get('LOCAL_RANK', None) is not None:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

    DISABLE_EVAL = conf.training.disable_eval
    if DISABLE_EVAL and is_rank0():
        print("Evaluation is disabled.")

    train_virtues(conf)