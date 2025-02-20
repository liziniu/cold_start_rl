import torch
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    ###
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    is_torch_xla_available,
)
from typing import List, Optional, Dict


class SFTTrainer(Trainer):

    @torch.no_grad
    def compute_training_logs(self, logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        entropy = chunked_entropy_from_logits(
            shift_logits,
            chunk_size=max(1, shift_logits.size(0) // 4),
        ).mean()
        training_logs = {"entropy": round(entropy.item(), 2)}

        return training_logs

    def gem_loss(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="logsigmoid"):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            elif h == "logsigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(h)

        gene_log_probs = F.log_softmax(shift_logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        )

        if num_items_in_batch is not None:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).sum() / num_items_in_batch
        else:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).mean()

        return loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if self.args.loss == "ce" or self.control.should_evaluate:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            else:
                loss = self.gem_loss(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h
                )

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        # ziniu add logs
        if not self.control.should_evaluate:
            self.training_logs = self.compute_training_logs(
                outputs.logits, inputs["labels"]
            )
            self.training_logs["ce_loss"] = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)


def chunked_entropy_from_logits(logits, chunk_size=None):
    if chunk_size is None:
        chunk_size = logits.size(0)
    # Total number of sequences (batch size)
    total_sequences = logits.size(0)
    # Initialize a tensor to hold the entropy results for each sequence and time step
    entropy_results = torch.zeros_like(
        logits[..., 0], dtype=logits.dtype, device=logits.device
    )

    # Process in chunks to manage memory
    for start in range(0, total_sequences, chunk_size):
        end = start + chunk_size
        chunk_logits = logits[start:end]

        pd = F.softmax(chunk_logits, dim=-1)
        entropy_chunk = torch.logsumexp(chunk_logits, axis=-1) - torch.sum(
            pd * chunk_logits, axis=-1
        )

        # Store results
        entropy_results[start:end] = entropy_chunk

    return entropy_results