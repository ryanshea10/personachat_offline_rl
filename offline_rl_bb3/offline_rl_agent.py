# from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from projects.seeker.agents.seeker import ComboFidGoldDocumentAgent
from parlai.core.torch_generator_agent import PPLMetric
from parlai.core.metrics import (
    AverageMetric,
    Metrics,
    Metric,
    GlobalAverageMetric,
    GlobalFixedMetric,
    GlobalTimerMetric,
)

import torch
import numpy as np


class OfflineRLAgent(ComboFidGoldDocumentAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)

    def compute_loss(self, batch, return_output= False):
        """
        Override standard TGA.compute_loss to call relevant RAG Model Interface.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.get_model_output(batch)
        scores, preds, enc_state, *_ = model_output

        self._record_retrieval_metrics(batch, enc_state)
        (
            loss,
            metric_loss,
            metric_correct,
            metric_target_tokens,
        ) = self.compute_rag_loss(
            self.criterion, scores, preds, enc_state, batch.label_vec, batch
        )

        self.record_local_metric(
            'loss', AverageMetric.many(metric_loss, metric_target_tokens)
        )
        self.record_local_metric(
            'ppl', PPLMetric.many(metric_loss, metric_target_tokens)
        )
        self.record_local_metric(
            'token_acc', AverageMetric.many(metric_correct, metric_target_tokens)
        )
        self.record_local_metric(
            'token_em',
            AverageMetric.many(
                [x == y for x, y in zip(metric_correct, metric_target_tokens)]
            ),
        )

        if return_output:
            return loss, model_output
        else:
            return loss
    
    def compute_rag_loss(
        self,
        criterion,
        scores,
        preds,
        enc_state,
        label_vec,
        batch
    ):
        """
        Compute RAG Token Loss.

        This is a simple NLL Loss.

        :param criterion:
            presumably the NLL criterion.
        :param scores:
            model scores
        :param preds:
            model "predicions" of tokens
        :param enc_state:
            encoder states
        :param label_vec:
            target tokens

        :return (loss, metric_loss, correct_tokens, target_tokens):
            loss: the loss through which we backpropagate
            metric_loss: loss we use for metrics
            correct_tokens: correct predictions from the model
            target_tokens: the ground truth tokens.
        """
        if scores.size(1) != label_vec.size(1):
            assert self.generation_model == 'bart'
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]  # type: ignore

        # compute loss
        score_view = scores.reshape(-1, scores.size(-1))
        loss = criterion(score_view, label_vec.view(-1))
        # loss = loss.view(scores.shape[:-1]).sum(dim=1)
        loss_mat = loss.view(scores.shape[:-1])

        # importance sampling
        if batch.rewards is not None:
            vec = []
            probs = torch.nn.Softmax(dim=1)(score_view)
            for i in range(len(probs.gather(dim=-1, index=batch.label_vec))):
                if batch.rewards[i] == -1 or self.opt['importance_sampling'] == 'gold':
                    p = probs.gather(dim=-1, index=batch.label_vec)[i]
                    vec.append(list(torch.clamp(p, min=self.opt['is_lower_bound'], max=1.0).cpu().detach().numpy()))
                else:
                    vec.append(list(np.ones(len(batch.label_vec[0]), dtype=np.float16)))
            loss_mat *= torch.tensor(vec, dtype=torch.float16).to('cuda')
        
        loss = loss_mat.sum(dim=1)
        if batch.rewards is not None:
            loss *= batch.rewards

        # calculate metric counters
        metric_loss = loss.tolist()
        notnull = label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((label_vec == preds) * notnull).sum(dim=-1)

        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        return loss, metric_loss, correct, target_tokens

    def compute_loss_torch_agent(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss_mat = loss.view(scores.shape[:-1])
        
        # importance sampling
        if batch.rewards is not None:
            vec = []
            probs = torch.nn.Softmax(dim=1)(score_view)
            for i in range(len(probs.gather(dim=-1, index=batch.label_vec))):
                if batch.rewards[i] == -1 or self.opt['importance_sampling'] == 'gold':
                    p = probs.gather(dim=-1, index=batch.label_vec)[i]
                    vec.append(list(torch.clamp(p, min=.15, max=1.0).cpu().detach().numpy()))
                else:
                    vec.append(list(np.ones(len(batch.label_vec[0]), dtype=np.float16)))
            loss_mat *= torch.tensor(vec, dtype=torch.float16).to('cuda')

        loss = loss_mat.sum(dim=1)
        if batch.rewards is not None:
            loss *= batch.rewards

        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        # cross entropy loss
        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        # perplexity
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        # token-wise accuracy
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # utterance-wise exact match
        self.record_local_metric(
            'token_em', AverageMetric.many(correct == target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

