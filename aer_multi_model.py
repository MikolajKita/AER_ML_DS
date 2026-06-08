from collections import deque
import copy
import logging

from river import base, ensemble, tree, naive_bayes, forest
from river.drift.binary import EDDM


logger = logging.getLogger(__name__)


class ModelFactory:
    GLOBAL_SEED = 42

    @classmethod
    def createModel(cls, modelType):
        # Hoeffding trees
        if modelType == 'HT':
            return tree.HoeffdingTreeClassifier()
        # Adaptive random forest
        if modelType == 'ARF':
            return forest.ARFClassifier(seed=cls.GLOBAL_SEED)
        # Streaming Random Patches
        if modelType == 'SRP':
            return ensemble.SRPClassifier(seed=cls.GLOBAL_SEED)
        if modelType == 'NaiveBayes':
            return naive_bayes.GaussianNB()

class ActiveExpertRepositoryMultiModel(base.Classifier):
    """
    Active Expert Repository (AER) Algorithm with Multi-Model Shadow Experts.
    Adaptive Stream Mining with Tournament-Based Selection using multiple model types.
    """
    def __init__(
        self,
        base_estimator: base.Classifier,
        drift_detector=None,
        available_models=['HT', 'ARF', 'SRP', 'NaiveBayes'],
        min_samples=10,
        window_size=30,
        warmup_steps=0,
    ):
        self.base_estimator = base_estimator
        self.drift_detector = (
            drift_detector.clone()
            if drift_detector is not None
            else EDDM(warm_start=50, alpha=0.8, beta=0.5)
        )
        self.model_types = available_models

        self.active_expert = base_estimator.clone()
        self.repository = [] # List of frozen experts
        self.repository_labels = []
        self.expert_counter = 0
        self.active_expert_label = self._make_label(self.expert_counter, self.active_expert)

        self.min_samples = min_samples
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        
        self.in_warning = False
        self.warning_step_count = 0
        self.active_snapshot = None
        self.shadow_experts = []
        self.updated_clones = []
        self.updated_clone_labels = []
        
        self.active_correct = None
        self.shadow_correct = []
        self.clone_correct = []
        
    def get_active_expert_label(self):
        return self.active_expert_label
        
    def predict_one(self, x, **kwargs):
        return self.active_expert.predict_one(x, **kwargs)
        
    def predict_proba_one(self, x, **kwargs):
        return self.active_expert.predict_proba_one(x, **kwargs)
        
    def learn_one(self, x, y):
        # Predict first to update detector
        y_pred_active = self.active_expert.predict_one(x)
        error = 0.0 if y_pred_active == y else 1.0
        
        # Update detector
        self.drift_detector.update(error)
        
        if self.drift_detector.warning_detected and not self.in_warning:
            self._start_warning_phase()

        if self.in_warning and self.drift_detector.warning_detected:
            if self.warning_step_count >= self.warmup_steps:
                self._score_warning_candidates(x, y, y_pred_active)
            self._train_warning_candidates(x, y)
            self.warning_step_count += 1
        else:
            self.active_expert.learn_one(x, y)
        
        if self.drift_detector.drift_detected:
            logger.info(
                "AER multi-model drift detected: active_expert=%s",
                self.active_expert_label,
            )
            self._select_active_expert()
            self._reset_warning_phase()
            self.drift_detector = self.drift_detector.clone() # reset detector
        elif self.in_warning and not self.drift_detector.warning_detected:
            self._reset_warning_phase()
        
        return self

    def _start_warning_phase(self):
        self.in_warning = True
        self.warning_step_count = 0
        self.active_snapshot = copy.deepcopy(self.active_expert)

        self.shadow_experts = [ModelFactory.createModel(mt) for mt in self.model_types]
        self.updated_clones = [copy.deepcopy(expert) for expert in self.repository]
        self.updated_clone_labels = list(self.repository_labels)

        self.active_correct = deque(maxlen=self.window_size)
        self.shadow_correct = [
            deque(maxlen=self.window_size) for _ in self.shadow_experts
        ]
        self.clone_correct = [
            deque(maxlen=self.window_size) for _ in self.updated_clones
        ]

    def _score_warning_candidates(self, x, y, y_pred_active):
        self.active_correct.append(self._is_correct(y, y_pred_active))

        for shadow, correct_window in zip(self.shadow_experts, self.shadow_correct):
            y_pred_shadow = shadow.predict_one(x)
            correct_window.append(self._is_correct(y, y_pred_shadow))

        for clone, correct_window in zip(self.updated_clones, self.clone_correct):
            y_pred_clone = clone.predict_one(x)
            correct_window.append(self._is_correct(y, y_pred_clone))

    def _train_warning_candidates(self, x, y):
        self.active_expert.learn_one(x, y)

        for shadow in self.shadow_experts:
            shadow.learn_one(x, y)

        for clone in self.updated_clones:
            clone.learn_one(x, y)

    def _select_active_expert(self):
        self._archive_active_snapshot()

        if self.active_correct is None or len(self.active_correct) < self.min_samples:
            scored_samples = len(self.active_correct) if self.active_correct is not None else 0
            logger.info(
                "AER multi-model tournament skipped: active_expert=%s "
                "scored_samples=%d min_samples=%d. Resetting to fresh expert.",
                self.active_expert_label,
                scored_samples,
                self.min_samples,
            )
            self._replace_with_fresh_expert()
            return

        active_score = self._accuracy(self.active_correct)
        candidates = self._tournament_candidates()

        if not candidates:
            self._log_tournament([], active_score=active_score)
            return

        best_score, best_model, best_label, is_new_expert = max(
            candidates,
            key=lambda candidate: candidate[0],
        )
        self._log_tournament(candidates, active_score=active_score, best_score=best_score)

        if best_score <= active_score:
            return

        self.active_expert = best_model
        self.active_expert_label = best_label

        if is_new_expert:
            self.expert_counter += len(self.shadow_experts)

    def _tournament_candidates(self):
        candidates = []

        for i, (shadow, correct_window) in enumerate(zip(self.shadow_experts, self.shadow_correct)):
            if len(correct_window) >= self.min_samples:
                label = self._make_label(self.expert_counter + i + 1, shadow)
                candidates.append((
                    self._accuracy(correct_window),
                    shadow,
                    label,
                    True,
                ))

        for clone, label, correct_window in zip(
            self.updated_clones,
            self.updated_clone_labels,
            self.clone_correct,
        ):
            if len(correct_window) >= self.min_samples:
                candidates.append((
                    self._accuracy(correct_window),
                    clone,
                    label,
                    False,
                ))

        return candidates

    def _replace_with_fresh_expert(self):
        if self.active_snapshot is None and self.active_expert_label not in self.repository_labels:
            self.repository.append(self.active_expert.clone())
            self.repository_labels.append(self.active_expert_label)

        previous_label = self.active_expert_label
        self.expert_counter += 1
        self.active_expert = self.base_estimator.clone()
        self.active_expert_label = self._make_label(self.expert_counter, self.active_expert)

        logger.info(
            "AER multi-model fresh expert activated: previous_expert=%s active_expert=%s",
            previous_label,
            self.active_expert_label,
        )

    def _archive_active_snapshot(self):
        if self.active_snapshot is None:
            return

        if self.active_expert_label in self.repository_labels:
            index = self.repository_labels.index(self.active_expert_label)
            self.repository[index] = self.active_snapshot
        else:
            self.repository.append(self.active_snapshot)
            self.repository_labels.append(self.active_expert_label)

        self.active_snapshot = None

    def _reset_warning_phase(self):
        self.in_warning = False
        self.warning_step_count = 0
        self.active_snapshot = None
        self.shadow_experts = []
        self.updated_clones = []
        self.updated_clone_labels = []
        self.active_correct = None
        self.shadow_correct = []
        self.clone_correct = []

    def _log_tournament(self, candidates, active_score=None, best_score=None):
        candidate_scores = [
            {
                "label": label,
                "score": score,
                "is_new_expert": is_new_expert,
            }
            for score, _, label, is_new_expert in candidates
        ]

        if best_score is None and candidate_scores:
            best_score = max(candidate["score"] for candidate in candidate_scores)

        logger.info(
            "AER multi-model tournament: active_expert=%s active_score=%s candidate_count=%d "
            "best_score=%s candidates=%s",
            self.active_expert_label,
            active_score,
            len(candidate_scores),
            best_score,
            candidate_scores,
        )

    @staticmethod
    def _make_label(index, expert):
        return f"Expert_{index}_{expert.__class__.__name__}"

    @staticmethod
    def _is_correct(y_true, y_pred):
        return 1.0 if y_pred == y_true else 0.0

    @staticmethod
    def _accuracy(correct_window):
        return sum(correct_window) / len(correct_window)
