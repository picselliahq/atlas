import ast

import numpy as np
import pandas as pd
import picsellia
from sklearn.metrics.pairwise import cosine_similarity

from agents.annotation_agent.models.annotations import (
    Annotation,
    Image,
    Shape,
    ShapeIssueType,
)
from agents.common.ai_models.clip_model import CLIPModelHandler
from agents.common.stats.analysis.base_stats import BaseStats


class ShapeMislabeledStats(BaseStats["ShapeMislabeledStats"]):
    def __init__(
        self, df: pd.DataFrame, labels: list[picsellia.Label], threshold: float = 0.01
    ):
        super().__init__(df)

        self.threshold = threshold

        self.df["shape_embeddings"] = self.df["shape_embeddings"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.clip_handler = CLIPModelHandler()
        self.clip_embedded_labels = self.clip_handler.compute_text_embeddings(
            [label.name for label in labels]
        )
        self.picsellia_labels_map = {label.name: label.id for label in labels}

    def compute(
        self,
    ) -> "ShapeMislabeledStats":
        wrong, ambiguous = self._find_wrongly_annotated_shapes()
        self.wrongly_labeled_shape = wrong
        self.ambiguous_labeled_shape = ambiguous
        return self

    def _find_wrongly_annotated_shapes(
        self,
    ) -> tuple[list[Image], list[Image]]:
        wrongly_annotated_for_sure = []
        ambiguous_annotations = []
        correct_shapes = 0
        for _, row in self.df.iterrows():
            image_embedding = row["shape_embeddings"]
            label_annotated = str(row["label"])
            annotation_id = str(row["annotation_id"])
            if image_embedding:
                status, top_labels, _ = self._evaluate_label_quality(
                    shape_embedding=image_embedding,
                    annotated_label=label_annotated,
                    label_embeddings=self.clip_embedded_labels,
                )
                if status == "ambiguous":
                    ambiguous_annotations.append(
                        Image(
                            asset_id=str(row["asset_id"]),
                            annotation=Annotation(
                                id=annotation_id,
                                shapes=[
                                    Shape(
                                        id=str(row["shape_id"]),
                                        x=row["x"],
                                        y=row["y"],
                                        w=row["w"],
                                        h=row["h"],
                                        issue=ShapeIssueType.AMBIGIOUS_CLASS,
                                        label_annotated=str(
                                            self.picsellia_labels_map[label_annotated]
                                        ),
                                        label_suggested=str(
                                            self.picsellia_labels_map[top_labels[0][0]]
                                        ),
                                    )
                                ],
                            ),
                        )
                    )
                elif status == "wrong":
                    wrongly_annotated_for_sure.append(
                        Image(
                            asset_id=str(row["asset_id"]),
                            annotation=Annotation(
                                id=annotation_id,
                                shapes=[
                                    Shape(
                                        id=str(row["shape_id"]),
                                        x=row["x"],
                                        y=row["y"],
                                        w=row["w"],
                                        h=row["h"],
                                        issue=ShapeIssueType.WRONG_CLASS,
                                        label_annotated=str(
                                            self.picsellia_labels_map[label_annotated]
                                        ),
                                        label_suggested=str(
                                            self.picsellia_labels_map[top_labels[0][0]]
                                        ),
                                    )
                                ],
                            ),
                        )
                    )
                elif status == "correct":
                    correct_shapes += 1

        return ambiguous_annotations, wrongly_annotated_for_sure

    def _evaluate_label_quality(
        self,
        shape_embedding,
        annotated_label,
        label_embeddings,
        min_confidence=0.5,
        rank_tolerance=1,  # Maximum acceptable rank for small datasets
        verbose=False,
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """
        Simplified version to handle datasets with fewer classes.
        For datasets with < 10 classes, returns only 'correct' or 'wrong'.

        Parameters:
            shape_embedding: vector representation of shape
            annotated_label: label assigned by annotator
            label_embeddings: dict of {label: embedding_vector}
            min_confidence: float → minimum similarity of top match to call something 'wrong'
            verbose: bool → whether to print detailed information

        Returns:
            status: 'correct' | 'wrong'
            top_labels: list of (label, similarity)
            annotated_label_rank: int
        """

        shape_vec = np.array(shape_embedding).reshape(1, -1)
        label_names = list(label_embeddings.keys())
        label_vecs = np.array([label_embeddings[label] for label in label_names])
        num_labels = len(label_names)

        # Determine if we should use simplified logic based on class count
        use_simplified = num_labels < 10

        # Always check at least top 2 labels, but cap at 5% for larger datasets
        top_n = self._determine_top_n(num_labels)

        similarities, sorted_labels = self._compute_similarities(
            shape_vec, label_vecs, label_names
        )

        top_labels = sorted_labels[:top_n]
        annotated_label_rank = self._get_annotated_label_rank(
            sorted_labels, annotated_label
        )
        top_label, top_sim = sorted_labels[0]
        annotated_sim = self._get_annotated_similarity(
            annotated_label, label_names, similarities
        )

        # If annotated label not in embeddings, always wrong
        if annotated_label_rank is None:
            return self._handle_missing_annotated_label(verbose, top_labels)

        # If top match, correct
        if annotated_label_rank == 0:
            return self._handle_correct_label(verbose, top_labels)

        # For simplified case (< 10 classes): More strict evaluation for non-top matches
        if use_simplified:
            return self._handle_simplified_case(
                annotated_label_rank=annotated_label_rank,
                top_sim=top_sim,
                annotated_sim=annotated_sim,
                top_labels=top_labels,
                min_confidence=min_confidence,
                rank_tolerance=rank_tolerance,
                verbose=verbose,
            )

        # For datasets with >= 10 classes: Use original logic
        # If not in top-N, check if it's wrong or ambiguous
        return self._handle_full_case(
            annotated_label_rank=annotated_label_rank,
            top_n=top_n,
            annotated_label=annotated_label,
            top_sim=top_sim,
            annotated_sim=annotated_sim,
            top_labels=top_labels,
            top_label=top_label,
            min_confidence=min_confidence,
            verbose=verbose,
        )

    def _determine_top_n(self, num_labels: int) -> int:
        """Determine how many top labels to consider based on dataset size."""
        return max(2, int(0.05 * num_labels))

    def _compute_similarities(
        self, shape_vec, label_vecs, label_names
    ) -> tuple[np.ndarray, list[tuple[str, float]]]:
        """Compute cosine similarities and sort labels by similarity."""
        similarities = cosine_similarity(shape_vec, label_vecs)[0]
        sorted_indices = similarities.argsort()[::-1]
        sorted_labels = [(label_names[i], similarities[i]) for i in sorted_indices]
        return similarities, sorted_labels

    def _get_annotated_label_rank(self, sorted_labels, annotated_label) -> int | None:
        """Find the rank of the annotated label in the sorted labels."""
        return next(
            (
                i
                for i, (label, _) in enumerate(sorted_labels)
                if label == annotated_label
            ),
            None,
        )

    def _get_annotated_similarity(
        self, annotated_label, label_names, similarities
    ) -> float | None:
        """Get the similarity of the annotated label with the top label."""
        if annotated_label in label_names:
            return similarities[label_names.index(annotated_label)]
        return None

    def _handle_missing_annotated_label(
        self, verbose, top_labels
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """Handle the case where the annotated label is not in the embeddings."""
        if verbose:
            print("❌ Annotated label not in label embeddings.")
        return "wrong", top_labels, None

    def _handle_correct_label(
        self, verbose, top_labels
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """Handle the case where the top label matches the annotated label."""
        if verbose:
            print("✅ Result: CORRECT (top-1 match)")
        return "correct", top_labels, 0

    def _handle_simplified_case(
        self,
        annotated_label_rank,
        top_sim,
        annotated_sim,
        top_labels,
        min_confidence,
        rank_tolerance,
        verbose,
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """Handle the simplified case where there are fewer than 10 classes."""
        if annotated_label_rank > rank_tolerance:
            if verbose:
                print(
                    f"🚩 Result: WRONG (rank {annotated_label_rank + 1} exceeds max tolerated rank {rank_tolerance + 1})"
                )
            return "wrong", top_labels, annotated_label_rank

        if annotated_sim < min_confidence:
            if verbose:
                print(
                    f"🚩 Result: WRONG (similarity {annotated_sim:.4f} below min confidence {min_confidence:.4f})"
                )
            return "wrong", top_labels, annotated_label_rank

        strict_threshold = self.threshold / 2
        if (top_sim - annotated_sim) > strict_threshold:
            if verbose:
                print(
                    f"🚩 Result: WRONG (gap {top_sim - annotated_sim:.4f} > strict threshold {strict_threshold:.4f})"
                )
            return "wrong", top_labels, annotated_label_rank
        else:
            if verbose:
                print(
                    "✅ Result: CORRECT (within rank tolerance with small similarity gap)"
                )
            return "correct", top_labels, annotated_label_rank

    def _handle_full_case(
        self,
        annotated_label_rank,
        top_n,
        annotated_label,
        top_sim,
        annotated_sim,
        top_labels,
        top_label,
        min_confidence,
        verbose,
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """Handle the full case where there are 10 or more classes."""
        if annotated_label_rank >= top_n:
            return self._handle_non_top_prediction(
                annotated_label=annotated_label,
                top_labels=top_labels,
                annotated_sim=annotated_sim,
                top_sim=top_sim,
                top_label=top_label,
                min_confidence=min_confidence,
                verbose=verbose,
            )

        return self._handle_top_prediction(
            annotated_label_rank=annotated_label_rank,
            top_labels=top_labels,
            verbose=verbose,
        )

    def _handle_non_top_prediction(
        self,
        annotated_label,
        top_labels,
        annotated_sim,
        top_sim,
        top_label,
        min_confidence,
        verbose,
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """Handles the case where the annotated label is not in the top N predictions."""
        if verbose:
            self._print_prediction_details(
                annotated_label, top_labels, annotated_sim, top_sim, top_label
            )

        if top_sim >= min_confidence and (top_sim - annotated_sim) > self.threshold:
            if verbose:
                print(
                    f"🚩 Result: WRONG (not in top-{len(top_labels)} and confidence gap {top_sim - annotated_sim:.4f} > {self.threshold})"
                )
            return "wrong", top_labels, None

        if verbose:
            print(
                f"⚠️ Result: AMBIGUOUS (not in top-{len(top_labels)} but similarity gap is small)"
            )
        return "ambiguous", top_labels, None

    def _handle_top_prediction(
        self,
        annotated_label_rank,
        top_labels,
        verbose,
    ) -> tuple[str, list[tuple[str, float]], int | None]:
        """Handles the case where the annotated label is in the top N predictions."""
        if verbose:
            print(f"✅ Result: CORRECT (top-{annotated_label_rank + 1} match)")

        sim_vals = [sim for _, sim in top_labels]
        sim_range = max(sim_vals) - min(sim_vals)
        is_clustered = sim_range < self.threshold

        if is_clustered:
            if verbose:
                print("⚠️ Result: AMBIGUOUS (similarity values clustered)")
            return "ambiguous", top_labels, annotated_label_rank

        return "correct", top_labels, annotated_label_rank

    def _print_prediction_details(
        self, annotated_label, top_labels, annotated_sim, top_sim, top_label
    ):
        """Prints details about the prediction for debugging purposes."""
        print(f"\n🔎 Annotated Label: {annotated_label}")
        print(f"📦 Top {len(top_labels)} Predicted Labels:")
        for i, (label, sim) in enumerate(top_labels):
            print(f"  {i + 1}. {label:20s} | Similarity: {sim:.4f}")
        if annotated_sim is not None:
            print(f"🧭 Annotated label similarity: {annotated_sim:.4f}")
        print(f"🏅 Top prediction: {top_label} ({top_sim:.4f})")
