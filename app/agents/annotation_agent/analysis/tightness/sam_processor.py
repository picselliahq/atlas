import cv2
import numpy as np
import requests
from picsellia import Asset, Rectangle
from PIL import Image

from agents.annotation_agent.models.annotations import (
    ShapeIssueType,
    TightnessAnnotation,
    TightnessImage,
    TightnessShape,
)
from agents.common.ai_models.sam_model import SAMModel


class TightnessProcessor:
    def __init__(self) -> None:
        pass

    def compute_iou(self, gt_bbox, sam_bbox) -> float:
        """Compute IoU between two boxes."""
        box1_x1, box1_y1 = gt_bbox[0], gt_bbox[1]
        box1_x2, box1_y2 = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]

        box2_x1, box2_y1 = sam_bbox[0], sam_bbox[1]
        box2_x2, box2_y2 = sam_bbox[0] + sam_bbox[2], sam_bbox[1] + sam_bbox[3]

        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = np.int32(gt_bbox[2] * gt_bbox[3])
        box2_area = np.int32(sam_bbox[2] * sam_bbox[3])
        union_area = box1_area + box2_area - intersection_area

        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0

    def generate_tightness_shapes(self, sam_boxes, gt_boxes, rectangles, gt_labels):
        """Generate tightness shapes from SAM and GT boxes."""
        shapes = []
        for i, sam_box in enumerate(sam_boxes):
            iou = self.compute_iou(gt_bbox=gt_boxes[i], sam_bbox=sam_box)
            shapes.append(
                TightnessShape(
                    issue=ShapeIssueType.BOX_TIGHTNESS,
                    box=gt_boxes[i],
                    suggestion=sam_box,
                    iou=iou,
                    id=str(rectangles[i].id),
                    label=gt_labels[i].name,
                    label_id=str(gt_labels[i].id),
                )
            )
        return shapes


class SAMProcessor:
    def __init__(self) -> None:
        self.sam_model = SAMModel()
        self.tightness_processor = TightnessProcessor()

    def analyse_image(self, asset: Asset, debug: bool = False) -> TightnessImage:
        image = Image.open(requests.get(asset.url, stream=True).raw)
        annotation = asset.list_annotations()[0]
        rectangles = annotation.list_rectangles()

        sam_boxes, gt_boxes, input_boxes, gt_labels = self._predict(
            image=image, rectangles=rectangles
        )

        # Generate tightness shapes
        shapes = self.tightness_processor.generate_tightness_shapes(
            sam_boxes, gt_boxes, rectangles, gt_labels
        )

        if debug:
            self._debug_plot(image, sam_boxes, gt_boxes, input_boxes, asset.id)

        return TightnessImage(
            asset_id=str(asset.id),
            annotation=TightnessAnnotation(id=str(annotation.id), shapes=shapes),
        )

    def _resize_image(self, image: Image, input_size: int = 1024):
        w, h = image.size
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))
        return image, scale

    def _predict(self, image: Image, rectangles: list[Rectangle]):
        rescale_image, scale = self._resize_image(image, 1024)

        img_w, img_h = image.size[:2]
        rimg_w, rimg_h = rescale_image.size[:2]

        # Prepare the input boxes for inference
        box_prompts = []
        input_labels = []
        gt_boxes = []
        gt_labels = []

        for r in rectangles:
            x1 = int(r.x * scale)
            y1 = int(r.y * scale)
            x2 = int((r.x + r.w) * scale)
            y2 = int((r.y + r.h) * scale)

            # Expanding the box
            expanded_x1 = int(max(x1 * 0.9, 1))
            expanded_y1 = int(max(y1 * 0.9, 1))
            expanded_x2 = int(min(x2 * 1.1, rimg_w - 1))
            expanded_y2 = int(min(y2 * 1.1, rimg_h - 1))

            box_prompts.append([expanded_x1, expanded_y1, expanded_x2, expanded_y2])
            input_labels.append(f"a photo of a {r.label.name}")
            gt_boxes.append([r.x, r.y, r.w, r.h])
            gt_labels.append(r.label)

        # Predict with SAM and post-process the results
        sam_results = []
        for i, box_prompt in enumerate(box_prompts):
            results = self.sam_model.predict(
                image=rescale_image, box_prompt=box_prompt, label=input_labels[i]
            )
            sam_results.append(results)

        # Post-process the results
        sam_boxes = []
        for results in sam_results:
            sam_boxes.extend(self.sam_model.post_process(results, (img_w, img_h)))

        return sam_boxes, gt_boxes, box_prompts, gt_labels

    def _debug_plot(self, image, sam_boxes, gt_boxes, input_boxes, asset_id):
        """Visualize SAM and GT boxes for debugging."""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for sam_box in sam_boxes:
            x, y, w, h = sam_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for gt_box in gt_boxes:
            x, y, w, h = gt_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for input_box in input_boxes:
            x, y, w, h = input_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imwrite(f"analysis_{asset_id}.jpg", img)
