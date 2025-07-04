import logging
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from atlas_vision.agents.common.charts.types import ChartRenderer, HistogramDisplayChart

logger = logging.getLogger(__name__)


class DataParser:
    def __init__(self, report_id: str):
        self.report_id = report_id

    def parse_hist_chart(self, value, name, chart_id) -> str:
        chart = HistogramDisplayChart.from_dict(value)
        filename = os.path.join(str(self.report_id), f"histogram-{chart_id}.png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        figure_file = ChartRenderer.to_png(chart)
        with open(filename, "wb") as f:
            f.write(figure_file)
        return f"\n![{name}]({filename})\n"

    def parse_shape_list_data(
        self,
        ids: list[int],
        annotation_metadata: pd.DataFrame,
        image_metadata: pd.DataFrame,
    ) -> str:
        """
        Process shape IDs to create cropped images with overlaid rectangles and labels.

        Args:
            ids: List of shape IDs to process
            annotation_metadata: DataFrame containing annotation data
            image_metadata: DataFrame containing image data

        Returns:
            str: Markdown string with image references
        """
        if not ids or annotation_metadata.empty or image_metadata.empty:
            return ""

        # Find the corresponding annotation data for the given shape IDs
        shape_data = annotation_metadata[annotation_metadata["shape_id"].isin(ids)]

        if shape_data.empty:
            return ""

        image_references = []
        max_images_per_line = 5
        image_size = 180  # square size in pixels
        crop_padding = 10  # context padding around the object
        bg_color = (200, 200, 200)  # light grey background

        for _, row in shape_data.iterrows():
            try:
                # Get image information
                filename = row["filename"]
                image_row = image_metadata[image_metadata["filename"] == filename]

                if image_row.empty:
                    continue

                image_path = image_row.iloc[0]["target_path"]

                # Load the full image
                image = Image.open(image_path)
                orig_w, orig_h = image.size

                # Get shape coordinates
                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                label = row["label"]

                # Crop area with context padding
                crop_x1 = max(0, int(x - crop_padding))
                crop_y1 = max(0, int(y - crop_padding))
                crop_x2 = min(orig_w, int(x + w + crop_padding))
                crop_y2 = min(orig_h, int(y + h + crop_padding))

                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1

                cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # Determine scaling factor to fit crop within the square
                scale = min(image_size / crop_w, image_size / crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                resized_crop = cropped_image.resize((new_w, new_h), Image.LANCZOS)

                # Create a square background
                square_img = Image.new("RGB", (image_size, image_size), bg_color)

                # Determine positioning based on aspect ratio
                if new_h > new_w:  # Portrait: center horizontally
                    paste_x = (image_size - new_w) // 2
                    paste_y = 0
                else:  # Landscape: center vertically
                    paste_x = 0
                    paste_y = (image_size - new_h) // 2

                square_img.paste(resized_crop, (paste_x, paste_y))

                # Adjust shape coordinates for the square image
                # Shape's position in the crop:
                shape_x_in_crop = x - crop_x1
                shape_y_in_crop = y - crop_y1
                adjusted_x = int(shape_x_in_crop * scale) + paste_x
                adjusted_y = int(shape_y_in_crop * scale) + paste_y
                adjusted_w = int(w * scale)
                adjusted_h = int(h * scale)

                # Draw on the square image
                draw = ImageDraw.Draw(square_img)

                # Draw rectangle around the shape
                rectangle_coords = [
                    adjusted_x,
                    adjusted_y,
                    adjusted_x + adjusted_w,
                    adjusted_y + adjusted_h,
                ]
                draw.rectangle(rectangle_coords, outline=(255, 0, 0), width=3)

                # Add label text
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = adjusted_x
                text_y = max(0, adjusted_y - text_height - 5)

                # Draw text background
                text_bg_coords = [
                    text_x - 2,
                    text_y - 2,
                    text_x + text_width + 2,
                    text_y + text_height + 2,
                ]
                draw.rectangle(text_bg_coords, fill=(255, 255, 255))

                # Draw text
                draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

                # Save the processed image
                output_filename = f"shape_{row['shape_id']}_{filename}"
                output_path = os.path.join(str(self.report_id), output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                square_img.save(output_path)

                # Add image reference to list using HTML with legend
                image_references.append(
                    f'<div style="display: inline-block; text-align: center; margin: 5px;">'
                    f'<img src="{output_path}" alt="Shape {row["shape_id"]} - {label}" width="{image_size}" height="{image_size}"/>'
                    f'<br><small style="font-size: 10px; color: #666;">{filename}</small>'
                    f"</div>"
                )

            except Exception as e:
                logger.warning(
                    f"Error processing shape {row.get('shape_id', 'unknown')}: {e}"
                )
                continue

        # Format images in a grid with max 5 images per line
        if image_references:
            markdown_content = "\n"
            for i in range(0, len(image_references), max_images_per_line):
                line_images = image_references[i : i + max_images_per_line]
                markdown_content += " ".join(line_images) + "\n"
            markdown_content += "\n"
            return markdown_content

        return ""
