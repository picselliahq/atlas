from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import NamedTuple


class SectionName(StrEnum):
    DATASET_OVERVIEW = "Dataset Overview"
    IMAGE_QUALITY = "Image Quality"
    ANNOTATION_QUALITY = "Annotation Quality"


class SubSectionData(NamedTuple):
    section: SectionName
    name: str


class SubSectionName(Enum):
    # === DATASET OVERVIEW ===
    OVERVIEW = SubSectionData(SectionName.DATASET_OVERVIEW, "Overview")

    # === IMAGE QUALITY ===
    IMAGE_STATISTICS_ANALYSIS = SubSectionData(
        SectionName.IMAGE_QUALITY, "Image Statistics Analysis"
    )
    IMAGE_QUALITY_ISSUES = SubSectionData(
        SectionName.IMAGE_QUALITY, "Image Quality Issues"
    )

    # === ANNOTATION QUALITY ===
    CLASS_ANALYSIS = SubSectionData(SectionName.ANNOTATION_QUALITY, "Class Analysis")
    SINGLE_OBJECT_ANALYSIS = SubSectionData(
        SectionName.ANNOTATION_QUALITY, "Single Object Analysis"
    )
    INTER_CLASS_RELATION_ANALYSIS = SubSectionData(
        SectionName.ANNOTATION_QUALITY, "Inter-Class Relation Analysis"
    )
    ANNOTATION_OUTLIERS_ANALYSIS = SubSectionData(
        SectionName.ANNOTATION_QUALITY, "Annotation Outlier Detection"
    )

    @property
    def section(self):
        return self.value.section

    @property
    def name(self):
        return self.value.name


@dataclass(frozen=True)
class ContentLocator:
    section: str
    sub_section: str
    content: str


class ReportContentName(Enum):
    # === DATASET OVERVIEW ===
    METADATA_OVERVIEW = ContentLocator(
        section=SubSectionName.OVERVIEW.section,
        sub_section=SubSectionName.OVERVIEW.name,
        content="Metadata Overview",
    )

    IMAGE_STATISTICS = ContentLocator(
        section=SubSectionName.IMAGE_STATISTICS_ANALYSIS.section,
        sub_section=SubSectionName.IMAGE_STATISTICS_ANALYSIS.name,
        content="Image Statistics Analysis",
    )

    LUMINANCE_ISSUES = ContentLocator(
        section=SubSectionName.IMAGE_QUALITY_ISSUES.section,
        sub_section=SubSectionName.IMAGE_QUALITY_ISSUES.name,
        content="Luminance Issues",
    )
    CONTRAST_ISSUES = ContentLocator(
        section=SubSectionName.IMAGE_QUALITY_ISSUES.section,
        sub_section=SubSectionName.IMAGE_QUALITY_ISSUES.name,
        content="Contrast Issues",
    )
    BLUR_ISSUES = ContentLocator(
        section=SubSectionName.IMAGE_QUALITY_ISSUES.section,
        sub_section=SubSectionName.IMAGE_QUALITY_ISSUES.name,
        content="Blurry Images",
    )
    CLIP_OUTLIER_ANALYSIS = ContentLocator(
        section=SubSectionName.IMAGE_QUALITY_ISSUES.section,
        sub_section=SubSectionName.IMAGE_QUALITY_ISSUES.name,
        content="CLIP Outlier Analysis",
    )
    CLIP_DUPLICATE_ANALYSIS = ContentLocator(
        section=SubSectionName.IMAGE_QUALITY_ISSUES.section,
        sub_section=SubSectionName.IMAGE_QUALITY_ISSUES.name,
        content="CLIP Duplicate Analysis",
    )

    # === ANNOTATION QUALITY → CLASS DISTRIBUTION ===
    CLASS_DISTRIBUTION = ContentLocator(
        section=SubSectionName.CLASS_ANALYSIS.section,
        sub_section=SubSectionName.CLASS_ANALYSIS.name,
        content="Class Distribution",
    )

    SINGLE_OBJECT_ANALYSIS = ContentLocator(
        section=SubSectionName.SINGLE_OBJECT_ANALYSIS.section,
        sub_section=SubSectionName.SINGLE_OBJECT_ANALYSIS.name,
        content="Single Object Analysis",
    )

    TOP_COOCCURRENCES = ContentLocator(
        section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.section,
        sub_section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name,
        content="Top Class Co-occurrences",
    )
    UNEXPECTED_COOCCURRENCE = ContentLocator(
        section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.section,
        sub_section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name,
        content="Unexpected Co-occurrences",
    )
    MISSING_COOCCURRENCE = ContentLocator(
        section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.section,
        sub_section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name,
        content="Missing Expected Co-occurrences",
    )
    TOP_OVERLAPS = ContentLocator(
        section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.section,
        sub_section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name,
        content="Top Class Overlaps",
    )
    UNEXPECTED_OVERLAP = ContentLocator(
        section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.section,
        sub_section=SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name,
        content="Unexpected Overlaps",
    )

    OUTLIER_DENSITY = ContentLocator(
        section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.section,
        sub_section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.name,
        content="Shape Density Outliers",
    )
    OUTLIER_SHAPE_AREA = ContentLocator(
        section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.section,
        sub_section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.name,
        content="Area-Based Shape Outliers",
    )
    OUTLIER_ASPECT_RATIO = ContentLocator(
        section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.section,
        sub_section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.name,
        content="Aspect Ratio Outliers",
    )
    INTRA_CLASS_CENTROID_OUTLIERS = ContentLocator(
        section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.section,
        sub_section=SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.name,
        content="Intra-Class Centroid Distance Outliers",
    )

    @property
    def section(self):
        return self.value.section

    @property
    def sub_section(self):
        return self.value.sub_section

    @property
    def content(self):
        return self.value.content
