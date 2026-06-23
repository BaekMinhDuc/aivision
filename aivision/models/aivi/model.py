
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from aivision.data.build import load_inference_source
from aivision.engine.model import Model
from aivision.models import aivi
from aivision.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    SemanticSegmentationModel,
    WorldModel,
    AIVIEModel,
    AIVIESegModel,
)
from aivision.utils import ROOT, YAML


class AIVI(Model):
    """AIVI (You Only Look Once) object detection model.

    This class provides a unified interface for AIVI models, automatically switching to specialized model types
    (AIVIWorld or AIVIE) based on the model filename. It supports various computer vision tasks including object
    detection, instance segmentation, semantic segmentation, classification, pose estimation, and oriented bounding box
    detection.

    Attributes:
        model: The loaded AIVI model instance.
        task: The task type (detect, segment, semantic, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a AIVI model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained AIVI26n detection model
        >>> model = AIVI("aivi26n.pt")

        Load a pretrained AIVI26n segmentation model
        >>> model = AIVI("aivi26n-seg.pt")

        Initialize from a YAML configuration
        >>> model = AIVI("aivi26n.yaml")
    """

    def __init__(self, model: str | Path = "aivi26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a AIVI model.

        This constructor initializes a AIVI model, automatically switching to specialized model types (AIVIWorld or
        AIVIE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'aivi26n.pt', 'aivi26n.yaml'.
            task (str, optional): AIVI task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'. Defaults
                to auto-detection based on model.
            verbose (bool): Display model info on load.
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if AIVIWorld PyTorch model
            new_instance = AIVIWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "aivie" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if AIVIE PyTorch model
            new_instance = AIVIE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default AIVI initialization
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from aivision import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": aivi.classify.ClassificationTrainer,
                "validator": aivi.classify.ClassificationValidator,
                "predictor": aivi.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": aivi.detect.DetectionTrainer,
                "validator": aivi.detect.DetectionValidator,
                "predictor": aivi.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": aivi.segment.SegmentationTrainer,
                "validator": aivi.segment.SegmentationValidator,
                "predictor": aivi.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": aivi.pose.PoseTrainer,
                "validator": aivi.pose.PoseValidator,
                "predictor": aivi.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": aivi.obb.OBBTrainer,
                "validator": aivi.obb.OBBValidator,
                "predictor": aivi.obb.OBBPredictor,
            },
            "semantic": {
                "model": SemanticSegmentationModel,
                "trainer": aivi.semantic.SemanticSegmentationTrainer,
                "validator": aivi.semantic.SemanticSegmentationValidator,
                "predictor": aivi.semantic.SemanticSegmentationPredictor,
            },
        }


class AIVIWorld(Model):
    """AIVI-World object detection model.

    AIVI-World is an open-vocabulary object detection model that can detect objects based on text descriptions without
    requiring training on specific classes. It extends the AIVI architecture to support real-time open-vocabulary
    detection.

    Attributes:
        model: The loaded AIVI-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize AIVIv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a AIVIv8-World model
        >>> model = AIVIWorld("aiviv8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "aiviv8s-world.pt", verbose: bool = False) -> None:
        """Initialize AIVIv8-World model with a pre-trained model file.

        Loads a AIVIv8-World model for object detection. If no custom class names are provided, it assigns default COCO
        class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": aivi.detect.DetectionValidator,
                "predictor": aivi.detect.DetectionPredictor,
                "trainer": aivi.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class AIVIE(Model):
    """AIVIE object detection and segmentation model.

    AIVIE is an enhanced AIVI model that supports both object detection and instance segmentation tasks with improved
    performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded AIVIE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize AIVIE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the AIVIE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a AIVIE segmentation model
        >>> model = AIVIE("aivie-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(self, model: str | Path = "aivie-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """Initialize AIVIE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": AIVIEModel,
                "validator": aivi.aivie.AIVIEDetectValidator,
                "predictor": aivi.detect.DetectionPredictor,
                "trainer": aivi.aivie.AIVIETrainer,
            },
            "segment": {
                "model": AIVIESegModel,
                "validator": aivi.aivie.AIVIESegValidator,
                "predictor": aivi.segment.SegmentationPredictor,
                "trainer": aivi.aivie.AIVIESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, AIVIEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires that the
        model is an instance of AIVIEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = AIVIE("aivie-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, AIVIEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        """Set vocabulary and class names for the AIVIE model.

        This method configures the vocabulary and class names used by the model for text processing and classification
        tasks. The model must be an instance of AIVIEModel.

        Args:
            vocab (list[str]): Vocabulary list containing tokens or words used by the model for text processing.
            names (list[str]): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of AIVIEModel.

        Examples:
            >>> model = AIVIE("aivie-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, AIVIEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, AIVIEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor, optional): Embeddings corresponding to the classes.
        """
        # Verify no background class is present
        assert " " not in classes
        assert isinstance(self.model, AIVIEModel)
        if sorted(list(self.model.names.values())) != sorted(classes):
            if embeddings is None:
                embeddings = self.get_text_pe(classes)  # generate text embeddings if not provided
            self.model.set_classes(classes, embeddings)

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = self.model.names

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: str | None = None,
        **kwargs,
    ):
        """Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict[str, list] = {},
        refer_image=None,
        predictor=aivi.aivie.AIVIEVPDetectPredictor,
        **kwargs,
    ):
        """Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths, directory
                paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a generator as they
                are computed.
            visual_prompts (dict[str, list]): Dictionary containing visual prompts for the model. Must include 'bboxes'
                and 'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable): Custom predictor class for visual prompt predictions. Defaults to
                AIVIEVPDetectPredictor.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (list | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = AIVIE("aivie-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", False),
                        "imgsz": kwargs.get("imgsz", self.overrides.get("imgsz", 640)),
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # NOTE: set the first frame as refer image for videos/streams inference
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, aivi.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # reset predictor
        elif isinstance(self.predictor, aivi.aivie.AIVIEVPDetectPredictor):
            self.predictor = None  # reset predictor if no visual prompts
        self.overrides["agnostic_nms"] = True  # use agnostic nms for AIVIE default

        return super().predict(source, stream, **kwargs)
