# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from aphrodite.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataParser,
    VideoProcessorItems,
)

H, W = 480, 640


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (W, H)),
        # HWC, e.g. from np.array(PIL.Image)
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        # CHW, standard PyTorch / numpy convention
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_image_size_hwc_chw(image):
    """Image sizes must be channel-layout agnostic.

    `get_image_size` determines the multimodal placeholder count; reading an
    HWC array (the layout `np.array(PIL.Image)` produces) as CHW yields a
    bogus size and a placeholder/embedding count mismatch at inference time.
    """
    items = ImageProcessorItems([image])

    assert items.get_image_size(0) == (W, H)


@pytest.mark.parametrize(
    "frame",
    [
        Image.new("RGB", (W, H)),
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_frame_size_hwc_chw(frame):
    """`get_frame_size` must stay consistent with `get_image_size`."""
    items = VideoProcessorItems([[frame]])

    assert items.get_frame_size(0) == (W, H)


class TestNonePlaceholderItems:
    """A media item referenced by UUID arrives as `None` until the cache
    supplies its data.

    Dereferencing it in the parser used to hit `assert_never`, producing an
    `AssertionError` (HTTP 500) instead of a request error. The image path
    always tolerated `None`; video and audio must behave the same way.
    """

    def test_video_none_item_does_not_assert(self):
        items = MultiModalDataParser()._parse_video_data([None])

        assert items is not None
        assert len(items) == 1

    def test_audio_none_item_does_not_assert(self):
        items = MultiModalDataParser()._parse_audio_data([None])

        assert items is not None
        assert len(items) == 1

    def test_image_none_item_unchanged(self):
        """The behaviour video and audio are being aligned to."""
        items = MultiModalDataParser()._parse_image_data([None])

        assert items is not None
        assert len(items) == 1

    def test_video_mixed_none_and_real_frames(self):
        video = np.zeros((1, H, W, 3), dtype=np.uint8)
        items = MultiModalDataParser()._parse_video_data([None, video])

        assert len(items) == 2
        assert items.get_frame_size(1) == (W, H)
