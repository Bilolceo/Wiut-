"""Video decoding and frame sampling (CPU / NVDEC backends)."""
from src.io.decode import Frame, VideoMeta, read_meta, sample_frames

__all__ = ["Frame", "VideoMeta", "read_meta", "sample_frames"]
