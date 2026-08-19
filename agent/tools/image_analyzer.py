"""图片分析工具。

支持分析本地图片的基本信息（尺寸、格式、大小）和内容摘要。
如果 PIL/Pillow 可用，提供更丰富的分析（主色调、尺寸、EXIF 等）。
配合 GUI 文件上传功能使用。
"""

from __future__ import annotations

import os
import struct

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve

_HAS_PIL = False
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    pass


_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".webp", ".tiff", ".tif", ".ico", ".ppm",
    ".pgm", ".pbm",
})

_MAX_IMAGE_SIZE = 20 * 1024 * 1024

_MAGIC_BYTES = {
    b"\xff\xd8": "JPEG",
    b"\x89PNG": "PNG",
    b"GIF87": "GIF",
    b"GIF89": "GIF",
    b"BM": "BMP",
    b"RIFF": "WEBP",
    b"II\x2a\x00": "TIFF",
    b"MM\x00\x2a": "TIFF",
}


def _detect_format_by_magic(data: bytes) -> str | None:
    for magic, fmt in _MAGIC_BYTES.items():
        if data[:len(magic)] == magic:
            return fmt
    return None


def _get_image_metadata(filepath: str) -> dict:
    meta = {
        "format": "未知",
        "width": 0,
        "height": 0,
        "size": 0,
        "mode": "",
        "has_exif": False,
        "error": None,
    }

    try:
        file_size = os.path.getsize(filepath)
        meta["size"] = file_size

        ext = os.path.splitext(filepath)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            meta["format"] = ext.lstrip(".").upper()

        with open(filepath, "rb") as f:
            header = f.read(32)

        magic_fmt = _detect_format_by_magic(header)
        if magic_fmt:
            meta["format"] = magic_fmt

        if meta["format"] == "JPEG" and len(header) >= 4:
            try:
                f_size = os.path.getsize(filepath)
                with open(filepath, "rb") as f:
                    while True:
                        marker = f.read(2)
                        if len(marker) < 2:
                            break
                        if marker[0:1] != b"\xff":
                            continue
                        seg_type = marker[1]
                        if seg_type == 0xD9 or seg_type == 0xDA:
                            break
                        length_bytes = f.read(2)
                        if len(length_bytes) < 2:
                            break
                        length = struct.unpack(">H", length_bytes)[0]
                        if seg_type in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                            data = f.read(5)
                            if len(data) == 5:
                                meta["height"] = struct.unpack(">H", data[1:3])[0]
                                meta["width"] = struct.unpack(">H", data[3:5])[0]
                            break
                        else:
                            f.seek(length - 2, 1)
            except Exception:
                pass

        elif meta["format"] == "PNG" and len(header) >= 24:
            try:
                width = struct.unpack(">I", header[16:20])[0]
                height = struct.unpack(">I", header[20:24])[0]
                meta["width"] = width
                meta["height"] = height
            except Exception:
                pass

        elif meta["format"] == "GIF" and len(header) >= 10:
            try:
                width = struct.unpack("<H", header[6:8])[0]
                height = struct.unpack("<H", header[8:10])[0]
                meta["width"] = width
                meta["height"] = height
            except Exception:
                pass

        elif meta["format"] == "BMP" and len(header) >= 26:
            try:
                width = struct.unpack("<i", header[18:22])[0]
                height = struct.unpack("<i", header[22:26])[0]
                meta["width"] = abs(width)
                meta["height"] = abs(height)
            except Exception:
                pass

    except Exception as exc:
        meta["error"] = str(exc)

    return meta


def _analyze_with_pil(filepath: str, meta: dict) -> dict:
    if not _HAS_PIL:
        return meta

    try:
        img = Image.open(filepath)
        meta["width"] = img.width
        meta["height"] = img.height
        meta["mode"] = img.mode
        meta["format"] = img.format or meta["format"]

        try:
            exif = img._getexif()
            if exif:
                meta["has_exif"] = True
                exif_summary = []
                for tag_id, value in list(exif.items())[:10]:
                    tag_name = Image.ExifTags.TAGS.get(tag_id, str(tag_id))
                    val_str = str(value)[:80]
                    exif_summary.append(f"{tag_name}: {val_str}")
                meta["exif"] = exif_summary
        except Exception:
            pass

        try:
            img_small = img.copy()
            img_small.thumbnail((80, 80))
            colors = img_small.getcolors(maxcolors=100000)
            if colors:
                colors_sorted = sorted(colors, reverse=True)[:5]
                total = sum(c for c, _ in colors)
                dominant = []
                for count, rgb in colors_sorted:
                    pct = count / total * 100 if total else 0
                    dominant.append({
                        "rgb": f"rgb{rgb}",
                        "percentage": round(pct, 1),
                        "count": count,
                    })
                meta["dominant_colors"] = dominant
        except Exception:
            pass

        try:
            if img.mode in ("RGB", "RGBA", "L"):
                extrema = img.getextrema()
                meta["extrema"] = str(extrema)
        except Exception:
            pass

    except Exception as exc:
        meta["pil_error"] = str(exc)

    return meta


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


@register(tags={"file"})
@tool
def image_analyze(filepath: str) -> str:
    """分析图片文件，返回尺寸、格式、大小等信息。"""
    try:
        safe = safe_resolve(filepath, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe):
        return f"❌ 文件不存在：{filepath}"

    ext = os.path.splitext(safe)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
        return f"❌ 不支持的图片格式：{ext}。支持：{', '.join(sorted(_IMAGE_EXTENSIONS))}"

    file_size = os.path.getsize(safe)
    if file_size > _MAX_IMAGE_SIZE:
        return f"❌ 图片过大（{_format_size(file_size)}），最大支持 {_format_size(_MAX_IMAGE_SIZE)}"

    meta = _get_image_metadata(safe)

    if meta["format"] != "未知" or meta["width"] > 0:
        pass

    if _HAS_PIL:
        meta = _analyze_with_pil(safe, meta)

    lines = [
        "🖼️ 图片分析结果：",
        f"  文件：{filepath}",
        f"  格式：{meta['format']}",
        f"  尺寸：{meta['width']} x {meta['height']} 像素",
        f"  大小：{_format_size(meta['size'])}",
    ]

    if meta.get("mode"):
        lines.append(f"  色彩模式：{meta['mode']}")

    if meta.get("has_exif"):
        lines.append(f"  EXIF：✅ 存在")
        if meta.get("exif"):
            lines.append("  EXIF 详情：")
            for item in meta["exif"][:6]:
                lines.append(f"    - {item}")

    if meta.get("dominant_colors"):
        lines.append("  主色调 Top 5：")
        for i, c in enumerate(meta["dominant_colors"], 1):
            lines.append(f"    {i}. {c['rgb']}  占比 {c['percentage']}%")

    if meta.get("error"):
        lines.append(f"  ⚠️ 部分解析失败：{meta['error']}")

    if not _HAS_PIL:
        lines.append("  💡 安装 Pillow 可获得更详细的分析：pip install Pillow")

    return "\n".join(lines)


@register(tags={"file"})
@tool
def image_batch_analyze(directory: str) -> str:
    """批量分析目录下的所有图片。"""
    try:
        safe_dir = safe_resolve(directory, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isdir(safe_dir):
        return f"❌ 目录不存在：{directory}"

    images: list[tuple[str, dict]] = []
    for fname in sorted(os.listdir(safe_dir)):
        full = os.path.join(safe_dir, fname)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _IMAGE_EXTENSIONS:
            continue

        rel = os.path.relpath(full, os.path.realpath(settings.FILE_READER_ROOT))
        file_size = os.path.getsize(full)
        if file_size > _MAX_IMAGE_SIZE:
            images.append((rel, {"format": "跳过（过大）", "width": 0, "height": 0, "size": file_size}))
            continue

        meta = _get_image_metadata(full)
        if _HAS_PIL:
            meta = _analyze_with_pil(full, meta)
        images.append((rel, meta))

    if not images:
        return f"📁 目录 {directory} 中未找到图片文件"

    lines = [f"📊 批量分析结果（{directory}）：", ""]
    total_size = 0
    for rel, meta in images:
        w = meta.get("width", 0)
        h = meta.get("height", 0)
        fmt = meta.get("format", "?")
        sz = meta.get("size", 0)
        total_size += sz
        dim = f"{w}x{h}" if w and h else "?"
        lines.append(f"  📄 {rel}")
        lines.append(f"     格式: {fmt}  尺寸: {dim}  大小: {_format_size(sz)}")

    lines.append("")
    lines.append(f"共 {len(images)} 张图片，总大小 {_format_size(total_size)}")

    return "\n".join(lines)