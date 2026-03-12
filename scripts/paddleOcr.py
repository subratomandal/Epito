#!/usr/bin/env python3

import sys
import json
import os


def get_ocr_engine(lang="en"):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
    )


def sort_blocks_reading_order(blocks):
    if not blocks:
        return []

    blocks.sort(key=lambda b: b["center_y"])

    lines = []
    current_line = [blocks[0]]
    threshold = blocks[0]["height"] * 0.5 if blocks[0]["height"] > 0 else 15

    for block in blocks[1:]:
        if abs(block["center_y"] - current_line[-1]["center_y"]) < threshold:
            current_line.append(block)
        else:
            current_line.sort(key=lambda b: b["center_x"])
            lines.append(current_line)
            current_line = [block]
            threshold = block["height"] * 0.5 if block["height"] > 0 else 15

    current_line.sort(key=lambda b: b["center_x"])
    lines.append(current_line)

    paragraphs = []
    current_para = []
    prev_y = None

    for line in lines:
        line_text = " ".join(b["text"] for b in line)
        line_y = sum(b["center_y"] for b in line) / len(line)
        line_h = max(b["height"] for b in line) if line else 15

        if prev_y is not None and (line_y - prev_y) > line_h * 1.8:
            if current_para:
                paragraphs.append(" ".join(current_para))
            current_para = []

        current_para.append(line_text)
        prev_y = line_y

    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs


def parse_ocr_result(result):
    if not result or not result[0]:
        return [], ""

    blocks = []
    for line in result[0]:
        bbox, (text, confidence) = line
        center_y = (bbox[0][1] + bbox[2][1]) / 2
        center_x = (bbox[0][0] + bbox[2][0]) / 2
        height = abs(bbox[2][1] - bbox[0][1])

        blocks.append(
            {
                "text": text,
                "confidence": float(confidence),
                "center_y": center_y,
                "center_x": center_x,
                "height": height,
            }
        )

    paragraphs = sort_blocks_reading_order(blocks)
    text = "\n\n".join(paragraphs)

    return blocks, text


def process_image(file_path, lang="en"):
    ocr = get_ocr_engine(lang)
    result = ocr.ocr(file_path, cls=True)
    blocks, text = parse_ocr_result(result)

    return {
        "text": text,
        "block_count": len(blocks),
        "page_count": 1,
    }


def process_pdf(file_path, lang="en"):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return {"error": "pdf2image not installed. Run: pip install pdf2image"}

    import tempfile

    ocr = get_ocr_engine(lang)

    try:
        images = convert_from_path(file_path, dpi=300)
    except Exception as e:
        return {"error": f"PDF conversion failed: {str(e)}"}

    pages = []
    full_parts = []

    for i, image in enumerate(images):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, "PNG")
            tmp_path = tmp.name

        try:
            result = ocr.ocr(tmp_path, cls=True)
        finally:
            os.unlink(tmp_path)

        blocks, page_text = parse_ocr_result(result)

        pages.append(
            {
                "page": i + 1,
                "text": page_text,
                "block_count": len(blocks),
            }
        )

        full_parts.append(f"Page {i + 1}\n{page_text}")

    return {
        "text": "\n\n".join(full_parts),
        "pages": pages,
        "page_count": len(pages),
    }


def main():
    if len(sys.argv) < 3:
        json.dump(
            {"success": False, "error": "Usage: paddleOcr.py <image|pdf> <file_path>"},
            sys.stdout,
        )
        sys.exit(1)

    file_type = sys.argv[1]
    file_path = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"

    if not os.path.exists(file_path):
        json.dump(
            {"success": False, "error": f"File not found: {file_path}"}, sys.stdout
        )
        sys.exit(1)

    try:
        if file_type == "image":
            result = process_image(file_path, lang)
        elif file_type == "pdf":
            result = process_pdf(file_path, lang)
        else:
            json.dump(
                {"success": False, "error": f"Unknown type: {file_type}"}, sys.stdout
            )
            sys.exit(1)

        if "error" in result:
            json.dump({"success": False, "error": result["error"]}, sys.stdout)
            sys.exit(1)

        json.dump(
            {
                "success": True,
                "text": result.get("text", ""),
                "page_count": result.get("page_count", 1),
                "pages": result.get("pages", []),
            },
            sys.stdout,
        )
    except Exception as e:
        json.dump({"success": False, "error": str(e)}, sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
