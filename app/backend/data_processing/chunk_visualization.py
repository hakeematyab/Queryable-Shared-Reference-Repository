from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def draw_chunk_bboxes(
    chunk: Dict,
    images_dir: str,
    output_path: str = None,
    color: str = "green",
    thickness: int = 3,
) -> Image.Image:
    metadata = chunk.get("metadata", chunk)
    bboxes = metadata.get("bboxes", [])
    pages = metadata.get("page_numbers", [])
    doc_id = metadata.get("doc_id", "")
    
    if not pages or not bboxes:
        raise ValueError("Chunk has no page/bbox info")
    
    page_no = pages[0]
    img_path = Path(images_dir) / f"{doc_id}_page_{page_no}.jpg"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Page image not found: {img_path}")
    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    

    font = ImageFont.load_default()
    
    img_width, img_height = img.size
    page_bboxes = [b for b in bboxes if b.get('page_no') == page_no]
    
    for bbox in page_bboxes:
        l, t, r, b = bbox['l'], bbox['t'], bbox['r'], bbox['b']
        coord_origin = bbox.get('coord_origin', 'BOTTOMLEFT')
        
        if 'BOTTOMLEFT' in coord_origin.upper():
            scale = img_height / 842 
            x1, x2 = l * scale, r * scale
            y1 = img_height - (t * scale)
            y2 = img_height - (b * scale)
            if y1 > y2:
                y1, y2 = y2, y1
        else:
            x1, y1, x2, y2 = l, t, r, b
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)
    
    if output_path:
        img.save(output_path)
    
    return img

def visualize_search_results(
    results: List[Dict],
    images_dir: str,
    output_dir: str = "./visualizations",
) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    for i, chunk in enumerate(results):
        try:
            out_path = output_dir / f"result_{i}.jpg"
            draw_chunk_bboxes(chunk, images_dir, str(out_path))
            saved.append(str(out_path))
        except Exception as e:
            logger.warning(f"Could not visualize result {i}: {e}")
    return saved

