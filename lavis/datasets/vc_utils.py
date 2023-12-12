''' VisualCOMET utils
Includes functions for preprocessing images
'''
from tqdm import tqdm
from PIL import Image, ImageFile, ImageDraw
import colorsys
import numpy as np
import json

def read_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip()))
    return data

# image processing
def crop_image(image, bbox):
    return image.crop((bbox[0], bbox[1], bbox[2], bbox[3])) 

def highlight_region(image, region, mode, fill, outline):
    if mode == 'boxes':
        return highlight_bbox(image, region, fill=fill, outline=outline)
    elif mode == 'segms':
        return highlight_polygon(image, region, fill=fill, outline=outline)
    else:
        raise ValueError(f"draw mode:{mode} is not supported. Should be one of ['boxes', 'segms']")

def highlight_bbox(image, bbox, fill='#ff05cd3c', outline='#05ff37ff'):
    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, '#00000000')
    draw = ImageDraw.Draw(overlay, 'RGBA')
    x = bbox[0]
    y = bbox[1]
    draw.rectangle([(x, y), (bbox[2], bbox[3])],
                    fill=fill, outline=outline, width=2)
    
    image = Image.alpha_composite(image, overlay)
    return image.convert('RGB')

def highlight_polygon(image, polygon, fill='#ff05cd3c', outline='#05ff37ff'):
    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, '#00000000')
    if len(polygon) != 0:
        draw = ImageDraw.Draw(overlay, 'RGBA')
        polygon = np.vstack(polygon).flatten().tolist()
        draw.polygon(polygon, fill=fill, outline=outline, width=2)
    
    image = Image.alpha_composite(image, overlay)
    return image.convert('RGB')

def draw_PersonX(image, coord, mode='boxes'):
    return highlight_region(image, coord, mode, fill='#ff05cd3c', outline='#05ff37ff')

def draw_PersonY(image, coord, mode='boxes'):
    return highlight_region(image, coord, mode, fill='#ff80004e', outline='#99ccffff')

def draw_PersonZ(image, coord, mode='boxes'):
    return highlight_region(image, coord, mode, fill='#862d2d7d', outline='#cc99ffff')

## color coding from merlot_reserve
# https://github.com/rowanz/merlot_reserve/blob/c73420a54f5e7d1f9a5b62846d73cd8bc9f6898b/finetune/vcr/prep_data.py
def get_color(color_id):
    hue = (color_id % 1024) / 1024
    sat = (color_id % 1023) / 1023

    # luminosity around [0.5, 1.0] for border
    l_start = 0.4
    l_offset = ((color_id % 1025) / 1025)
    lum = l_offset * (1.0 - l_start) + l_start

    color_i = tuple((np.array(colorsys.hls_to_rgb(hue, lum, sat)) * 255.0).astype(np.int32).tolist())

    fill = color_i + (32,)
    outline = color_i + (255,)

    return fill, outline
     
def draw_region(image, coord, color_id, mode='boxes'):
    fill_outline_colors = {
        0: ('#ff05cd20', '#05ff37ff'), 
        1: ('#ff800020', '#99ccffff') , 
        2: ('#862d2d20', '#cc99ffff'),
        3: ('#ff800020', '#f5ad42ff'),
        4: ('#862d2d20', '#f55d42ff'),
    }
    if color_id in fill_outline_colors:
        fill_color, outline_color = fill_outline_colors[color_id]
    else:
        fill_color, outline_color = get_color(color_id)
    return highlight_region(image, coord, mode, fill=fill_color, outline=outline_color)

def draw_regions(image, region_references, mode):
    """ draws up to 3 regions ihe image"""
    for ref_idx, ref in enumerate(region_references):        
        region = ref[mode][0]
        if ref_idx == 0:
            image = draw_PersonX(image, region, mode=mode)
        elif ref_idx == 1:
            image = draw_PersonY(image, region, mode=mode)
        elif ref_idx == 2:
            image = draw_PersonZ(image, region, mode=mode)
    
    return image

# unified annotation processing
def split_qa(qa):
    question = qa.split('?')[0]
    question = question+'?'
    answer = '?'.join(qa.split('?')[1:])
    
    return question, answer

def tokens2str(tokens):
    return ' '.join([str(w) for w in  tokens])

def get_references(datum, is_train=True):
    references = {}
    if isinstance(datum['references'], dict):
        references = datum['references'] 
    elif isinstance(datum['references_in_input'], dict):
        references = datum['references_in_input']
    references = {int(k): v for k,v in references.items()} # change key to int...
    return references

def get_region_references(references, tokens):
    """
    get references in the order of it appears in the tokens
    Args:
        references (Dict): reference metadata info
        tokens (List): word tokens that also contains list IDs.

    Returns:
        List[Dict]: List of references sorted based on the order they appear in tokens.
    """
    parsed_reference = []
    region_references = []
    for w in tokens:
        if isinstance(w, list):
            for id in w:
                if id in references and id not in parsed_reference:
                    parsed_reference.append(id)
                    region_references.append(references[id])
    return region_references

def add_tags(tokens, refinfo, use_descriptions=False, use_object_tags=False, reorder_ids=True, no_ids=False):
    """
    Replace ID tag (type: List) with the corresponding tags in string.
    - use_descriptions: Use region description to fill in the ID tags.
    - use_tags: Use guessed object to fill in the tags. Usually used for VCR train/evaluation (disabled if use_descriptions is True.)
    - reorder_ids: Assign IDs based on the order they appear. example: [2] hugs [0] -> [0] hugs [1]. (disabled if use_descriptions is True.)
    """
    # assumes text is tokenized with ids 
    text_tags = []
    seen_ids = []
    for w in tokens:
        if isinstance(w, list):
            w_tag = []
            for id in w:
                if id in refinfo:
                    
                    # explicit description
                    if use_descriptions:
                        w_tag.append(refinfo[id]['description'])
                        
                    # id tags
                    else:
                        if reorder_ids:
                            if 'index' in refinfo[id]:  # use the pre-defined index key to refer to the regions.
                                w_id = refinfo[id]['index']
                            else:  # ids in consectuive order
                                if id not in seen_ids:
                                    seen_ids.append(id)
                                w_id = seen_ids.index(id)
                        else:
                            w_id = id
                        if use_object_tags:
                            if no_ids:
                                w_tag.append("{}".format(refinfo[id]["obj_guess"]))
                            else:
                                w_tag.append("{} [{}]".format(refinfo[id]["obj_guess"], w_id))
                        else:
                            w_tag.append("[{}]".format(w_id))
            w = ' , '.join(w_tag)
        text_tags.append(w)
    return text_tags

def add_sherlock_tags(tokens, refinfo, use_descriptions=False, reorder_ids=True):
    """ add_tags but deals with only one region"""
    text_tags = []
    for w in tokens:
        if isinstance(w, list):
            for id in refinfo: # only 1 region should be present in sherlock
                # ex    plicit description
                if use_descriptions:
                    w = refinfo[id]['description']
                # id tags
                else:
                    if reorder_ids:
                        w = "[0]"
                    else:
                        w = "[{}]".format(id)
        text_tags.append(w) 
    return text_tags