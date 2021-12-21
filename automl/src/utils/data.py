"""Utils for model compression.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import random
from multiprocessing import Pool
from typing import Tuple


def get_rand_bbox_coord(
    w: int, h: int, len_ratio: float
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get a coordinate of random box."""
    size_hole_w = int(len_ratio * w)
    size_hole_h = int(len_ratio * h)
    x = random.randint(0, w)  # [0, w]
    y = random.randint(0, h)  # [0, h]

    x0 = max(0, x - size_hole_w // 2)
    y0 = max(0, y - size_hole_h // 2)
    x1 = min(w, x + size_hole_w // 2)
    y1 = min(h, y + size_hole_h // 2)
    return (x0, y0), (x1, y1)

def weights_for_balanced_classes(subset, nclasses):                        
    count = [0] * nclasses                            
    for i in subset:                                                         
        count[i[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weightget_rand_bbox_coord