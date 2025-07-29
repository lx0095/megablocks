# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

def topology(padded_bins, block_size, output_block_rows, output_block_columns, out=None):
    num_bins = padded_bins.numel()
    total_elements = output_block_rows * output_block_columns
    
    # 若未提供输出张量则创建
    if out is None:
        out = torch.empty(total_elements, dtype=torch.int16, device=padded_bins.device)
    
    # 空张量直接返回
    if total_elements == 0:
        return out
    
    # 计算每个bin的起始和结束块索引
    starts = torch.zeros_like(padded_bins)
    starts[1:] = padded_bins[:-1]
    starts = starts // block_size
    ends = padded_bins // block_size
    
    # 为每个bin生成索引
    for bin_idx in range(num_bins):
        start = starts[bin_idx]
        end = ends[bin_idx]
        num_rows = end - start
        
        if num_rows <= 0:
            continue
        
        # 计算当前bin在输出张量中的位置
        bin_offset = start * output_block_columns
        bin_length = num_rows * output_block_columns
        
        # 确保不会超出输出张量范围
        if bin_offset >= total_elements:
            break
        if bin_offset + bin_length > total_elements:
            bin_length = total_elements - bin_offset
        
        # 生成列索引 [0, 1, ..., output_block_columns-1]
        col_indices = torch.arange(output_block_columns, dtype=torch.int16, device=padded_bins.device)
        
        # 生成行偏移 [0, output_block_columns, 2*output_block_columns, ...]
        row_offsets = (torch.arange(num_rows, dtype=torch.int16, device=padded_bins.device) * 
                      output_block_columns)
        
        # 生成基础索引 (bin_idx * output_block_columns)
        base_idx = torch.tensor(bin_idx * output_block_columns, dtype=torch.int16, device=padded_bins.device)
        
        # 计算完整索引矩阵并展平
        indices_matrix = base_idx + col_indices + row_offsets.unsqueeze(1)
        flat_indices = indices_matrix.flatten()
        
        # 将生成的索引写入输出张量
        out[bin_offset:bin_offset+bin_length] = flat_indices[:bin_length]
    
    return out
