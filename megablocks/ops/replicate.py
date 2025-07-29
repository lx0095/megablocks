# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# Autograd wrapper for replicate kernel.
class ReplicateOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bins, num_outputs):
        ctx.num_bins = num_bins
        ctx.save_for_backward(bins)
        batch_size, num_bins = x.shape
        columns = num_outputs
        
        # 创建输出张量（如果未提供）
        if out is None:
            out = torch.empty(batch_size, columns, dtype=x.dtype, device=x.device)

        # 处理空输入的情况
        if out.numel() == 0:
            return out
        
        # 计算每个bin的起始和结束索引
        bin_starts = torch.zeros_like(bins)
        if num_bins > 1:
            bin_starts[1:] = bins[:-1]
        bin_ends = bins
        
        # 对每个批次和每个bin进行复制操作
        for batch_idx in range(batch_size):
            for bin_idx in range(num_bins):
                # 获取当前bin的值
                value = x[batch_idx, bin_idx]
                
                # 获取当前bin的范围
                start = bin_starts[bin_idx].item()
                end = bin_ends[bin_idx].item()
                
                # 将值复制到输出的对应范围
                out[batch_idx, start:end] = value
        
        return out

    @staticmethod
    def backward(ctx, grad):
        batch_size, columns = grad.shape
        num_bins = ctx.num_bins
        bins = ctx.saved_tensors

        # 创建输出张量（如果未提供）
        if out is None:
            out = torch.zeros(batch_size, num_bins, dtype=grad.dtype, device=grad.device)

        # 处理空输入的情况
        if grad.numel() == 0:
            return out
        
        # 计算每个bin的起始和结束索引
        bin_starts = torch.zeros_like(bins)
        if num_bins > 1:
            bin_starts[1:] = bins[:-1]
        bin_ends = bins
        
        # 对每个批次和每个bin进行求和操作
        for batch_idx in range(batch_size):
            for bin_idx in range(num_bins):
                # 获取当前bin的范围
                start = bin_starts[bin_idx].item()
                end = bin_ends[bin_idx].item()
                
                # 对该范围内的梯度求和，作为输入梯度
                out[batch_idx, bin_idx] = grad[batch_idx, start:end].sum()
        
        return out
replicate = ReplicateOp.apply
