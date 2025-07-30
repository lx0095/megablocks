import torch
import triton
import triton.language as tl

def assert_is_3d_tensor(x):
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor but got {x.ndim}D tensor")


def assert_tensor_dim(x, expected_dim):
    if x.ndim != expected_dim:
        raise ValueError(f"Expected {expected_dim}D tensor, got {x.ndim}D")


def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f"Expected {ndim}-tensor but got {x.ndim}-tensor")


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f"Expected 1-tensor but got {x.ndim}-tensor")


def assert_equal(a, b):
    if a != b:
        raise ValueError(f"Expected dimensions to be equal but got {a} and {b}.")


def gather(x, indices, bin_ids, weights, bins, top_k):
    """
    纯Python实现的gather函数，功能与Triton版本类似
    
    参数:
        x: 输入矩阵，形状为(tokens, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        bin_ids: 分箱ID向量，形状为(tokens * top_k,)
        weights: 权重向量，形状为(tokens * top_k,)，可为None
        bins: 分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        out: 收集后的矩阵，形状为(tokens * top_k, hidden_size)
    """
    # 输入验证
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)
    
    if weights is not None:
        assert_is_vector(weights)
        assert_equal(weights.shape[0], x.shape[0] * top_k)
    
    # 计算输出形状
    output_rows = x.shape[0] * top_k
    hidden_size = x.shape[1]
    out = torch.empty((output_rows, hidden_size), dtype=x.dtype, device=x.device)
    
    # 遍历每个元素进行收集
    for i in range(indices.shape[0]):
        # 获取输入索引
        index_a = indices[i]
        
        # 计算分箱偏移
        bin_idx = bin_ids[i]
        offset_in_bin = i
        if bin_idx > 0:
            offset_in_bin -= bins[bin_idx - 1]
        
        # 计算输出索引
        index_b = offset_in_bin
        if bin_idx > 0:
            index_b += bins[bin_idx - 1]
        
        # 获取输入元素
        input_element = x[index_a // top_k]  # 因为每个token有top_k个索引
        
        # 应用权重（如果提供）
        if weights is not None:
            input_element = input_element * weights[i]
        
        # 存储到输出
        out[index_b] = input_element
    
    return out


def scatter(x, indices, bin_ids, weights, bins, top_k):
    """
    纯Python实现的scatter函数，功能与Triton版本类似
    
    参数:
        x: 输入矩阵（通常是专家处理后的输出），形状为(padded_rows, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        bin_ids: 分箱ID向量，形状为(tokens * top_k,)
        weights: 权重向量，形状为(tokens * top_k,)，可为None
        bins: 分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        out: 聚合后的矩阵，形状为(tokens, hidden_size)
    """
    # 输入验证
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    
    if weights is not None:
        assert_is_vector(weights)
        assert_equal(indices.shape[0], weights.shape[0])
    
    # 计算输出形状
    tokens = indices.shape[0] // top_k
    hidden_size = x.shape[1]
    
    # 初始化中间张量，存储每个token的top_k个结果
    intermediate = torch.empty((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)
    
    # 遍历每个元素进行分散操作
    for i in range(indices.shape[0]):
        # 获取输出索引（对应原始tokens中的位置）
        index_out = indices[i]
        
        # 计算分箱偏移，确定从x中取元素的位置
        bin_idx = bin_ids[i]
        offset_in_bin = i
        if bin_idx > 0:
            offset_in_bin -= bins[bin_idx - 1]
        
        index_x = offset_in_bin
        if bin_idx > 0:
            index_x += bins[bin_idx - 1]
        
        # 确保索引在有效范围内
        if index_x >= x.shape[0]:
            raise IndexError(f"Index {index_x} out of bounds for x with shape {x.shape}")
        
        # 从x中获取元素
        input_element = x[index_x]
        
        # 应用权重（如果提供）
        if weights is not None:
            input_element = input_element * weights[i]
        
        # 计算在中间张量中的位置
        token_idx = index_out // top_k
        k_idx = index_out % top_k
        
        # 存储到中间张量
        intermediate[token_idx, k_idx] = input_element
    
    # 对top_k维度求和，聚合结果
    if top_k > 1:
        out = intermediate.sum(dim=1)
    else:
        out = intermediate.view(tokens, hidden_size)
    
    return out


def scatter_wgrad(x, grad, indices, bin_ids, bins, top_k):
    """
    纯Python实现的scatter_wgrad函数，用于计算权重梯度
    
    参数:
        x: 专家的输出张量，形状为(num_experts, expert_capacity, hidden_size)
        grad: 最终输出的梯度，形状为(tokens, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        bin_ids: 分箱ID向量，形状为(tokens * top_k,)
        bins: 分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        wgrad: 权重的梯度，形状为(tokens * top_k,)
    """
    # 输入验证
    assert_is_tensor(x, 3)  # 专家输出是3D张量
    assert_is_matrix(grad)  # 梯度是2D张量
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.shape[0], x.shape[0])  # bins数量应等于专家数量
    
    # 获取维度信息
    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    
    # 初始化权重梯度张量
    wgrad = torch.zeros(indices.shape[0], dtype=x.dtype, device=x.device)
    
    # 遍历每个元素计算梯度
    for i in range(indices.shape[0]):
        # 获取输出索引（对应权重的位置）
        index_out = indices[i]
        
        # 计算分箱偏移，确定从x中取元素的位置
        bin_idx = bin_ids[i]
        offset_in_bin = i
        if bin_idx > 0:
            offset_in_bin -= bins[bin_idx - 1]
        
        # 计算在专家输出x中的索引
        expert_idx = bin_idx
        entry_idx = offset_in_bin
        
        # 确保索引在有效范围内
        if expert_idx >= num_experts:
            raise IndexError(f"Expert index {expert_idx} out of bounds for {num_experts} experts")
        if entry_idx >= expert_capacity:
            raise IndexError(f"Entry index {entry_idx} out of bounds for capacity {expert_capacity}")
        
        # 获取对应的专家输出和梯度
        expert_output = x[expert_idx, entry_idx]  # 形状: (hidden_size,)
        token_grad = grad[index_out // top_k]     # 形状: (hidden_size,)
        
        # 计算权重梯度：专家输出与梯度的点积
        weight_grad = torch.sum(expert_output * token_grad)
        
        # 存储到权重梯度张量
        wgrad[i] = weight_grad
    
    return wgrad


def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    """
    纯Python实现的binned_gather函数，按专家分箱收集元素
    
    参数:
        x: 输入矩阵，形状为(tokens, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        weights: 权重向量，形状为(tokens * top_k,)，可为None
        bins: 分箱累积计数向量，形状为(num_experts,)
        expert_capacity: 每个专家的最大容量
        top_k: 每个token选择的专家数量
        
    返回:
        out: 按专家分箱的输出张量，形状为(num_experts, expert_capacity, hidden_size)
    """
    # 输入验证
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    
    if weights is not None:
        assert_is_vector(weights)
        assert_equal(weights.shape[0], indices.shape[0])
    
    # 获取维度信息
    num_experts = bins.shape[0]
    tokens, hidden_size = x.shape
    
    # 初始化输出张量：(num_experts, expert_capacity, hidden_size)
    out = torch.zeros(
        (num_experts, expert_capacity, hidden_size),
        dtype=x.dtype,
        device=x.device
    )
    
    # 遍历每个专家
    for expert_idx in range(num_experts):
        # 计算当前专家负责的元素范围
        start_idx = 0 if expert_idx == 0 else bins[expert_idx - 1]
        end_idx = bins[expert_idx]
        num_elements = end_idx - start_idx
        
        # 遍历当前专家负责的每个元素
        for pos_in_expert in range(num_elements):
            # 超过专家容量则跳过（防止溢出）
            if pos_in_expert >= expert_capacity:
                continue
                
            # 计算全局索引
            global_idx = start_idx + pos_in_expert
            
            # 确保全局索引在有效范围内
            if global_idx >= indices.shape[0]:
                break
                
            # 获取输入矩阵x中的索引
            x_index = indices[global_idx] // top_k  # 每个token有top_k个索引
            
            # 获取输入元素
            input_element = x[x_index]
            
            # 应用权重（如果提供）
            if weights is not None:
                input_element = input_element * weights[global_idx]
            
            # 将元素存储到当前专家的对应位置
            out[expert_idx, pos_in_expert] = input_element
    
    return out


def binned_scatter(x, indices, weights, bins, top_k):
    """
    纯Python实现的binned_scatter函数
    
    参数:
        x: 专家处理后的输出，形状为(num_experts, expert_capacity, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)，指示每个元素对应的原始位置
        weights: 权重向量，形状为(tokens * top_k,)，可为None
        bins: 分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        out: 聚合后的结果，形状为(tokens, hidden_size)
    """
    # 输入验证
    assert_is_3d_tensor(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])  # 专家数量必须与bins长度匹配
    
    if weights is not None:
        assert_is_vector(weights)
        assert_equal(indices.shape[0], weights.shape[0])
    
    # 解析维度信息
    num_experts, expert_capacity, hidden_size = x.shape
    total_elements = indices.shape[0]
    tokens = total_elements // top_k  # 计算原始token数量
    
    # 初始化中间张量，存储每个token的top_k个专家输出
    intermediate = torch.zeros(
        (tokens, top_k, hidden_size),
        dtype=x.dtype,
        device=x.device
    )
    
    # 遍历每个专家
    for expert_idx in range(num_experts):
        # 确定当前专家负责的元素范围
        start = 0 if expert_idx == 0 else bins[expert_idx - 1]
        end = bins[expert_idx]
        expert_element_count = end - start
        
        # 遍历当前专家的输出元素
        for pos_in_expert in range(expert_element_count):
            # 跳过超过专家容量的元素
            if pos_in_expert >= expert_capacity:
                continue
                
            # 计算全局索引
            global_idx = start + pos_in_expert
            if global_idx >= total_elements:
                break  # 防止索引越界
                
            # 获取该元素对应的输出索引
            out_index = indices[global_idx]
            
            # 计算在中间张量中的位置
            token_idx = out_index // top_k  # 属于哪个token
            k_idx = out_index % top_k        # 是该token的第几个专家输出
            
            # 获取专家输出元素
            expert_output = x[expert_idx, pos_in_expert]
            
            # 应用权重（如果提供）
            if weights is not None:
                expert_output = expert_output * weights[global_idx]
            
            # 存储到中间张量
            intermediate[token_idx, k_idx] = expert_output
    
    # 聚合top_k维度的结果
    if top_k > 1:
        out = intermediate.sum(dim=1)  # 对每个token的多个专家输出求和
    else:
        out = intermediate.view(tokens, hidden_size)  # 直接调整形状
    
    return out


def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    """
    计算路由权重的梯度，用于混合专家模型的反向传播
    
    参数:
        x: 专家处理后的输出张量，形状为(num_experts, expert_capacity, hidden_size)
        grad: 最终输出的梯度张量，形状为(tokens, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)，指示每个元素对应的原始位置
        bins: 分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        wgrad: 路由权重的梯度，形状为(tokens * top_k,)
    """
    # 输入验证
    assert_tensor_dim(x, 3)       # 专家输出是3D张量
    assert_tensor_dim(grad, 2)    # 梯度是2D张量
    assert_tensor_dim(indices, 1) # 索引是1D向量
    assert_tensor_dim(bins, 1)    # 分箱信息是1D向量
    
    # 维度检查
    num_experts, expert_capacity, hidden_size = x.shape
    tokens, grad_hidden = grad.shape
    total_elements = indices.shape[0]
    
    assert bins.shape[0] == num_experts, "bins长度必须等于专家数量"
    assert grad_hidden == hidden_size, "梯度的特征维度必须与专家输出匹配"
    assert total_elements == tokens * top_k, "索引数量必须等于tokens * top_k"
    
    # 初始化权重梯度张量
    wgrad = torch.zeros(total_elements, dtype=x.dtype, device=x.device)
    
    # 遍历每个专家
    for expert_idx in range(num_experts):
        # 确定当前专家负责的元素范围
        start_idx = 0 if expert_idx == 0 else bins[expert_idx - 1]
        end_idx = bins[expert_idx]
        expert_element_count = end_idx - start_idx
        
        # 遍历当前专家的输出元素
        for pos_in_expert in range(expert_element_count):
            # 超出专家容量则跳过
            if pos_in_expert >= expert_capacity:
                continue
                
            # 计算全局索引
            global_idx = start_idx + pos_in_expert
            if global_idx >= total_elements:
                break  # 防止索引越界
                
            # 找到对应的原始token索引
            out_index = indices[global_idx]
            token_idx = out_index // top_k  # 计算属于哪个token
            
            # 验证token索引有效性
            if token_idx >= tokens:
                raise IndexError(f"Token index {token_idx} out of bounds for {tokens} tokens")
            
            # 获取专家输出和对应的梯度
            expert_output = x[expert_idx, pos_in_expert]  # 形状: (hidden_size,)
            token_gradient = grad[token_idx]              # 形状: (hidden_size,)
            
            # 计算权重梯度：专家输出与梯度的点积
            weight_gradient = torch.sum(expert_output * token_gradient)
            
            # 存储权重梯度
            wgrad[global_idx] = weight_gradient
    
    return wgrad


def padded_gather(x, indices, weights, bins, padded_bins, expert_capacity, top_k):
    """
    带填充支持的gather函数，用于混合专家模型中收集输入到各个专家
    
    参数:
        x: 输入特征矩阵，形状为(tokens, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        weights: 权重向量，形状为(tokens * top_k,)，可为None
        bins: 分箱累积计数向量，形状为(num_experts,)，实际元素数量
        padded_bins: 填充后的分箱累积计数向量，形状为(num_experts,)，包含填充
        expert_capacity: 每个专家的容量（包含填充）
        top_k: 每个token选择的专家数量
        
    返回:
        out: 按专家分箱的输出张量（含填充），形状为(num_experts, expert_capacity, hidden_size)
    """
    # 输入验证
    assert_tensor_dim(x, 2)       # 输入是2D张量
    assert_tensor_dim(indices, 1) # 索引是1D向量
    assert_tensor_dim(bins, 1)    # 分箱信息是1D向量
    assert_tensor_dim(padded_bins, 1)  # 填充分箱信息是1D向量
    
    # 维度检查
    tokens, hidden_size = x.shape
    num_experts = bins.shape[0]
    total_elements = indices.shape[0]
    
    assert padded_bins.shape[0] == num_experts, "padded_bins长度必须等于专家数量"
    assert total_elements == tokens * top_k, "索引数量必须等于tokens * top_k"
    assert bins[-1] <= total_elements, "bins最后一个元素不能超过总元素数"
    assert padded_bins[-1] <= num_experts * expert_capacity, "padded_bins超出专家总容量"
    
    # 初始化输出张量（含填充）
    out = torch.zeros(
        (num_experts, expert_capacity, hidden_size),
        dtype=x.dtype,
        device=x.device
    )
    
    # 遍历每个专家
    for expert_idx in range(num_experts):
        # 计算当前专家的实际元素范围（不含填充）
        start = 0 if expert_idx == 0 else bins[expert_idx - 1]
        end = bins[expert_idx]
        actual_elements = end - start
        
        # 计算填充后的起始位置（用于输出索引）
        padded_start = 0 if expert_idx == 0 else padded_bins[expert_idx - 1]
        
        # 遍历当前专家的实际元素
        for pos_in_bin in range(actual_elements):
            # 计算在专家输出中的位置（包含填充偏移）
            pos_in_expert = padded_start + pos_in_bin
            if pos_in_expert >= expert_capacity:
                continue  # 超过专家容量则跳过
                
            # 计算全局索引
            global_idx = start + pos_in_bin
            if global_idx >= total_elements:
                break  # 防止索引越界
                
            # 获取输入特征的索引
            x_idx = indices[global_idx] // top_k  # 每个token有top_k个索引
            
            # 获取输入元素
            input_element = x[x_idx]
            
            # 应用权重（如果提供）
            if weights is not None:
                input_element = input_element * weights[global_idx]
            
            # 存储到输出张量
            out[expert_idx, pos_in_expert] = input_element
        
        # 填充部分保持为0（已在初始化时设置）
    
    return out


def padded_scatter(x, indices, weights, bins, padded_bins, top_k):
    """
    带填充支持的scatter函数，用于混合专家模型中聚合专家输出
    
    参数:
        x: 专家处理后的输出，形状为(num_experts, expert_capacity, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        weights: 权重向量，形状为(tokens * top_k,)，可为None
        bins: 分箱累积计数向量（不含填充），形状为(num_experts,)
        padded_bins: 填充后的分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        out: 聚合后的结果，形状为(tokens, hidden_size)
    """
    # 输入验证
    assert_tensor_dim(x, 3)       # 专家输出是3D张量
    assert_tensor_dim(indices, 1) # 索引是1D向量
    assert_tensor_dim(bins, 1)    # 分箱信息是1D向量
    assert_tensor_dim(padded_bins, 1)  # 填充分箱信息是1D向量
    
    # 维度检查
    num_experts, expert_capacity, hidden_size = x.shape
    total_elements = indices.shape[0]
    tokens = total_elements // top_k  # 计算原始token数量
    
    assert bins.shape[0] == num_experts, "bins长度必须等于专家数量"
    assert padded_bins.shape[0] == num_experts, "padded_bins长度必须等于专家数量"
    assert bins[-1] <= total_elements, "bins最后一个元素不能超过总元素数"
    
    if weights is not None:
        assert_tensor_dim(weights, 1)
        assert weights.shape[0] == total_elements, "权重数量必须与总元素数匹配"
    
    # 初始化中间张量，存储每个token的top_k个专家输出
    intermediate = torch.zeros(
        (tokens, top_k, hidden_size),
        dtype=x.dtype,
        device=x.device
    )
    
    # 遍历每个专家
    for expert_idx in range(num_experts):
        # 计算当前专家的实际元素范围（不含填充）
        start = 0 if expert_idx == 0 else bins[expert_idx - 1]
        end = bins[expert_idx]
        actual_elements = end - start
        
        # 计算填充后的起始位置（用于定位专家输出）
        padded_start = 0 if expert_idx == 0 else padded_bins[expert_idx - 1]
        
        # 遍历当前专家的实际元素（忽略填充部分）
        for pos_in_bin in range(actual_elements):
            # 计算在专家输出中的位置（考虑填充偏移）
            pos_in_expert = padded_start + pos_in_bin
            if pos_in_expert >= expert_capacity:
                continue  # 超过专家容量则跳过
                
            # 计算全局索引
            global_idx = start + pos_in_bin
            if global_idx >= total_elements:
                break  # 防止索引越界
                
            # 获取该元素对应的输出索引
            out_index = indices[global_idx]
            
            # 计算在中间张量中的位置
            token_idx = out_index // top_k  # 对应哪个token
            k_idx = out_index % top_k        # 是该token的第几个专家输出
            
            # 获取专家输出元素（跳过填充部分）
            expert_output = x[expert_idx, pos_in_expert]
            
            # 应用权重（如果提供）
            if weights is not None:
                expert_output = expert_output * weights[global_idx]
            
            # 存储到中间张量
            intermediate[token_idx, k_idx] = expert_output
    
    # 聚合top_k维度的结果
    if top_k > 1:
        out = intermediate.sum(dim=1)  # 对每个token的多个专家输出求和
    else:
        out = intermediate.view(tokens, hidden_size)  # 直接调整形状
    
    return out


def padded_scatter_wgrad(x, grad, indices, bins, padded_bins, top_k):
    """
    带填充支持的权重梯度计算函数，用于混合专家模型反向传播
    
    参数:
        x: 专家处理后的输出，形状为(num_experts, expert_capacity, hidden_size)
        grad: 最终输出的梯度，形状为(tokens, hidden_size)
        indices: 索引向量，形状为(tokens * top_k,)
        bins: 分箱累积计数向量（不含填充），形状为(num_experts,)
        padded_bins: 填充后的分箱累积计数向量，形状为(num_experts,)
        top_k: 每个token选择的专家数量
        
    返回:
        wgrad: 路由权重的梯度，形状为(tokens * top_k,)
    """
    # 输入验证
    assert_tensor_dim(x, 3)       # 专家输出是3D张量
    assert_tensor_dim(grad, 2)    # 梯度是2D张量
    assert_tensor_dim(indices, 1) # 索引是1D向量
    assert_tensor_dim(bins, 1)    # 分箱信息是1D向量
    assert_tensor_dim(padded_bins, 1)  # 填充分箱信息是1D向量
    
    # 维度检查
    num_experts, expert_capacity, hidden_size = x.shape
    total_elements = indices.shape[0]
    tokens, grad_hidden = grad.shape
    
    assert bins.shape[0] == num_experts, "bins长度必须等于专家数量"
    assert padded_bins.shape[0] == num_experts, "padded_bins长度必须等于专家数量"
    assert grad_hidden == hidden_size, "梯度特征维度必须与专家输出匹配"
    assert total_elements == tokens * top_k, "索引数量必须等于tokens * top_k"
    assert bins[-1] <= total_elements, "bins最后一个元素不能超过总元素数"
    
    # 初始化权重梯度张量
    wgrad = torch.zeros(total_elements, dtype=x.dtype, device=x.device)
    
    # 遍历每个专家
    for expert_idx in range(num_experts):
        # 计算当前专家的实际元素范围（不含填充）
        start = 0 if expert_idx == 0 else bins[expert_idx - 1]
        end = bins[expert_idx]
        actual_elements = end - start
        
        # 计算填充后的起始位置（用于定位专家输出）
        padded_start = 0 if expert_idx == 0 else padded_bins[expert_idx - 1]
        
        # 遍历当前专家的实际元素（忽略填充部分）
        for pos_in_bin in range(actual_elements):
            # 计算在专家输出中的位置（考虑填充偏移）
            pos_in_expert = padded_start + pos_in_bin
            if pos_in_expert >= expert_capacity:
                continue  # 超过专家容量则跳过
                
            # 计算全局索引
            global_idx = start + pos_in_bin
            if global_idx >= total_elements:
                break  # 防止索引越界
                
            # 获取该元素对应的原始token索引
            out_index = indices[global_idx]
            token_idx = out_index // top_k  # 对应哪个token
            
            # 验证token索引有效性
            if token_idx >= tokens:
                raise IndexError(f"Token index {token_idx} out of bounds for {tokens} tokens")
            
            # 获取专家输出和对应的梯度
            expert_output = x[expert_idx, pos_in_expert]  # 形状: (hidden_size,)
            token_gradient = grad[token_idx]              # 形状: (hidden_size,)
            
            # 计算权重梯度：专家输出与梯度的点积
            weight_gradient = torch.sum(expert_output * token_gradient)
            
            # 存储权重梯度
            wgrad[global_idx] = weight_gradient
    
    return wgrad
