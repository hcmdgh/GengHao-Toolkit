"""
提供对PyTorch和SciPy的稀疏矩阵的操作。
"""

from .imports import * 

from sklearn.preprocessing import normalize as sklearn_normalize

__all__ = [
    'convert_csr_matrix_to_csr_tensor',
    'convert_csr_tensor_to_csr_matrix',
    'convert_to_csr_tensor', 
    'sparse_matmul', 
    'L1_normalize',
    'L2_normalize',
]


def convert_csr_matrix_to_csr_tensor(csr_matrix: sp.csr_matrix,
                                     device: Optional[torch.device] = None) -> SparseTensor:
    N, M = csr_matrix.shape 
    
    csr_tensor = torch.sparse_csr_tensor(
        crow_indices = csr_matrix.indptr,
        col_indices = csr_matrix.indices,
        values = csr_matrix.data,
        size = [N, M],
        dtype = torch.float32,
        device = device, 
    )
    
    return csr_tensor 


def convert_csr_tensor_to_csr_matrix(csr_tensor: SparseTensor) -> sp.csr_matrix:
    N, M = csr_tensor.shape 

    data = csr_tensor.values().detach().cpu().numpy() 
    indices = csr_tensor.col_indices().detach().cpu().numpy() 
    indptr = csr_tensor.crow_indices().detach().cpu().numpy() 
    
    csr_matrix = sp.csr_matrix((data, indices, indptr), shape=[N, M], dtype=np.float32)
    
    return csr_matrix 


def convert_to_csr_tensor(inp,
                          device = None) -> SparseTensor:
    if isinstance(inp, np.ndarray):
        csr_tensor = torch.from_numpy(inp).to_sparse_csr() 
        
    elif isinstance(inp, sp.csr_matrix):
        csr_tensor = convert_csr_matrix_to_csr_tensor(inp)
        
    elif isinstance(inp, torch.Tensor):
        if inp.layout == torch.strided:
            csr_tensor = inp.detach().to_sparse_csr() 
        elif inp.layout == torch.sparse_csr:
            csr_tensor = inp
        else:
            raise TypeError
            
    else:
        raise TypeError 
    
    csr_tensor = csr_tensor.to(torch.float32).to(device)

    return csr_tensor 


def convert_to_csr_matrix(inp) -> sp.csr_matrix:
    if isinstance(inp, np.ndarray):
        csr_matrix = sp.csr_matrix(inp) 
        
    elif isinstance(inp, sp.csr_matrix):
        csr_matrix = inp
        
    elif isinstance(inp, torch.Tensor):
        if inp.layout == torch.strided:
            csr_matrix = sp.csr_matrix(inp.detach().cpu().numpy()) 
        elif inp.layout == torch.sparse_csr:
            csr_matrix = convert_csr_tensor_to_csr_matrix(inp) 
        else:
            raise TypeError 
            
    else:
        raise TypeError 
    
    csr_matrix = csr_matrix.astype(np.float32) 
    
    return csr_matrix 


def sparse_matmul(sparse_mat, 
                  dense_mat: FloatTensor) -> FloatTensor:
    """
    矩阵乘法，要求左侧为稀疏矩阵，右侧为稠密矩阵。
    """
    csr_tensor = convert_to_csr_tensor(sparse_mat)
    
    assert csr_tensor.ndim == dense_mat.ndim == 2 
    
    result = torch.mm(csr_tensor, dense_mat)
    
    return result 


def _normalize(mat,
               norm: str):
    """
    矩阵按行进行正则化操作。
    支持稠密矩阵和稀疏矩阵的4种格式：ndarray, Tensor, csr_matrix, Tensor(CSR)。
    输出格式与输入格式保持一致。
    """
    assert mat.ndim == 2 
    
    if isinstance(mat, ndarray):
        result = sklearn_normalize(mat, norm=norm).astype(np.float32)
        return result 
    
    elif isinstance(mat, Tensor) and mat.layout == torch.strided:
        device = mat.device 
        result = torch.from_numpy(
            sklearn_normalize(
                mat.detach().cpu().numpy(), 
                norm = norm,
            )
        ).to(torch.float32).to(device) 
        return result 
    
    elif isinstance(mat, sp.csr_matrix):
        result = sklearn_normalize(mat, norm=norm).astype(np.float32)
        return result 
    
    elif isinstance(mat, Tensor) and mat.layout == torch.sparse_csr:
        device = mat.device 
        result = convert_csr_matrix_to_csr_tensor(
            sklearn_normalize(
                convert_csr_tensor_to_csr_matrix(mat), 
                norm = norm,
            ),
            device = device, 
        ) 
        return result 
    
    else:
        raise TypeError 


def L1_normalize(mat):
    return _normalize(mat=mat, norm='l1')


def L2_normalize(mat):
    return _normalize(mat=mat, norm='l2')
