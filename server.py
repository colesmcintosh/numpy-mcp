# server.py
from typing import List, Union, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Numpy")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a (int): First number to add
        b (int): Second number to add
        
    Returns:
        int: Sum of the two numbers
    """
    return a + b

@mcp.tool()
def matrix_multiply(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> List[List[float]]:
    """Multiply two matrices using NumPy.
    
    Args:
        matrix_a (List[List[float]]): First input matrix
        matrix_b (List[List[float]]): Second input matrix
        
    Returns:
        List[List[float]]: Result of matrix multiplication
        
    Raises:
        ValueError: If matrices have incompatible dimensions
    """
    try:
        result = np.matmul(np.array(matrix_a), np.array(matrix_b))
        return result.tolist()
    except ValueError as e:
        raise ValueError(f"Matrix multiplication failed: {str(e)}") from e

@mcp.tool()
def statistical_analysis(
    data: List[float]
) -> dict[str, float]:
    """Perform statistical analysis on a dataset.
    
    Args:
        data (List[float]): Input data for analysis
        
    Returns:
        dict[str, float]: Dictionary containing statistical measures
    """
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

@mcp.tool()
def eigen_decomposition(
    matrix: List[List[float]]
) -> Tuple[List[float], List[List[float]]]:
    """Compute eigenvalues and eigenvectors of a square matrix.
    
    Args:
        matrix (List[List[float]]): Input square matrix
        
    Returns:
        Tuple[List[float], List[List[float]]]: Eigenvalues and eigenvectors
        
    Raises:
        ValueError: If matrix is not square or computation fails
    """
    try:
        arr = np.array(matrix)
        eigenvalues, eigenvectors = np.linalg.eig(arr)
        return eigenvalues.tolist(), eigenvectors.tolist()
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Eigendecomposition failed: {str(e)}") from e

@mcp.tool()
def polynomial_fit(
    x: List[float], 
    y: List[float], 
    degree: int = 2
) -> List[float]:
    """Fit a polynomial to the given data points.
    
    Args:
        x (List[float]): X coordinates
        y (List[float]): Y coordinates
        degree (int, optional): Degree of polynomial. Defaults to 2.
        
    Returns:
        List[float]: Coefficients of the fitted polynomial
    """
    coefficients = np.polyfit(x, y, degree)
    return coefficients.tolist()