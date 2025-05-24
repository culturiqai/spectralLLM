# Contributing to SpectralLLM

We welcome contributions to SpectralLLM! This document provides guidelines for contributing to the project.

## 📋 Types of Contributions

We welcome the following types of contributions:

- **Bug Reports**: Found a bug? Please report it!
- **Feature Requests**: Have an idea for a new feature?
- **Code Contributions**: Bug fixes, new features, optimizations
- **Documentation**: Improvements to docs, examples, tutorials
- **Testing**: Additional test cases, performance benchmarks
- **Research**: Novel spectral processing techniques

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/culturiqai/spectralllm.git
   cd spectralllm
   ```

2. **Create a development environment**:
   ```bash
   python -m venv spectralllm-dev
   source spectralllm-dev/bin/activate  # On Windows: spectralllm-dev\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,docs,examples]"
   ```

4. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

### Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking

Run these before submitting:
```bash
black spectralllm/
isort spectralllm/
flake8 spectralllm/
mypy spectralllm/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spectralllm --cov-report=html

# Run specific test categories
pytest tests/core/  # Core architecture tests
pytest tests/training/  # Training tests
pytest tests/validation/  # Validation tests
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Include both unit and integration tests
- Test edge cases and error conditions
- Aim for >90% code coverage

Example test structure:
```python
def test_spectral_embedding_shape():
    """Test that SpectralEmbedding outputs correct shapes"""
    config = Config(vocab_size=1000, embed_dim=256, harmonic_bases=32)
    embedding = SpectralEmbedding(config)
    
    input_ids = torch.randint(0, 1000, (2, 128))
    output = embedding(input_ids)
    
    assert output.shape == (2, 128, 256)
```

## 🔬 Architecture Guidelines

### Spectral Processing Principles

When contributing to the core architecture:

1. **Maintain Spectral Focus**: New components should primarily use frequency-domain operations
2. **Preserve O(n log n) Complexity**: Avoid introducing O(n²) operations
3. **Evolutionary Compatibility**: Ensure new components work with HRFEvo optimization
4. **Wavelet Compatibility**: Support multiple wavelet families and decomposition levels

### Code Organization

```
spectralllm/
├── core/           # Core architecture components
├── training/       # Training infrastructure  
├── validation/     # Validation and testing
├── evaluation/     # Benchmarking and evaluation
├── utils/          # Utilities and optimizations
└── demos/          # Demonstration scripts
```

## 📝 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def calculate_spectral_attention(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Calculate spectral attention using FFT operations.
    
    Args:
        query: Query tensor of shape (batch, seq_len, dim)
        key: Key tensor of shape (batch, seq_len, dim)
        
    Returns:
        Attention weights of shape (batch, seq_len, seq_len)
        
    Example:
        >>> query = torch.randn(2, 128, 256)
        >>> key = torch.randn(2, 128, 256)
        >>> attention = calculate_spectral_attention(query, key)
        >>> attention.shape
        torch.Size([2, 128, 128])
    """
```

### Adding Examples

When adding new features, include:
1. Usage example in the docstring
2. Integration test demonstrating the feature
3. Example script in `examples/` directory if appropriate

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - Python version
   - PyTorch version
   - Operating system
   - Hardware (GPU/CPU/Apple Silicon)

2. **Reproduction Steps**:
   - Minimal code example
   - Input data (if applicable)
   - Expected vs actual behavior

3. **Error Information**:
   - Full error traceback
   - Relevant log output

## 💡 Feature Requests

For feature requests, please provide:

1. **Use Case**: Why is this feature needed?
2. **Proposal**: How should it work?
3. **Spectral Alignment**: How does it fit with spectral processing principles?
4. **Implementation Ideas**: Any thoughts on implementation approach?

## 🔄 Pull Request Process

1. **Fork and Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement Changes**:
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Test Thoroughly**:
   ```bash
   pytest tests/
   black spectralllm/
   flake8 spectralllm/
   ```

4. **Submit Pull Request**:
   - Clear title and description
   - Link to related issues
   - Include test results
   - Request review from maintainers

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New functionality is tested
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] No breaking changes (or clearly documented)

## 🎯 Research Contributions

SpectralLLM is a research-oriented project. We especially welcome:

- **Novel Spectral Techniques**: New frequency-domain processing methods
- **Wavelet Innovations**: Advanced wavelet transform applications
- **Evolutionary Improvements**: Enhancements to HRFEvo optimization
- **Performance Optimizations**: Efficiency improvements maintaining spectral principles
- **Theoretical Analysis**: Mathematical analysis of spectral properties

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug Reports**: Create a GitHub Issue
- **Feature Requests**: Create a GitHub Issue with "enhancement" label
- **Email**: contact@culturiq.ai for major contributions or research collaborations

## 📄 License

By contributing to SpectralLLM, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for their contributions
- Research papers (for significant algorithmic contributions)

Thank you for contributing to SpectralLLM! 🌊 