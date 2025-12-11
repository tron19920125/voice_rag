"""
Pytest 配置文件
"""

import pytest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def embedding_service():
    """提供 Embedding 服务实例"""
    from rag_utils import EmbeddingService
    return EmbeddingService()


@pytest.fixture(scope="session")
def reranking_service():
    """提供 Reranking 服务实例"""
    from rag_utils import RerankingService
    return RerankingService()


@pytest.fixture(scope="session")
def vector_index(embedding_service):
    """提供预加载的向量索引"""
    from rag_utils import VectorIndex
    from data.knowledge_base import DOCUMENTS

    index = VectorIndex(embedding_service)
    index.add_documents(DOCUMENTS)
    return index


@pytest.fixture(scope="session")
def knowledge_base():
    """提供知识库数据"""
    from data.knowledge_base import DOCUMENTS, TEST_QUERIES
    return {
        "documents": DOCUMENTS,
        "queries": TEST_QUERIES
    }
